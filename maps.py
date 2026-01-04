import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import binom
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 1. Noise Corrected Backbone Function
# ==========================================
def noise_corrected(table, undirected=False, return_self_loops=False, calculate_p_value=False):
    sys.stderr.write("Calculating NC score...\n")
    table = table.copy()
    
    # Calculate source sums
    src_sum = table.groupby(by="src")[["nij"]].sum()
    table = table.merge(src_sum, left_on="src", right_index=True, suffixes=("", "_src_sum"))
    
    # Calculate target sums
    trg_sum = table.groupby(by="trg")[["nij"]].sum()
    table = table.merge(trg_sum, left_on="trg", right_index=True, suffixes=("", "_trg_sum"))
    
    # Rename and calculate totals
    table.rename(columns={"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace=True)
    table["n.."] = table["nij"].sum()
    table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
    
    if calculate_p_value:
        table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
        return table[["src", "trg", "nij", "score"]]
    
    # Calculate Kappa and Score
    table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
    table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
    
    # Calculate variances and priors
    table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
    table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
    table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
    
    table["alpha_post"] = table["alpha_prior"] + table["nij"]
    table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
    table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
    table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
    
    table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
    table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2) 
    table["sdev_cij"] = table["variance_cij"] ** .5
    
    if not return_self_loops:
        table = table[table["src"] != table["trg"]]
    if undirected:
        table = table[table["src"] <= table["trg"]]
        
    return table[["src", "trg", "nij", "score", "sdev_cij"]]

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
# Direct loading without try/except blocks
airports = pd.read_csv("data/airports.csv")
routes = pd.read_csv("data/routes.csv")

# Replace missing continent for U.S. airports with "NA" (North America)
airports.loc[airports["iso_country"] == "US", "continent"] = airports.loc[airports["iso_country"] == "US", "continent"].fillna("NA")

# Drop airports missing essential info
airports = airports.dropna(subset=["id", "latitude_deg", "longitude_deg"])

# Clean route ID columns ('\N' means missing in OpenFlights data)
routes["Source airport ID"] = routes["Source airport ID"].replace("\\N", pd.NA)
routes["Destination airport ID"] = routes["Destination airport ID"].replace("\\N", pd.NA)

# Drop missing IDs and convert to int
routes = routes.dropna(subset=["Source airport ID", "Destination airport ID"])
routes["Source airport ID"] = routes["Source airport ID"].astype(int)
routes["Destination airport ID"] = routes["Destination airport ID"].astype(int)
airports["id"] = airports["id"].astype(int)

# Filter for NATO Countries
nato_countries = [
    "US","CA","IS","NO","DK","NL","BE","LU","FR","DE","IT","PT",
    "UK","ES","GR","TR","PL","CZ","HU","SK","SI","HR","BG","RO",
    "EE","LV","LT","AL","ME","MK","SE","FI"
]
airports_nato = airports[airports["iso_country"].isin(nato_countries)]
print("NATO airports:", len(airports_nato))

# Keep only routes where BOTH airports are NATO members
routes_nato = routes[
    routes["Source airport ID"].isin(airports_nato["id"]) &
    routes["Destination airport ID"].isin(airports_nato["id"])
]
print("NATO routes:", len(routes_nato))

# Group by source/destination to count flights (Weight)
route_weights = (
    routes_nato
    .groupby(["Source airport ID", "Destination airport ID"])
    .size()
    .reset_index(name="weight")
)
print("Unique weighted routes:", len(route_weights))

# ==========================================
# 3. Initial Graph Construction
# ==========================================
G = nx.DiGraph()

# Add NATO airports as nodes
for _, row in airports_nato.iterrows():
    G.add_node(
        row["id"],
        country=row["iso_country"],
        name=row["name"],
        lat=row["latitude_deg"],
        lon=row["longitude_deg"]
    )

# Add directed edges
for _, row in route_weights.iterrows():
    G.add_edge(
        row["Source airport ID"],
        row["Destination airport ID"],
        weight=row["weight"]
    )

# Remove isolated nodes
isolated_nodes = [n for n, d in G.degree() if d == 0]
print(f"Removing {len(isolated_nodes)} isolated airports")
G.remove_nodes_from(isolated_nodes)

# Component stats
largest_wcc = len(max(nx.weakly_connected_components(G), key=len))
largest_scc = len(max(nx.strongly_connected_components(G), key=len))
print("Largest weakly connected component size:", largest_wcc)
print("Largest strongly connected component size:", largest_scc)

# ==========================================
# 4. Backbone Extraction
# ==========================================
# Convert graph to DataFrame for the function
routes_df = pd.DataFrame([
    {"src": u, "trg": v, "nij": data.get("weight", 1.0)}
    for u, v, data in G.edges(data=True)
])

# Run Noise-Corrected Backbone
nc_df = noise_corrected(routes_df, undirected=False)
print("NCB computed.")

# Filter significant edges (Score > 0.95)
backbone_edges = nc_df[nc_df["score"] > 0.95]
print(f"Edges retained: {len(backbone_edges)} ({len(backbone_edges)/len(routes_df):.1%} of original)")

# Create Backbone Graph
backboned_G = nx.from_pandas_edgelist(
    backbone_edges, 
    source="src", 
    target="trg", 
    edge_attr=["score", "nij"], 
    create_using=nx.DiGraph()
)

# Copy node attributes (lat/lon) from original G to backboned_G
for n, d in G.nodes(data=True):
    if n in backboned_G:
        backboned_G.nodes[n].update(d)

print(f"Backbone: {len(backboned_G.nodes())} nodes, {len(backboned_G.edges())} edges")


# ==========================================
# 5. Visualization (Cartopy) - FINAL FIX
# ==========================================
# We use a wide, short figure (18x6) to match the "wide" shape of the Atlantic/US map
# This prevents the vertical "squashing" distortion.
plt.figure(figsize=(18, 6))

# PlateCarree is the standard lat/lon projection
ax = plt.axes(projection=ccrs.PlateCarree())

# --- COLORS ---
# Light greenish theme as requested
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#004d00')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#004d00')
ax.add_feature(cfeature.LAND, facecolor='#d5e8d4') 
ax.add_feature(cfeature.OCEAN, facecolor='#e0f2f1')

# --- DATA ---
pos = {n: (d['lon'], d['lat']) for n, d in backboned_G.nodes(data=True)}

# Draw Edges (Very thin lines)
nx.draw_networkx_edges(
    backboned_G, pos, ax=ax,
    width=0.1,             # Ultra thin
    alpha=0.6, 
    edge_color='forestgreen',
    arrows=True,
    arrowstyle='-|>',
    arrowsize=4
)

# Draw Nodes
nx.draw_networkx_nodes(
    backboned_G, pos, ax=ax,
    node_size=10, 
    node_color='darkgreen', 
    alpha=0.9
)

# --- STRICT CROP ---
# West: -170 (Includes Hawaii)
# East: 40 (Cuts off Russia/Asia, stops at Turkey/Eastern Europe)
# South: 15 (Mexico/Hawaii latitude)
# North: 80 (Greenland/Arctic)
ax.set_extent([-170, 40, 15, 80], crs=ccrs.PlateCarree())

plt.tight_layout()
plt.show()
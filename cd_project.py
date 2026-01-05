import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm, binom
from community import community_louvain   # pip install python-louvain
from collections import Counter, defaultdict
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# 🧩 1. Import and Load Data
# =====================================================
airports = pd.read_csv("data/airports.csv")
routes = pd.read_csv("data/routes.csv")

print("✅ Data loaded successfully")

# =====================================================
# 🧹 2. Clean and Prepare Data
# =====================================================
# Replace missing continent for U.S. airports with "NA"
airports.loc[airports["iso_country"] == "US", "continent"] = airports.loc[airports["iso_country"] == "US", "continent"].fillna("NA")

# Drop airports missing important info
airports = airports.dropna(subset=["id", "latitude_deg", "longitude_deg"])

# Clean route ID columns ('\N' means missing)
routes["Source airport ID"] = routes["Source airport ID"].replace("\\N", pd.NA)
routes["Destination airport ID"] = routes["Destination airport ID"].replace("\\N", pd.NA)

# Drop missing IDs and convert to int
routes = routes.dropna(subset=["Source airport ID", "Destination airport ID"])
routes["Source airport ID"] = routes["Source airport ID"].astype(int)
routes["Destination airport ID"] = routes["Destination airport ID"].astype(int)
airports["id"] = airports["id"].astype(int)

# =====================================================
# 🌍 3. Filter for NATO Countries
# =====================================================
nato_countries = [
    "US","CA","IS","NO","DK","NL","BE","LU","FR","DE","IT","PT",
    "GB","ES","GR","TR","PL","CZ","HU","SK","SI","HR","BG","RO",
    "EE","LV","LT","AL","ME","MK", "FI", "SE"
]

# European subset for the specific map later
europe_nato = [
    "IS","NO","DK","NL","BE","LU","FR","DE","IT","PT",
    "UK","ES","GR","TR","PL","CZ","HU","SK","SI","HR","BG","RO",
    "EE","LV","LT","AL","ME","MK", "FI", "SE"
]

airports_nato = airports[airports["iso_country"].isin(nato_countries)].copy()
print("NATO airports:", len(airports_nato))

nato_ids = set(airports_nato["id"].astype(int))
routes_nato = routes[
    routes["Source airport ID"].isin(nato_ids) &
    routes["Destination airport ID"].isin(nato_ids)
].copy()
print("NATO routes:", len(routes_nato))

# =====================================================
# 💪 4. Add Edge Weights Based on Route Counts
# =====================================================
wdir = (routes_nato
        .groupby(["Source airport ID","Destination airport ID"], as_index=False)
        .size()
        .rename(columns={"size":"weight"}))

# =====================================================
# 🕸️ 5. Build Weighted Network (Directed)
# =====================================================
G_dir = nx.DiGraph()

# nodes with metadata
for _, r in airports_nato.iterrows():
    G_dir.add_node(int(r["id"]), name=r["name"], country=r["iso_country"],
                   lat=r.get("latitude_deg", np.nan), lon=r.get("longitude_deg", np.nan), iata=r.get("iata_code", ""))

# edges
for _, r in wdir.iterrows():
    G_dir.add_edge(int(r["Source airport ID"]), int(r["Destination airport ID"]),
                   weight=int(r["weight"]))

print("Directed NATO graph built:",
      "nodes=", G_dir.number_of_nodes(), "edges=", G_dir.number_of_edges())

# =====================================================
# 🧬 6. Noise-Corrected Backbone (Z-Score)
# =====================================================

# 1) Build edge table
E = pd.DataFrame(
    [(u, v, d.get("weight", 1.0)) for u, v, d in G_dir.edges(data=True)],
    columns=["src", "trg", "nij"]
)

# 2) NC Function
def noise_corrected(table, undirected=False, return_self_loops=False, calculate_p_value=False):
    sys.stderr.write("Calculating NC score...\n")
    table = table.copy()
    src_sum = table.groupby(by="src").sum()[["nij"]]
    table = table.merge(src_sum, left_on="src", right_index=True, suffixes=("", "_src_sum"))
    trg_sum = table.groupby(by="trg").sum()[["nij"]]
    table = table.merge(trg_sum, left_on="trg", right_index=True, suffixes=("", "_trg_sum"))
    table.rename(columns={"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace=True)
    table["n.."] = table["nij"].sum()
    table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
    
    if calculate_p_value:
        table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
        return table[["src", "trg", "nij", "score"]]
        
    table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
    table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
    
    # Variance calculations
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

# 3) Score edges
nc = noise_corrected(E, undirected=False, return_self_loops=False, calculate_p_value=False)

# 4) Threshold by Z-score (alpha=0.05 -> z ~ 1.645)
alpha = 0.05
z_alpha = float(norm.ppf(1 - alpha))
mask = (nc["sdev_cij"] > 0) & ((nc["score"] / nc["sdev_cij"]) >= z_alpha)
nc_keep = nc[mask]

# 5) Build the directed backbone graph
G_ACTIVE = nx.DiGraph()
G_ACTIVE.add_nodes_from(G_dir.nodes(data=True))

for _, r in nc_keep.iterrows():
    u, v = int(r["src"]), int(r["trg"])
    w = float(r["nij"])
    G_ACTIVE.add_edge(u, v, weight=w, nc_score=float(r["score"]), z=float(r["score"]/r["sdev_cij"]))

# Remove isolated nodes
isolated_nodes = [n for n, d in G_ACTIVE.degree() if d == 0]
print(f"Removing {len(isolated_nodes)} isolated airports")
G_ACTIVE.remove_nodes_from(isolated_nodes) 
print(f"Final Backbone: {G_ACTIVE.number_of_nodes()} nodes, {G_ACTIVE.number_of_edges()} edges")

# =====================================================
# 🏘️ 7. Community Detection (Louvain)
# =====================================================
# Louvain requires an UNDIRECTED graph.
G_undirected = G_ACTIVE.to_undirected()
partition = community_louvain.best_partition(G_undirected, weight="weight")

# Organize communities
communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

# Filter small communities for analysis
MIN_SIZE = 10
large_communities = {cid: nodes for cid, nodes in communities.items() if len(nodes) >= MIN_SIZE}
sorted_comms = sorted(large_communities.items(), key=lambda x: len(x[1]), reverse=True)
top5 = sorted_comms[:5]

print(f"\nDetected {len(communities)} total communities.")
print(f"Analyzing {len(large_communities)} large communities (size >= {MIN_SIZE}).")

# =====================================================
# 🔍 8. Analysis: Dominant Airlines
# =====================================================

def get_dominant_airline(community_nodes, routes_df):
    """
    Finds the airline with the most routes strictly WITHIN this community.
    """
    # Filter routes where Source AND Dest are in this community
    internal_routes = routes_df[
        routes_df["Source airport ID"].isin(community_nodes) &
        routes_df["Destination airport ID"].isin(community_nodes)
    ]
    
    if internal_routes.empty:
        return "None", 0
    
    # Count frequency of each airline code
    counts = internal_routes["Airline"].value_counts()
    top_airline_code = counts.idxmax()
    count = counts.max()
    
    return top_airline_code, count

for cid, nodes in top5:
    print(f"COMMUNITY {cid} (Size: {len(nodes)})")
    
    # Top Countries
    node_subset = airports_nato[airports_nato["id"].isin(nodes)]
    top_countries = node_subset["iso_country"].value_counts().head(3)
    print(f"Top Countries: {dict(top_countries)}")
    
    # Dominant Airline
    airline, count = get_dominant_airline(nodes, routes_nato)
    print(f"Dominant Airline: {airline} ({count} routes)")

# =====================================================
# 🗺️ 9. Visualization 1: ALL NATO (Greenish Theme)
# =====================================================
community_ids = sorted(set(partition.values()))
cmap = plt.get_cmap("tab20")
color_lookup = {cid: cmap(i % 20) for i, cid in enumerate(community_ids)}

pos = {n: (d['lon'], d['lat']) for n, d in G_ACTIVE.nodes(data=True)}

plt.figure(figsize=(18, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-170, 40, 15, 80], crs=ccrs.PlateCarree())

# Style
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#004d00')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#004d00')
ax.add_feature(cfeature.LAND, facecolor='#d5e8d4') 
ax.add_fea
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================
# 1. LOAD DATA
# =====================================================
airports = pd.read_csv("data/airports.csv")
routes = pd.read_csv("data/routes.csv")

# Clean Airports
airports.loc[airports["iso_country"] == "US", "continent"] = airports.loc[airports["iso_country"] == "US", "continent"].fillna("NA")
airports = airports.dropna(subset=["id", "latitude_deg", "longitude_deg"])
airports["id"] = airports["id"].astype(int)

# Clean Routes
routes = routes[routes["Source airport ID"] != "\\N"]
routes = routes[routes["Destination airport ID"] != "\\N"]
routes["Source airport ID"] = routes["Source airport ID"].astype(int)
routes["Destination airport ID"] = routes["Destination airport ID"].astype(int)

# =====================================================
# 2. FILTER FOR NATO COUNTRIES
# =====================================================
nato_countries = [
    "US","CA","IS","NO","DK","NL","BE","LU","FR","DE","IT","PT",
    "UK","ES","GR","TR","PL","CZ","HU","SK","SI","HR","BG","RO",
    "EE","LV","LT","AL","ME","MK", "FI", "SE"
]

# Filter Airports
airports_nato = airports[airports["iso_country"].isin(nato_countries)].copy()
valid_nato_ids = set(airports_nato["id"])

# Filter Routes (Both Source AND Dest must be NATO)
routes_nato = routes[
    routes["Source airport ID"].isin(valid_nato_ids) &
    routes["Destination airport ID"].isin(valid_nato_ids)
].copy()

# =====================================================
# 3. BUILD GRAPH
# =====================================================
G = nx.DiGraph()

# Add Nodes
for _, r in airports_nato.iterrows():
    G.add_node(int(r["id"]), country=r["iso_country"], name=r["name"])

# Add Edges
edge_counts = routes_nato.groupby(["Source airport ID", "Destination airport ID"]).size().reset_index(name="weight")
for _, r in edge_counts.iterrows():
    G.add_edge(int(r["Source airport ID"]), int(r["Destination airport ID"]), weight=int(r["weight"]))

# Remove isolated nodes (Airports with no flights)
G.remove_nodes_from(list(nx.isolates(G)))

# PRINT CLEAR STATS
print("=" * 40)
print(f"DOTS (Airports):  {G.number_of_nodes()}")
print(f"LINES (Routes):   {G.number_of_edges()}")
print("=" * 40)

# =====================================================
# 4. VISUALIZATION
# =====================================================
plt.figure(figsize=(14, 14))

print("Calculating layout...")
pos = nx.spring_layout(G, k=0.10, iterations=50, seed=42)

# Draw Edges (The 44,000 lines)
# We make them thin and transparent so they don't block the view
nx.draw_networkx_edges(
    G, pos,
    alpha=1,         
    width=0.1,
    edge_color='green',
    arrows=False
)

# Draw Nodes (The 1,800 dots)
nx.draw_networkx_nodes(
    G, pos,
    node_size=15,
    node_color='black',
    alpha=0.8,
    linewidths=0
)

# Title showing the exact counts
plt.title(f"NATO Network: {G.number_of_nodes()} Airports (Dots) | {G.number_of_edges()} Routes (Lines)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()
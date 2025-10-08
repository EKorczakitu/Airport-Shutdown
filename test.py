# ---- NATO airport network EDA using ROUTES ROWS AS EDGES (MultiDiGraph) ----
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# NATO ISO-2
NATO_ISO2 = {
    "AL","BE","BG","HR","CZ","DK","EE","FI","FR","DE","GR","HU","IS","IT",
    "LV","LT","LU","ME","NL","MK","NO","PL","PT","RO","SK","SI","ES","SE",
    "TR","GB","CA","US"
}

# ---------- Load ----------
airports_raw = pd.read_csv("data/airports.csv")
routes_raw   = pd.read_csv("data/routes.csv")

# ---------- Airports: flexible columns ----------
acols = {c.lower(): c for c in airports_raw.columns}
def pick(*cands):
    for c in cands:
        if c in acols: return acols[c]
    return None

col_ident     = pick("ident","icao","gps_code","icao_code")
col_iata      = pick("iata_code","iata")
col_country   = pick("iso_country","country_iso","country")
col_type      = pick("type")
col_name      = pick("name")
col_continent = pick("continent")

use_cols = [c for c in [col_name,col_type,col_country,col_iata,col_ident,col_continent] if c is not None]
airports = airports_raw[use_cols].copy()
std_cols = ["name","atype","country","iata","icao","continent"]
airports.columns = std_cols[:len(use_cols)]
for need in std_cols:
    if need not in airports.columns:
        airports[need] = np.nan

# Clean + fill US continent -> NA
for c in ["iata","icao","country","continent"]:
    airports[c] = airports[c].astype(str).str.upper().str.strip()

mask_us_missing = (airports["country"] == "US") & (
    airports["continent"].isna() | (airports["continent"] == "") | (airports["continent"] == "NAN")
)
airports.loc[mask_us_missing, "continent"] = "NA"

# Drop remaining critical missings AFTER fill
airports.replace({"NAN": np.nan, "": np.nan}, inplace=True)
airports = airports.dropna(subset=["country","continent"])

# NATO only
airports = airports[airports["country"].isin(NATO_ISO2)].copy()

# Canonical node = IATA if present else ICAO
airports["node"] = np.where(airports["iata"].notna() & (airports["iata"] != ""),
                            airports["iata"], airports["icao"])
airports.replace({"": np.nan}, inplace=True)
airports = airports.dropna(subset=["node"])

# De-duplicate nodes (prefer larger airports)
size_rank = {"LARGE_AIRPORT":0,"MEDIUM_AIRPORT":1,"SMALL_AIRPORT":2}
airports["_rank"] = airports["atype"].str.upper().map(size_rank).fillna(3)
airports_u = (airports.sort_values(["node","_rank"])
                     .drop_duplicates(subset="node", keep="first")
                     .drop(columns="_rank"))

# Build alias: both IATA and ICAO -> canonical node
alias = {}
for _, r in airports_u.iterrows():
    if isinstance(r["iata"], str) and r["iata"]: alias[r["iata"]] = r["node"]
    if isinstance(r["icao"], str) and r["icao"]: alias[r["icao"]] = r["node"]

# ---------- Routes: flexible columns ----------
rcols = {c.lower(): c for c in routes_raw.columns}
src_col = rcols.get("source airport") or rcols.get("src") or rcols.get("source")
dst_col = rcols.get("destination airport") or rcols.get("dst") or rcols.get("destination")
airline_col = rcols.get("airline")

use_rcols = [c for c in [airline_col, src_col, dst_col] if c is not None]
routes = routes_raw[use_rcols].copy()
if airline_col is None:
    routes["Airline"] = "UNK"
else:
    routes.rename(columns={airline_col:"Airline"}, inplace=True)
routes.rename(columns={src_col:"src_raw", dst_col:"dst_raw"}, inplace=True)

routes["src_raw"] = routes["src_raw"].astype(str).str.upper().str.strip()
routes["dst_raw"] = routes["dst_raw"].astype(str).str.upper().str.strip()

# Map endpoints -> canonical NATO nodes; KEEP EVERY ROW as an edge
routes["src"] = routes["src_raw"].map(alias)
routes["dst"] = routes["dst_raw"].map(alias)
routes = routes[routes["src"].notna() & routes["dst"].notna() & (routes["src"] != routes["dst"])].copy()

# ---------- Build MultiDiGraph DIRECTLY FROM ROUTES ROWS ----------
Gm = nx.MultiDiGraph()
meta = airports_u.set_index("node")[["name","atype","country","continent"]].to_dict(orient="index")

# Add nodes first (with attributes)
Gm.add_nodes_from((n, meta.get(n, {})) for n in airports_u["node"].unique())

# Add one edge PER ROUTE ROW (no collapsing)
for _, r in routes.iterrows():
    Gm.add_edge(r["src"], r["dst"], airline=r["Airline"])

# ---------- Quick stats ----------
n = Gm.number_of_nodes()
m = Gm.number_of_edges()
dens = nx.density(nx.DiGraph(Gm))  # density on simple digraph view
wccs = list(nx.weakly_connected_components(nx.DiGraph(Gm)))
lcc_size = max((len(c) for c in wccs), default=0)
lcc_frac = (lcc_size / n) if n else 0.0

print("Diagnostics:")
print(f"  Airports NATO (unique nodes): {len(airports_u):,}")
print(f"  Alias size: {len(alias):,}")
print(f"  Routes raw: {len(routes_raw):,}")
print(f"  Routes kept as edges (NATO→NATO): {len(routes):,}")

print("\n=== NATO network stats (MultiDiGraph from routes) ===")
print(f"Nodes: {n:,}")
print(f"Edges (route rows): {m:,}")
print(f"Density (simple digraph view): {dens:.6f}")
print(f"Weakly CCs: {len(wccs)} | LCC size: {lcc_size:,} ({lcc_frac:.2%})")

# ---------- Degrees (now properly count multiplicity) ----------
# For MultiDiGraph, out_degree()/in_degree() without weight counts parallel edges.
out_deg = dict(Gm.out_degree())
in_deg  = dict(Gm.in_degree())
deg     = {k: out_deg.get(k,0) + in_deg.get(k,0) for k in Gm.nodes()}

# Also compute weighted strengths by collapsing to simple DiGraph with weights
G = nx.DiGraph()
for u, v, _k, _d in Gm.edges(keys=True, data=True):
    if G.has_edge(u, v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, weight=1)

out_strength = dict(G.out_degree(weight="weight"))
in_strength  = dict(G.in_degree(weight="weight"))

# ---------- Top tables ----------
def top_k(series_dict, name, k=15):
    if not series_dict:
        return pd.DataFrame()
    df = (pd.DataFrame(series_dict.items(), columns=["node", name])
            .sort_values(name, ascending=False).head(k))
    return df.join(airports_u.set_index("node")[["name","country","continent"]], on="node")

top_out   = top_k(out_deg,      "out_degree")      # counts route rows
top_in    = top_k(in_deg,       "in_degree")
top_deg   = top_k(deg,          "degree")
top_sout  = top_k(out_strength, "out_strength")    # weighted by number of rows
top_sin   = top_k(in_strength,  "in_strength")

print("\nTop 15 by out-degree (route rows):")
display(top_out)

# ---------- Plots ----------
def plot_top_bar(df, value_col, title):
    if df.empty:
        print(f"[Plot skipped: {title} empty]"); return
    plt.figure(figsize=(9,5))
    plt.bar(df["node"].astype(str), df[value_col].values)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel(value_col); plt.title(title)
    plt.tight_layout(); plt.show()

plot_top_bar(top_out,  "out_degree",   "NATO: Top 15 by out-degree (rows)")
plot_top_bar(top_in,   "in_degree",    "NATO: Top 15 by in-degree (rows)")
plot_top_bar(top_deg,  "degree",       "NATO: Top 15 by total degree (rows)")
plot_top_bar(top_sout, "out_strength", "NATO: Top 15 by out-strength (weighted)")
plot_top_bar(top_sin,  "in_strength",  "NATO: Top 15 by in-strength (weighted)")

# Degree histograms (MultiDiGraph counts)
def plot_degree_hist(values, title):
    vals = np.fromiter(values.values(), dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        print(f"[Plot skipped: no positive values for {title}]"); return
    plt.figure(figsize=(6,4))
    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 30)
    plt.hist(vals, bins=bins, alpha=0.7)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Degree"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.show()

plot_degree_hist(in_deg,  "NATO: In-degree distribution (rows, log–log)")
plot_degree_hist(out_deg, "NATO: Out-degree distribution (rows, log–log)")

# Optional: LCC shortest-path estimate on simple digraph view (unweighted)
if n > 0:
    DG = nx.DiGraph(G)  # simple digraph with weights
    if DG.number_of_nodes() > 0:
        lcc_nodes = max(nx.weakly_connected_components(DG), key=len)
        und = DG.subgraph(lcc_nodes).to_undirected()
        sample_nodes = random.sample(list(und.nodes()), min(2000, und.number_of_nodes()))
        pairs = [(sample_nodes[i], sample_nodes[j]) for i in range(0, len(sample_nodes), 2) for j in range(i+1, len(sample_nodes), 2)]
        pairs = pairs[:5000]
        dists = []
        for u, v in pairs:
            try:
                dists.append(nx.shortest_path_length(und, u, v))
            except nx.NetworkXNoPath:
                pass
        if dists:
            print(f"Approx. avg shortest path (NATO LCC, unweighted): {np.mean(dists):.2f}")

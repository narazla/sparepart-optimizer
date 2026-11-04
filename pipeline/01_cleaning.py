"""
TAHAP 1 - DATA CLEANING, ENTITY RESOLUTION, ANOMALY FLAGGING
Input : data/raw/*.xlsx (6 file baseline)
Output: data/interim/*.pkl  (data bersih) + log data quality

Langkah:
 1. Buang duplikat persis di setiap file
 2. Seragamkan teks siklus Mplan -> interval dalam bulan, hitung jadwal berikutnya
 3. Validasi tanggal (PR, PO, notifikasi)
 4. Entity resolution: gabungkan material yang sama tetapi nomornya berbeda
 5. Bangun master material (kekritisan, site, kelas equipment, harga)
 6. Tandai konsumsi tidak wajar (Isolation Forest + robust z-score)
 7. Simpan log & skor kualitas data
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW, INT = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "interim")
os.makedirs(INT, exist_ok=True)

TODAY = pd.Timestamp("2026-10-01")          # data berakhir 30 Sep 2026
CRIT_ORDER = ["A", "B", "C", "S"]            # A tertinggi, S terendah (konfirmasi panitia)
FX = {"IDR": 1, "USD": 16300, "EUR": 17700, "SGD": 12600, "JPY": 105}   # asumsi kurs tetap

log = []
def note(source, issue, n, action, reason):
    log.append(dict(source=source, issue=issue, rows_affected=int(n), action=action, reason=reason))
    print(f"[{source}] {issue}: {n} -> {action}")

read = lambda f: pd.read_excel(os.path.join(RAW, f))
bom, mv, nt = read("BoM.xlsx"), read("Movement_Material.xlsx"), read("Notif_History.xlsx")
fl, pp, mp = read("FuncLoc_list.xlsx"), read("PR_PO.xlsx"), read("Mitem__Mplan.xlsx")
raw_n = {"BoM": len(bom), "Movement Material": len(mv), "Notif History": len(nt),
         "FuncLoc List": len(fl), "PR PO": len(pp), "Mitem & Mplan": len(mp)}
bad = {k: set() for k in raw_n}               # index baris bermasalah (untuk skor kualitas)

# ---------------------------------------------------------------- 1. duplikat persis
def dedup(df, name, reason):
    d = df.duplicated()
    if d.sum():
        note(name, "Exact duplicate rows", d.sum(), "Removed", reason)
        bad[name] |= set(df.index[d])
    return df[~d].reset_index(drop=True)

pp = dedup(pp, "PR PO", "Same PR + item appears twice; would double-count spend and lead-time statistics")
mv = dedup(mv, "Movement Material", "Identical issue (date, material, order, qty); would inflate consumption")
bom = dedup(bom, "BoM", "Same FuncLoc + material listed twice; would double-count installed quantity")

# ---------------------------------------------------------------- 2. siklus Mplan
UNIT_M = {"WEEK": 12 / 52, "MONTH": 1, "YEAR": 12}
def cycle_months(txt):
    m = re.match(r"\s*(\d+)?\s*(WEEK|MONTH|YEAR)", str(txt).upper())
    return np.nan if not m else int(m.group(1) or 1) * UNIT_M[m.group(2)]

mp["cycle_months"] = mp["Cycle Descriptions"].map(cycle_months)
n_var = mp["Cycle Descriptions"].nunique()
nonstd = ~mp["Cycle Descriptions"].str.match(r"^\d+ (WEEKLY|MONTHLY|YEARLY)$")
note("Mitem & Mplan", f"{n_var} different cycle spellings (e.g. YEARLY / Yearly / 1 YEARLY / 12 MONTHLY)",
     nonstd.sum(), "Standardized", "Converted every cycle to an interval in months so due dates can be computed")
bad["Mitem & Mplan"] |= set(mp.index[nonstd])

def next_due(row):
    s, c = row["Start of cycle"], row["cycle_months"]
    if pd.isna(c) or c <= 0:
        return pd.NaT
    if s >= TODAY:
        return s
    k = int(np.ceil(((TODAY - s).days / 30.4375) / c))
    return s + pd.DateOffset(days=int(round(k * c * 30.4375)))
mp["next_due"] = mp.apply(next_due, axis=1)

# ---------------------------------------------------------------- 3. validasi tanggal
pp["lead_time"] = (pp["PO-Delivery Date"] - pp["PR-Date"]).dt.days
neg = pp["lead_time"] < 0
note("PR PO", "PO delivered before PR was created", neg.sum(), "Removed from lead-time stats", "Impossible sequence")
bad["PR PO"] |= set(pp.index[neg])
reqb = pp["PR-Delivery Date"] < pp["PR-Date"]
note("PR PO", "Required date earlier than PR date", reqb.sum(), "Flagged", "Kept for price; excluded from lateness rate")
bad["PR PO"] |= set(pp.index[reqb])
long_lt = pp["lead_time"] > 365
note("PR PO", "Lead time above 365 days", long_lt.sum(), "Flagged", "Kept as real supply risk, capped at P99 for modelling")
pp["lt_valid"] = ~neg
pp["late_valid"] = ~neg & ~reqb
pp["is_late"] = pp["PO-Delivery Date"] > pp["PR-Delivery Date"]

c_bad = nt["Completion by Date"] < nt["Created On"]
note("Notif History", "Completion before creation date", c_bad.sum(), "Flagged", "Impossible sequence")
bad["Notif History"] |= set(nt.index[c_bad])
no_ord = nt["Order"].isna()
note("Notif History", "Notification without maintenance order", no_ord.sum(), "Kept",
     "Still a valid failure signal on the FuncLoc; cannot be linked to parts issued")

# ---------------------------------------------------------------- FuncLoc
site_nodes = fl["FuncLoc Criticallity"].isna()
note("FuncLoc List", "Missing criticality", site_nodes.sum(), "Kept",
     "All are site/system hierarchy nodes (SITE_*), not equipment; no parts attached")
fl["site"] = fl["Functional Location"].str.split("_").str[0]
EQUIP = {"PP": "Pump", "HE": "Heat Exchanger", "PRV": "Safety Valve", "SDV": "Shutdown Valve", "KC": "Compressor",
         "FCV": "Control Valve", "LCV": "Control Valve", "PCV": "Control Valve", "TCV": "Control Valve",
         "TEL": "Instrument", "PIT": "Instrument", "FIT": "Instrument", "TIT": "Instrument", "LIT": "Instrument",
         "DPT": "Instrument", "PGI": "Instrument", "ROF": "Instrument", "JBX": "Electrical", "VD": "Vessel",
         "CL": "Column", "BF": "Furnace", "TK": "Tank", "FL": "Filter", "MOV": "Motor Operated Valve",
         "KT": "Turbine"}
def equip_class(tag):
    if tag.startswith("SITE"):
        return "Site / System"
    body = tag.split("_", 1)[1] if "_" in tag else tag
    if re.match(r"^\d", body):
        return "Piping"
    if body.endswith("-M") or "-M-" in body:
        return "Motor"
    if body.endswith("-CST"):
        return "Civil Structure"
    pre = re.match(r"^([A-Z]+)", body)
    return EQUIP.get(pre.group(1), "Other Equipment") if pre else "Other Equipment"
fl["equip_class"] = fl["Functional Location"].map(equip_class)

# ---------------------------------------------------------------- 4. entity resolution (material kembar)
mdesc = bom.groupby("Material No")["Material Desc."].first()
def attr_key(d):
    """Kunci atribut: setiap token dipisah koma, spasi & huruf kecil dihilangkan, urutan diabaikan."""
    return "|".join(sorted(re.sub(r"\s+", "", t.upper()) for t in str(d).split(",")))
keys = mdesc.map(attr_key)
# kandidat dari kemiripan teks (TF-IDF karakter), lalu dikonfirmasi dengan kunci atribut
tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit_transform(mdesc.values)
S = cosine_similarity(tf, dense_output=False); S.setdiag(0); S = S.tocoo()
cand = [(i, j) for i, j, s in zip(S.row, S.col, S.data) if i < j and s >= 0.90]
same = [(i, j) for i, j in cand if keys.iloc[i] == keys.iloc[j]]
rejected = len(cand) - len(same)
# union-find
parent = list(range(len(mdesc)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j in same:
    parent[find(i)] = find(j)
roots = pd.Series([find(i) for i in range(len(mdesc))], index=mdesc.index)
grp_size = roots.map(roots.value_counts())
twins = roots[grp_size > 1]

used_cnt = mv.groupby("Material Number").size()
stock_by = bom.groupby("Material No")["Stock"].first()
tw = pd.DataFrame({"root": twins, "desc": mdesc[twins.index], "stock": stock_by[twins.index],
                   "uses": used_cnt.reindex(twins.index).fillna(0)})
tw["twin_group"] = tw["root"].rank(method="dense").astype(int)
tw = tw.sort_values(["twin_group", "uses", "stock"], ascending=[True, False, False])
tw["is_primary"] = (~tw["twin_group"].duplicated()).astype(int)
primary = tw[tw.is_primary == 1].set_index("twin_group")["root"]
remap = {m: tw.index[(tw.twin_group == g) & (tw.is_primary == 1)][0] for m, g in tw["twin_group"].items()}
g_stats = tw.groupby("twin_group").agg(n=("stock", "size"), zero=("stock", lambda x: (x == 0).sum()),
                                       pos=("stock", lambda x: (x > 0).sum()))
false_so = int(((g_stats.zero > 0) & (g_stats.pos > 0)).sum())
note("BoM", f"Duplicate material master: {tw.twin_group.nunique()} groups of identical parts under different numbers",
     len(tw), "Merged", f"Same attributes, different number; in {false_so} groups one number shows zero stock while a twin holds stock")
note("BoM", "Look-alike description pairs with different attributes (e.g. 1/2IN vs 12IN, P/N FE-1 vs FE-2)",
     rejected, "Kept separate", "TF-IDF similarity >= 0.90 but attribute check failed; prevents false merges")
twin_table = tw.reset_index().rename(columns={"index": "material_no", "Material No": "material_no", "desc": "description"})
twin_table = twin_table[["twin_group", "material_no", "description", "stock", "is_primary"]]

def apply_remap(s):
    return s.map(lambda m: remap.get(m, m))
bom["orig_material"] = bom["Material No"]; bom["Material No"] = apply_remap(bom["Material No"])
mv["orig_material"] = mv["Material Number"]; mv["Material Number"] = apply_remap(mv["Material Number"])
pp["orig_material"] = pp["PR-Material"]; pp["PR-Material"] = apply_remap(pp["PR-Material"])

# ---------------------------------------------------------------- 5. master material
crit_rank = {c: i for i, c in enumerate(CRIT_ORDER)}
b = bom.merge(fl[["Functional Location", "FuncLoc Criticallity", "site", "equip_class"]],
              left_on="Functional Loc.", right_on="Functional Location", how="left")
b["crank"] = b["FuncLoc Criticallity"].map(crit_rank).fillna(3)
# stok/min/max per nomor asli (nilainya sama di semua FuncLoc), lalu dijumlah per material gabungan
per_orig = bom.groupby("orig_material").agg(material=("Material No", "first"), stock=("Stock", "first"),
                                            mn=("Min.Qty", "first"), mx=("Max.Qty", "first"),
                                            mrp=("MRP Type", "first"))
inv = per_orig.groupby("material").agg(stock=("stock", "sum"), min_old=("mn", "sum"), max_old=("mx", "sum"),
                                       mrp=("mrp", lambda x: "Stock Item" if (x == "Stock Item").any() else "Non Stock"))
top = b.sort_values("crank").groupby("Material No").first()
mat = pd.DataFrame({
    "description": mdesc.reindex(inv.index),
    "unit": bom.groupby("Material No")["Unit"].first(),
    "criticality": top["FuncLoc Criticallity"].fillna("S"),
    "main_system": top["equip_class"],
    "site": b.groupby("Material No")["site"].agg(lambda x: "+".join(sorted(set(x.dropna())))),
    "n_funcloc": b.groupby("Material No")["Functional Loc."].nunique(),
    "installed_qty": b.groupby("Material No")["Installed Qty"].sum(),
}).join(inv)
mat.index.name = "material_no"
mat["category"] = mat["description"].str.split(",").str[0].str.strip()

nonstock_conf = bom[(bom["MRP Type"] == "Non Stock") & ((bom["Min.Qty"] > 0) | (bom["Stock"] > 0))]
note("BoM", "Non Stock items carrying Min/Max or stock", len(nonstock_conf), "Flagged",
     "MRP type contradicts parameters; scope limited to Stock Items, listed for master-data review")
bad["BoM"] |= set(nonstock_conf.index)
frac = bom["Installed Qty"] % 1 != 0
note("BoM", "Fractional installed quantity", frac.sum(), "Kept", "Valid for length/volume units (LG, L, M)")

# harga: median IDR per unit dari semua PO material tsb; estimasi dari kategori jika tidak ada PO
pp["unit_price_idr"] = pp["PO-Item Amount"] * pp["PO-Currency"].map(FX)
fxn = (pp["PO-Currency"] != "IDR").sum()
note("PR PO", "Non-IDR purchase orders (USD, EUR, SGD, JPY)", fxn, "Standardized", "Converted to IDR with fixed rates")
price = pp.groupby("PR-Material")["unit_price_idr"].median()
mat["price"] = price.reindex(mat.index)
mat["price_source"] = np.where(mat["price"].notna(), "actual", "estimated")
# Estimasi harga: kNN pada TF-IDF deskripsi (kata + karakter), median log-harga 5 tetangga terdekat
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.sparse import hstack
act = mat[mat.price_source == "actual"]
vec_w = TfidfVectorizer(token_pattern=r"[A-Z0-9/\.\-]+", lowercase=False).fit(mat["description"])
vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(mat["description"])
emb = lambda s: hstack([vec_w.transform(s), vec_c.transform(s)]).tocsr()
def knn_price(train_desc, train_logp, query_desc, k=5):
    nn = NearestNeighbors(n_neighbors=min(k, len(train_desc)), metric="cosine").fit(emb(train_desc))
    _, idx = nn.kneighbors(emb(query_desc))
    return np.median(train_logp.values[idx], axis=1)
# validasi silang 5-fold pada part yang punya harga aktual
logp = np.log(act["price"]); errs = []; base = []
for tr, te in KFold(5, shuffle=True, random_state=0).split(act):
    pred = knn_price(act["description"].iloc[tr], logp.iloc[tr], act["description"].iloc[te])
    errs.append(np.abs(pred - logp.iloc[te].values))
    cm = act.iloc[tr].groupby("category")["price"].median().map(np.log)
    bp = act["category"].iloc[te].map(cm).fillna(logp.iloc[tr].median()).values
    base.append(np.abs(bp - logp.iloc[te].values))
errs, base = np.concatenate(errs), np.concatenate(base)
PRICE_CV = dict(knn_within_2x=float(np.mean(errs < np.log(2))), cat_within_2x=float(np.mean(base < np.log(2))),
                knn_median_factor=float(np.exp(np.median(errs))), cat_median_factor=float(np.exp(np.median(base))))
print("Price model CV:", {k: round(v, 3) for k, v in PRICE_CV.items()})
miss = mat["price"].isna()
mat.loc[miss, "price"] = np.exp(knn_price(act["description"], logp, mat.loc[miss, "description"]))
note("BoM", "Stock items without any purchase history (no price)",
     ((mat.price_source == "estimated") & (mat.mrp == "Stock Item")).sum(), "Estimated",
     f"AI price model (kNN on description text): {PRICE_CV['knn_within_2x']:.0%} of test parts within 2x of true price "
     f"vs {PRICE_CV['cat_within_2x']:.0%} for category median")
pd.Series(PRICE_CV).to_pickle(os.path.join(INT, "price_cv.pkl"))
mat["stock_value"] = mat["stock"] * mat["price"]

# ---------------------------------------------------------------- 6. anomali konsumsi
g = mv.groupby("Material Number")["Quantity"]
med = g.transform("median"); mad = g.transform(lambda x: (x - x.median()).abs().median())
mv["robust_z"] = (mv["Quantity"] - med) / (1.4826 * mad.replace(0, np.nan))
mv["ratio_med"] = mv["Quantity"] / med
X = np.column_stack([np.log1p(mv["Quantity"]), np.log1p(mv["ratio_med"]), np.log1p(g.transform("size"))])
iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=42).fit(X)
mv["iso_outlier"] = iso.predict(X) == -1
mv["is_anomaly"] = mv["iso_outlier"] & (mv["ratio_med"] >= 5)
note("Movement Material", "Abnormal consumption (Isolation Forest + >=5x part median)", mv["is_anomaly"].sum(),
     "Flagged", "Shown as Monitor alerts; capped when training forecasts so one spike does not inflate stock")
bad["Movement Material"] |= set(mv.index[mv["is_anomaly"]])
mat["has_anomaly"] = mat.index.isin(mv.loc[mv.is_anomaly, "Material Number"]).astype(int)

v = mat.loc[mat.mrp == "Stock Item", "stock_value"].sort_values(ascending=False)
top5 = v.head(5).sum() / v.sum()
note("BoM", "Inventory value concentrated in few items", 5, "Flagged",
     f"Top 5 items = {top5:.0%} of stock value; verify price/stock before reporting totals")

# ---------------------------------------------------------------- 6b. Non-Stock yang salah klasifikasi
# Non Stock = dibeli hanya saat dibutuhkan. Jika part kritis (A/B) dipakai berulang dan lead time-nya panjang
# (atau tidak pernah dibeli lewat PO), status Non Stock memicu pembelian darurat -> kandidat Convert to Stock.
use_months = mv.assign(m=mv["Posting Date"].dt.to_period("M")).groupby("Material Number")["m"].nunique()
lt_med = pp[pp.lt_valid].groupby("PR-Material")["lead_time"].median()
mat["use_months"] = use_months.reindex(mat.index).fillna(0)
mat["lt_hist"] = lt_med.reindex(mat.index)
mat["convert_candidate"] = ((mat.mrp == "Non Stock") & mat.criticality.isin(["A", "B"]) & (mat.use_months >= 2)
                            & ((mat.lt_hist > 42) | mat.lt_hist.isna()))
mat["in_scope"] = (mat.mrp == "Stock Item") | mat.convert_candidate
note("BoM", "Non Stock items behaving like stock items (criticality A/B, used in 2+ months, lead time > 42 days or no PO)",
     int(mat.convert_candidate.sum()), "Flagged",
     "Recommended for MRP type change to Stock Item; added to optimisation scope")
ns_stock = int(((mat.mrp == "Non Stock") & (mat.stock > 0)).sum())
note("BoM", "Non Stock materials holding stock", ns_stock, "Flagged",
     "Stock exists but is not planned by MRP; included in master-data review list")

# ---------------------------------------------------------------- 6c. Purchasing Group = departemen pemilik barang
pg = pp.groupby("PR-Material")["PR-Purchasing Group"].agg(lambda x: x.mode().iat[0])
mat["purchasing_group"] = pg.reindex(mat.index)
mat["pg_source"] = np.where(mat["purchasing_group"].notna(), "PO history", "inferred from category")
cat_pg = mat.dropna(subset=["purchasing_group"]).groupby("category")["purchasing_group"].agg(lambda x: x.mode().iat[0])
mat["purchasing_group"] = mat["purchasing_group"].fillna(mat["category"].map(cat_pg)).fillna(pg.mode().iat[0])

# ---------------------------------------------------------------- 7. simpan
for name, df in [("bom", bom), ("mv", mv), ("nt", nt), ("fl", fl), ("pp", pp), ("mp", mp), ("mat", mat),
                 ("twins", twin_table)]:
    df.to_pickle(os.path.join(INT, f"{name}.pkl"))
dq_log = pd.DataFrame(log); dq_log.insert(0, "id", range(1, len(dq_log) + 1))
dq_log.to_pickle(os.path.join(INT, "dq_log.pkl"))
score = pd.DataFrame([dict(source=k, total_rows=n, clean_rows=n - len(bad[k]),
                           score=round(100 * (n - len(bad[k])) / n, 1)) for k, n in raw_n.items()])
score.to_pickle(os.path.join(INT, "dq_score.pkl"))
print("\nData quality score:\n", score.to_string(index=False))
sm = mat[mat.mrp == "Stock Item"]
print(f"\nStock items after merge: {len(sm)} | actual price {(sm.price_source=='actual').sum()}")
print(sm.criticality.value_counts().to_dict()); print(sm.main_system.value_counts().head(10).to_dict())

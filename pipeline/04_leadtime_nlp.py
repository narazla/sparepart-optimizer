"""
TAHAP 4 - LEAD TIME (QUANTILE GRADIENT BOOSTING) + NLP NOTIFIKASI
Input : data/interim/pp.pkl, mat.pkl, nt.pkl, bom.pkl, mv.pkl
Output: lead_time.pkl, suppliers.pkl, notif_class.pkl, notif_signal.pkl, notif_monthly.pkl, lt_eval.pkl

A. Lead time: prediksi P50 & P80 lead time per part (bukan rata-rata) karena ~58% PO terlambat.
   Model: gradient boosting quantile regression (LightGBM jika terpasang, selain itu scikit-learn
   HistGradientBoosting - algoritma histogram yang sama keluarganya dengan LightGBM).
   Validasi: latih PR sebelum Apr-26, uji PR Apr-26 ke atas. Dibandingkan dengan baseline
   "kuantil historis per material".
B. NLP: klasifikasi deskripsi notifikasi -> Failure / Repair-Replace / Routine-Support
   (weak supervision: aturan kata kunci -> TF-IDF + Logistic Regression untuk sisanya).
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT = os.path.join(ROOT, "data", "interim")
L = lambda n: pd.read_pickle(os.path.join(INT, f"{n}.pkl"))
S = lambda df, n: df.to_pickle(os.path.join(INT, f"{n}.pkl"))
pp, mat, nt, bom, mv = L("pp"), L("mat"), L("nt"), L("bom"), L("mv")
TODAY = pd.Timestamp("2026-10-01")

try:
    import lightgbm as lgb
    def qmodel(alpha):
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=20, verbose=-1)
    ENGINE = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def qmodel(alpha):
        return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, min_samples_leaf=20, categorical_features="from_dtype",
                                             random_state=0)
    ENGINE = "sklearn HistGradientBoosting"
print("Engine:", ENGINE)

# ================================================================ A. LEAD TIME
d = pp[pp.lt_valid].copy()
cap = d.lead_time.quantile(0.99)
d["y"] = d.lead_time.clip(upper=cap)
d["category"] = d["PR-Material"].map(mat["category"]).fillna("UNKNOWN")
d["crit"] = d["PR-Material"].map(mat["criticality"]).fillna("S")
d["log_qty"] = np.log1p(d["PR-Quantity"])
d["log_price"] = np.log1p(d["unit_price_idr"])
d["req_lead"] = (d["PR-Delivery Date"] - d["PR-Date"]).dt.days.clip(lower=0)
d["month"] = d["PR-Date"].dt.month
CAT = ["supplier", "pgroup", "prtype", "category", "crit"]
d["supplier"], d["pgroup"], d["prtype"] = d["PO-Supplier Code"].astype(str), d["PR-Purchasing Group"], d["PR-Type"]
FEATS = CAT + ["log_qty", "log_price", "req_lead", "month"]
for c in CAT:
    d[c] = d[c].astype("category")

def pinball(y, q, a):
    e = y - q
    return np.mean(np.maximum(a * e, (a - 1) * e))

split = pd.Timestamp("2026-04-01")
tr, te = d[d["PR-Date"] < split], d[d["PR-Date"] >= split]
res = {}
for a in (0.5, 0.8):
    mdl = qmodel(a).fit(tr[FEATS], tr["y"])
    pred = mdl.predict(te[FEATS])
    # baseline: kuantil historis per material, fallback global
    hq = tr.groupby("PR-Material")["y"].quantile(a)
    base = te["PR-Material"].map(hq).fillna(tr["y"].quantile(a)).values
    res[a] = dict(model=pinball(te.y.values, pred, a), baseline=pinball(te.y.values, base, a),
                  coverage_model=np.mean(te.y.values <= pred), coverage_base=np.mean(te.y.values <= base))
lt_eval = pd.DataFrame(res).T
lt_eval["improvement"] = 1 - lt_eval.model / lt_eval.baseline
print("\nLead time backtest (pinball loss, lower=better; coverage target = quantile):\n", lt_eval.round(3).to_string())

# fit final pada seluruh data, prediksi "order berikutnya" untuk setiap stock item
final = {a: qmodel(a).fit(d[FEATS], d["y"]) for a in (0.5, 0.8)}
sm = mat[mat.mrp == "Stock Item"]
hist = d.groupby("PR-Material").agg(supplier=("supplier", lambda x: x.mode().iat[0]),
                                    pgroup=("pgroup", lambda x: x.mode().iat[0]),
                                    prtype=("prtype", lambda x: x.mode().iat[0]),
                                    log_qty=("log_qty", "median"), req_lead=("req_lead", "median"),
                                    late_rate=("is_late", "mean"), n_po=("y", "size"),
                                    main_supplier=("PO-Supplier Name", lambda x: x.mode().iat[0]))
q = pd.DataFrame(index=sm.index).join(hist)
q["category"], q["crit"] = sm["category"], sm["criticality"]
q["log_price"] = np.log1p(sm["price"])
cat_mode = d.groupby("category", observed=True)[["pgroup", "prtype"]].agg(lambda x: x.mode().iat[0])
for c in ["pgroup", "prtype"]:
    q[c] = q[c].astype(object).fillna(q["category"].map(cat_mode[c]).astype(object)).fillna(d[c].mode().iat[0])
q["log_qty"] = q["log_qty"].fillna(d["log_qty"].median())
q["req_lead"] = q["req_lead"].fillna(d["req_lead"].median())
q["month"] = 10
for c in CAT:
    q[c] = pd.Categorical(q[c].astype(object), categories=d[c].cat.categories)
lt = pd.DataFrame(index=sm.index)
lt["lt_p50"] = np.round(final[0.5].predict(q[FEATS]))
lt["lt_p80"] = np.maximum(np.round(final[0.8].predict(q[FEATS])), lt["lt_p50"])
cat_late = d.groupby("category", observed=True)["is_late"].mean()
lt["late_rate"] = q["late_rate"].fillna(sm["category"].map(cat_late)).fillna(d.is_late.mean())
lt["main_supplier"] = q["main_supplier"]
lt["lt_source"] = np.where(q["n_po"].notna(), "model+history", "model (no PO history)")
print("\nLead time per part: P50 median", lt.lt_p50.median(), "| P80 median", lt.lt_p80.median())

sup = d.groupby("PO-Supplier Code").agg(name=("PO-Supplier Name", "first"), n_po=("y", "size"),
                                        lt_median=("lead_time", "median"),
                                        lt_cv=("lead_time", lambda x: x.std() / max(x.mean(), 1)),
                                        late_rate=("is_late", "mean"))
sup["reliability"] = (100 * (0.6 * (1 - sup.late_rate) + 0.4 * (1 - sup.lt_cv.clip(0, 1)))).round(1)
sup.index.name = "supplier_code"
S(lt, "lead_time"); S(sup.reset_index(), "suppliers"); S(lt_eval, "lt_eval")

# ================================================================ B. NLP NOTIFIKASI
FAIL = r"LEAK|BROKEN|CRACK|VIBRAT|NOISE|STUCK|PASSING|ABNORMAL|ERROR|HOT SPOT|HUNTING|TRIP|DAMAGE|FAIL|BURN|" \
       r"OVERHEAT|HIGH TEMP|LOW PRESS|HIGH PRESS|CORRO|JAMMED|NOT WORK|MALFUNCTION|SEIZE|WORN|LOOSE|BLOCK|PLUG"
REPAIR = r"REPLACE(?! LAMP)|REPAIR(?! INSULATION)|OVERHAUL|RETUBING|REBUILD|CHANGE|RENEW"
ROUTINE = r"SCAFFOLD|PAINT|INSULATION|CLEAN|INSPECT|THICKNESS|MEGGER|THERMOGRAPH|INSTALL COVER|LAMP|" \
          r"DECOKING|MEASUREMENT|CALIBRAT|LUBRIC|GREASING|HOUSEKEEP|CHECK"
txt = nt["Description"].fillna("").str.upper()
f, r, ro = txt.str.contains(FAIL), txt.str.contains(REPAIR), txt.str.contains(ROUTINE)
rule = np.select([f, r & ~ro, ro & ~f & ~r], ["Failure", "Repair/Replace", "Routine/Support"], None)
nt["rule_label"] = rule
lab = nt[nt.rule_label.notna()]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"[A-Z]{2,}", lowercase=False)
X_all = vec.fit_transform(txt)
clf = LogisticRegression(max_iter=2000, C=3, class_weight="balanced")
cv_acc = cross_val_score(clf, X_all[lab.index], lab.rule_label, cv=5).mean()
clf.fit(X_all[lab.index], lab.rule_label)
nt["nlp_class"] = nt["rule_label"]
unl = nt.rule_label.isna()
nt.loc[unl, "nlp_class"] = clf.predict(X_all[unl.values])
nt.loc[nt["Notification Type"] == "N2", "nlp_class"] = "Failure"           # N2 = semua Emergency
nt["is_failure_signal"] = nt["nlp_class"].isin(["Failure", "Repair/Replace"])
print(f"\nNLP: rule-labelled {len(lab)} ({len(lab)/len(nt):.0%}), model CV accuracy {cv_acc:.1%}, "
      f"model-labelled {unl.sum()}")
print(nt["nlp_class"].value_counts().to_dict())
# validasi independen: notifikasi yang ordernya mengeluarkan part
issued_orders = set(mv["Order"])
nt["parts_issued"] = nt["Order"].isin(issued_orders)
val = nt[nt.Order.notna()].groupby("nlp_class")["parts_issued"].mean()
print("Share of notifications whose order issued spare parts:", val.round(3).to_dict())
S(pd.Series({"cv_accuracy": cv_acc, "rule_coverage": len(lab) / len(nt), **{f"issued_{k}": v for k, v in val.items()}}),
  "nlp_eval")

# sinyal kerusakan per FuncLoc per bulan -> per material
nt["month"] = nt["Notification Date"].dt.to_period("M")
fs = nt[nt.is_failure_signal]
notif_monthly = fs.groupby(["Functional Location", "month"]).size().rename("n").reset_index()
S(notif_monthly, "notif_monthly")
recent = fs[fs["Notification Date"] >= TODAY - pd.DateOffset(months=12)]
cnt = recent.groupby("Functional Location").size()
b = bom[["Functional Loc.", "Material No"]].drop_duplicates()
b["n"] = b["Functional Loc."].map(cnt).fillna(0)
notif_signal = b.groupby("Material No")["n"].sum()
S(notif_signal, "notif_signal")
S(nt[["Notification", "Functional Location", "Notification Date", "Description", "nlp_class", "is_failure_signal"]],
  "notif_class")
print("Parts with >=1 failure signal in last 12m:", int((notif_signal.reindex(sm.index).fillna(0) > 0).sum()))

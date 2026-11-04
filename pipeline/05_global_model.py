"""
TAHAP 5 - MODEL GLOBAL: NEED PROBABILITY (peluang part dibutuhkan bulan depan)
Satu model gradient boosting dilatih dari SEMUA part sekaligus (panel part x bulan), sehingga
part dengan histori pendek tetap bisa diprediksi dari pola part lain.

Fitur (semua diketahui di akhir bulan t, target = ada pemakaian di bulan t+1):
- histori part: lag 1-3 bulan, jumlah bulan berdemand 3/6 bulan terakhir & kumulatif, bulan sejak terakhir dipakai
- equipment: jumlah terpasang (BoM), jumlah FuncLoc, kekritisan, kelas equipment, kategori, harga
- sinyal kerusakan: notifikasi Failure/Repair (hasil NLP tahap 4) di FuncLoc terkait, 1 & 3 bulan terakhir
- jadwal maintenance: jumlah PM jatuh tempo di FuncLoc terkait pada bulan t+1 (dari Mplan)
- event: bulan t+1 termasuk window TAM
Validasi: latih target s.d. Mar-26, uji target Apr-26..Sep-26. Baseline: frekuensi historis (TSB-style).
Metrik: ROC-AUC, Brier score, dan "top-decile capture" (berapa % pemakaian nyata tertangkap di 10% skor tertinggi).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT = os.path.join(ROOT, "data", "interim")
L = lambda n: pd.read_pickle(os.path.join(INT, f"{n}.pkl"))
mat, bom, mp, series = L("mat"), L("bom"), L("mp"), L("series")
notif_monthly, tam_months = L("notif_monthly"), set(L("tam_months"))

try:
    import lightgbm as lgb
    make_clf = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                          min_child_samples=40, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    ENGINE = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    make_clf = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                                      min_samples_leaf=40, l2_regularization=1.0,
                                                      categorical_features="from_dtype", random_state=0)
    ENGINE = "sklearn HistGradientBoosting"
print("Engine:", ENGINE)

sm = mat[mat.in_scope]
raw = series["raw"].reindex(sm.index).fillna(0)
HIST = list(raw.columns)
FUT = list(pd.period_range("2026-10", "2027-03", freq="M"))
ALL = HIST + FUT

# ---------------------------------------------------------------- PM jatuh tempo per FuncLoc per bulan
occ = []
lo, hi = pd.Timestamp("2025-05-01"), pd.Timestamp("2027-04-01")
for fl, s, c in mp[["Functional Location", "Start of cycle", "cycle_months"]].itertuples(index=False):
    if pd.isna(c) or c <= 0 or c < 1:          # siklus mingguan diabaikan (inspeksi rutin, tidak relevan untuk part)
        continue
    step = pd.DateOffset(days=int(round(c * 30.4375)))
    k0 = max(0, int(np.floor(((lo - s).days / 30.4375) / c)))
    d = s + pd.DateOffset(days=int(round(k0 * c * 30.4375)))
    while d < hi:
        if d >= lo:
            occ.append((fl, d.to_period("M")))
        d = d + step
pm = pd.DataFrame(occ, columns=["fl", "month"]).groupby(["fl", "month"]).size().rename("n").reset_index()
link = bom[["Functional Loc.", "Material No"]].drop_duplicates().rename(columns={"Functional Loc.": "fl", "Material No": "mat"})
pm_mat = pm.merge(link, on="fl").groupby(["mat", "month"])["n"].sum().unstack(fill_value=0) \
           .reindex(index=sm.index, columns=ALL, fill_value=0)
nf = notif_monthly.rename(columns={"Functional Location": "fl"}).merge(link, on="fl") \
     .groupby(["mat", "month"])["n"].sum().unstack(fill_value=0).reindex(index=sm.index, columns=ALL, fill_value=0)
pm_mat.to_pickle(os.path.join(INT, "pm_by_material.pkl"))

# ---------------------------------------------------------------- panel
static = pd.DataFrame({"installed": np.log1p(sm.installed_qty), "n_fl": sm.n_funcloc,
                       "log_price": np.log1p(sm.price), "crit": sm.criticality, "eq": sm.main_system,
                       "cat": sm.category}, index=sm.index)
for c in ["crit", "eq", "cat"]:
    static[c] = static[c].astype("category")
Y = (raw > 0).astype(int)

def feats_at(t_idx, target_month):
    """fitur di akhir bulan HIST[t_idx] untuk memprediksi target_month"""
    h = Y.iloc[:, : t_idx + 1]
    q = raw.iloc[:, : t_idx + 1]
    f = pd.DataFrame(index=sm.index)
    f["lag1"], f["lag2"] = h.iloc[:, -1], (h.iloc[:, -2] if t_idx >= 1 else 0)
    f["lag3"] = h.iloc[:, -3] if t_idx >= 2 else 0
    f["nz3"], f["nz6"], f["nz_all"] = h.iloc[:, -3:].sum(1), h.iloc[:, -6:].sum(1), h.sum(1)
    f["freq_all"] = f["nz_all"] / (t_idx + 1)
    last = h.values[:, ::-1].argmax(axis=1)
    f["since_last"] = np.where(h.values.any(axis=1), last, 99)
    f["log_qty_mean"] = np.log1p(q.where(q > 0).mean(axis=1).fillna(0))
    cols = ALL[max(0, t_idx - 2): t_idx + 1]
    f["notif1"], f["notif3"] = nf[ALL[t_idx]], nf[cols].sum(1)
    f["pm_next"] = pm_mat[target_month]
    f["tam_next"] = int(str(target_month) in tam_months)
    return f.join(static)

rows = []
for t in range(2, len(HIST) - 1):
    f = feats_at(t, HIST[t + 1])
    f["y"], f["target"] = Y[HIST[t + 1]], HIST[t + 1]
    rows.append(f)
panel = pd.concat(rows)
split = pd.Period("2026-04", "M")
tr, te = panel[panel.target < split], panel[panel.target >= split]
FEATS = [c for c in panel.columns if c not in ("y", "target")]
clf = make_clf().fit(tr[FEATS], tr["y"])
p = clf.predict_proba(te[FEATS])[:, 1]
base = te["freq_all"].values                         # baseline: frekuensi historis
def capture(score, y, top=0.10):
    k = int(len(score) * top); idx = np.argsort(-score)[:k]
    return y[idx].sum() / max(y.sum(), 1)
ev = pd.DataFrame({
    "global_gbm": [roc_auc_score(te.y, p), brier_score_loss(te.y, p), capture(p, te.y.values)],
    "historical_frequency": [roc_auc_score(te.y, base), brier_score_loss(te.y, base), capture(base, te.y.values)],
}, index=["roc_auc", "brier", "top10pct_capture"])
print(f"\nPanel rows: train {len(tr):,} test {len(te):,} | positive rate test {te.y.mean():.3%}")
print(ev.round(3).to_string())

# kontribusi fitur (permutation importance pada data uji, AUC)
from sklearn.inspection import permutation_importance
sub = te.sample(min(len(te), 15000), random_state=0)
pi = permutation_importance(clf, sub[FEATS], sub["y"], scoring="roc_auc", n_repeats=3, random_state=0)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
print("\nFeature importance (drop in AUC when shuffled):\n", imp.round(4).head(12).to_string())

# ---------------------------------------------------------------- prediksi 6 bulan ke depan
final = make_clf().fit(panel[FEATS], panel["y"])
fut = []
for m in FUT:
    f = feats_at(len(HIST) - 1, m)
    fut.append(pd.DataFrame({"material_no": sm.index, "month": str(m),
                             "need_prob_gbm": final.predict_proba(f[FEATS])[:, 1],
                             "pm_due": (pm_mat[m] > 0).astype(int).values}))
fut = pd.concat(fut)
fut.to_pickle(os.path.join(INT, "need_prob.pkl"))
# prediksi holdout (model hanya melihat data s.d. Mar-26) -> dipakai replay tanpa kebocoran
pd.Series(p, index=te.index).groupby(level=0).mean().to_pickle(os.path.join(INT, "need_prob_holdout.pkl"))
ev.to_pickle(os.path.join(INT, "gbm_eval.pkl")); imp.to_pickle(os.path.join(INT, "gbm_importance.pkl"))
print("\nFuture need prob: mean", round(fut.need_prob_gbm.mean(), 4), "| parts with PM due in horizon",
      fut[fut.pm_due == 1].material_no.nunique())

"""
TAHAP 2 - KLASIFIKASI POLA DEMAND + MODEL FORECAST STATISTIK + BACKTEST
Input : data/interim/mat.pkl, mv.pkl
Output: data/interim/series.pkl, demand_class.pkl, model_eval.pkl, stat_fc.pkl

Metode
- Pola demand (Syntetos-Boylan): ADI = rata-rata jarak antar bulan berdemand, CV2 = variasi ukuran demand
- Model: MA6 (baseline), SES, Croston, SBA, TSB  -> semua ditulis sendiri (rumus pendek, mudah dijelaskan)
- Backtest: latih Mei-25..Mar-26 (11 bln), uji Apr-26..Sep-26 (6 bln)
- Metrik: scaled MAE (MAE / rata-rata demand latih), bias, error total demand 6 bulan
- Model dipilih PER POLA DEMAND (bukan per part) supaya tidak overfit ke 6 bulan uji
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT = os.path.join(ROOT, "data", "interim")
mat = pd.read_pickle(os.path.join(INT, "mat.pkl"))
mv = pd.read_pickle(os.path.join(INT, "mv.pkl"))

MONTHS = pd.period_range("2025-05", "2026-09", freq="M")
N_TRAIN, H = 11, 6
stock_items = mat.index[mat.in_scope]

# ---------------------------------------------------------------- deret bulanan
mv["month"] = mv["Posting Date"].dt.to_period("M")
raw = (mv.groupby(["Material Number", "month"])["Quantity"].sum().unstack(fill_value=0)
       .reindex(columns=MONTHS, fill_value=0))
# untuk pelatihan, lonjakan anomali dipotong ke 3x median bulan berdemand milik part tsb
mv_c = mv.copy()
mv_c["Quantity"] = mv_c["Quantity"].astype(float)
med_tx = mv_c.groupby("Material Number")["Quantity"].transform("median")
mv_c.loc[mv_c.is_anomaly, "Quantity"] = np.minimum(mv_c.loc[mv_c.is_anomaly, "Quantity"], 3 * med_tx[mv_c.is_anomaly])
clean = (mv_c.groupby(["Material Number", "month"])["Quantity"].sum().unstack(fill_value=0)
         .reindex(columns=MONTHS, fill_value=0))
raw = raw.reindex(stock_items, fill_value=0)
clean = clean.reindex(stock_items, fill_value=0)

# ---------------------------------------------------------------- klasifikasi pola
def classify(y):
    nz = y[y > 0]
    if len(nz) == 0:
        return "No demand", np.nan, np.nan
    adi = len(y) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    if len(nz) == 1:
        return "Sporadic", adi, cv2
    if adi < 1.32:
        return ("Smooth" if cv2 < 0.49 else "Erratic"), adi, cv2
    return ("Intermittent" if cv2 < 0.49 else "Lumpy"), adi, cv2

# TAM window: Nov-25..Jan-26 = 37% konsumsi 17 bulan, maintenance order 2-3x normal, sejalan dengan TAM Jan 2026.
TAM = [p for p in MONTHS if str(p) in ("2025-11", "2025-12", "2026-01")]
BASE = [p for p in MONTHS if p not in TAM]
base = clean[BASE]                                           # demand rutin (tanpa bulan TAM)
tam_qty = raw[TAM].sum(axis=1)                               # kebutuhan saat TAM -> input TAM Readiness

cls = pd.DataFrame([classify(base.loc[m].values) for m in base.index], index=base.index,
                   columns=["pattern", "adi", "cv2"])
cls.loc[(cls.pattern == "No demand") & (tam_qty > 0), "pattern"] = "TAM-only"
cls["demand_months"] = (raw > 0).sum(axis=1)
cls["total_issued"] = raw.sum(axis=1)
cls["tam_qty"] = tam_qty
print("Pola demand rutin (stock items):", cls.pattern.value_counts().to_dict())
print(f"Konsumsi saat TAM: {raw[TAM].values.sum() / raw.values.sum():.0%} dari total")

# ---------------------------------------------------------------- model
def ma(y, w=6):
    return np.mean(y[-w:])

def ses(y, alpha=0.2):
    l = y[0]
    for v in y[1:]:
        l = alpha * v + (1 - alpha) * l
    return l

def croston(y, alpha=0.1, sba=False):
    nz = np.flatnonzero(y)
    if len(nz) == 0:
        return 0.0
    z, p, q = y[nz[0]], max(nz[0] + 1, 1), 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    f = z / p
    return f * (1 - alpha / 2) if sba else f

def tsb(y, alpha=0.1, beta=0.1, return_parts=False):
    nz = np.flatnonzero(y)
    if len(nz) == 0:
        return (0.0, 0.0, 0.0) if return_parts else 0.0
    z, pr = y[nz[0]], 1 / (nz[0] + 1)
    for t in range(nz[0] + 1, len(y)):
        d = 1.0 if y[t] > 0 else 0.0
        pr = pr + beta * (d - pr)          # peluang demand diperbarui SETIAP bulan -> bisa turun ke 0 (obsolete)
        if y[t] > 0:
            z = z + alpha * (y[t] - z)
    return (pr * z, pr, z) if return_parts else pr * z

MODELS = {"MA6": ma, "SES": ses, "Croston": croston, "SBA": lambda y: croston(y, sba=True), "TSB": tsb}

# ---------------------------------------------------------------- backtest
# Uji Apr-26..Sep-26. Dua varian latih:
#   naive     : semua 11 bulan Mei-25..Mar-26 (termasuk lonjakan TAM)
#   tam_aware : hanya bulan rutin Mei-25..Okt-25 + Feb-26..Mar-26 (8 bulan)
TEST = [p for p in MONTHS if p >= pd.Period("2026-04", "M")]
TR_NAIVE = [p for p in MONTHS if p < pd.Period("2026-04", "M")]
TR_AWARE = [p for p in TR_NAIVE if p not in TAM]
rows = []
for m in clean.index:
    te = raw.loc[m, TEST].values.astype(float)
    for variant, cols in (("naive", TR_NAIVE), ("tam_aware", TR_AWARE)):
        tr = clean.loc[m, cols].values.astype(float)
        if tr.sum() == 0:
            continue
        scale = tr.mean()
        for name, f in MODELS.items():
            fc = f(tr)
            rows.append(dict(material_no=m, variant=variant, model=name,
                             mae=np.mean(np.abs(te - fc)) / scale,
                             bias=(H * fc - te.sum()) / max(te.sum(), scale * H),
                             tot_err=abs(H * fc - te.sum()) / max(te.sum(), scale * H)))
ev_all = pd.DataFrame(rows).merge(cls[["pattern"]], left_on="material_no", right_index=True)
# bandingkan pada part yang sama (punya histori di kedua varian)
common = set(ev_all[ev_all.variant == "naive"].material_no) & set(ev_all[ev_all.variant == "tam_aware"].material_no)
ev_c = ev_all[ev_all.material_no.isin(common)]
summary = ev_c.groupby(["variant", "pattern", "model"])[["mae", "tot_err", "bias"]].mean().round(3)
print("\nBacktest TAM-aware (per pola):\n", summary.loc["tam_aware"].to_string())

aw = summary.loc["tam_aware"].reset_index()
best = aw.sort_values("tot_err").groupby("pattern").first()["model"]
for p in ["Sporadic", "No demand", "TAM-only"]:
    best[p] = best.get(p, "TSB") if p == "Sporadic" else "TSB"
print("\nModel terpilih per pola:", best.to_dict())

# ringkasan untuk deck: MA6 naive (cara lama) vs model terpilih TAM-aware
ev_c = ev_c.assign(sel=lambda d: d.model == d.pattern.map(best))
old = ev_c[(ev_c.variant == "naive") & (ev_c.model == "MA6")].set_index("material_no")
new = ev_c[(ev_c.variant == "tam_aware") & ev_c.sel].set_index("material_no")
naive_sel = ev_c[(ev_c.variant == "naive") & ev_c.sel].set_index("material_no")
HEAD = pd.Series({
    "parts_tested": len(common),
    "err_ma6_naive": old.tot_err.mean(), "err_selected_naive": naive_sel.tot_err.mean(),
    "err_selected_tam_aware": new.tot_err.mean(),
    "bias_ma6_naive": old.bias.mean(), "bias_selected_tam_aware": new.bias.mean(),
    "mae_ma6_naive": old.mae.mean(), "mae_selected_tam_aware": new.mae.mean(),
})
HEAD["improvement_vs_ma6"] = 1 - HEAD.err_selected_tam_aware / HEAD.err_ma6_naive
HEAD["tam_share"] = raw[TAM].values.sum() / raw.values.sum()
print("\nHeadline backtest:\n", HEAD.round(3).to_string())
ev = ev_all[ev_all.variant == "tam_aware"].drop(columns="variant")
ev["selected"] = (ev["model"] == ev["pattern"].map(best)).astype(int)

# ---------------------------------------------------------------- forecast final (fit pada 17 bulan)
FUT = pd.period_range("2026-10", "2027-03", freq="M")
fc_rows, dist = [], []
for m in clean.index:
    y = base.loc[m].values.astype(float)          # forecast rutin dilatih tanpa bulan TAM
    pat = cls.at[m, "pattern"]
    f = MODELS[best[pat]](y) if y.sum() > 0 else 0.0
    _, p, z = tsb(y, return_parts=True)
    nz = y[y > 0]
    size_mean = nz.mean() if len(nz) else 0.0
    size_sd = nz.std(ddof=0) if len(nz) > 1 else size_mean * 0.5
    p_month = min(max(p, 0.0), 1.0)
    dist.append(dict(material_no=m, fc_month=f, p_month=p_month, size_mean=size_mean, size_sd=size_sd,
                     need_prob_3m=1 - (1 - p_month) ** 3, model=best[pat]))
    # interval 80% dari model compound (Bernoulli x ukuran) -> pendekatan normal, dibatasi >= 0
    var = p_month * (size_sd ** 2 + size_mean ** 2) - (p_month * size_mean) ** 2
    sd = np.sqrt(max(var, 0))
    for mo in FUT:
        fc_rows.append(dict(material_no=m, month=str(mo), forecast=f, lower=max(0, f - 1.28 * sd),
                            upper=f + 1.28 * sd, need_prob=p_month))
stat_fc = pd.DataFrame(fc_rows)
dist = pd.DataFrame(dist).set_index("material_no")

pd.to_pickle({"raw": raw, "clean": clean}, os.path.join(INT, "series.pkl"))
cls.to_pickle(os.path.join(INT, "demand_class.pkl"))
ev.to_pickle(os.path.join(INT, "model_eval.pkl"))
summary.to_pickle(os.path.join(INT, "backtest_summary.pkl"))
HEAD.to_pickle(os.path.join(INT, "forecast_headline.pkl"))
stat_fc.to_pickle(os.path.join(INT, "stat_fc.pkl"))
dist.to_pickle(os.path.join(INT, "demand_dist.pkl"))
pd.Series([str(p) for p in TAM]).to_pickle(os.path.join(INT, "tam_months.pkl"))

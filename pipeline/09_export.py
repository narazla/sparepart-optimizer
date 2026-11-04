"""
EXPORT -> output/spareparts.db
Membaca semua hasil di data/interim. Jika hasil tahap lanjut (lead time, optimasi, TAM) belum ada,
tabel diisi perhitungan SEMENTARA (provisional) agar aplikasi tetap bisa dibangun dengan data asli.
"""
import os, sqlite3
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT, OUT = os.path.join(ROOT, "data", "interim"), os.path.join(ROOT, "output", "spareparts.db")
P = lambda n: os.path.join(INT, f"{n}.pkl")
has = lambda n: os.path.exists(P(n))
load = lambda n: pd.read_pickle(P(n))
TODAY = pd.Timestamp("2026-10-01")
SL_DEFAULT = {"A": 0.99, "B": 0.97, "C": 0.93, "S": 0.90}

mat, cls, dist = load("mat"), load("demand_class"), load("demand_dist")
series, pp = load("series"), load("pp")
sm = mat[mat.in_scope].copy()
sm["mrp_type"] = sm["mrp"]
if has("sourcing"):
    sm = sm.join(load("sourcing")[["supplier_normal", "price_normal", "supplier_urgent", "lt_urgent", "urgent_source"]])

# ---------------------------------------------------------------- lead time (tahap 4 / sementara)
if has("lead_time"):
    lt = load("lead_time")
else:
    v = pp[pp.lt_valid]
    lt = v.groupby("PR-Material").agg(lt_p50=("lead_time", "median"), lt_p80=("lead_time", lambda x: x.quantile(.8)),
                                      late_rate=("is_late", "mean"),
                                      main_supplier=("PO-Supplier Name", lambda x: x.mode().iat[0]))
    glob = dict(lt_p50=v.lead_time.median(), lt_p80=v.lead_time.quantile(.8), late_rate=v.is_late.mean())
    lt = lt.reindex(sm.index)
    for k, val in glob.items():
        lt[k] = lt[k].fillna(val)
sm = sm.join(lt[["lt_p50", "lt_p80", "late_rate", "main_supplier"]])
sm = sm.join(cls[["pattern", "adi", "cv2", "demand_months", "total_issued"]])
mv = load("mv")
sm["last_issue_date"] = mv.groupby("Material Number")["Posting Date"].max().dt.strftime("%Y-%m-%d").reindex(sm.index)
sm["notif_12m"] = load("notif_signal").reindex(sm.index).fillna(0).astype(int) if has("notif_signal") else 0
tw = load("twins")
sm["twin_group"] = tw[tw.is_primary == 1].set_index("material_no")["twin_group"].reindex(sm.index)

# ---------------------------------------------------------------- parameter & rekomendasi (tahap 6 / sementara)
if has("parameters") and has("recommendations"):
    params, recs = load("parameters"), load("recommendations")
else:
    d = dist.reindex(sm.index)
    mu_d = d["p_month"] * d["size_mean"] / 30.4                              # demand per hari
    var_m = d["p_month"] * (d["size_sd"] ** 2 + d["size_mean"] ** 2) - (d["p_month"] * d["size_mean"]) ** 2
    sd_d = np.sqrt(var_m.clip(lower=0) / 30.4)
    L, sdL = sm["lt_p50"], (sm["lt_p80"] - sm["lt_p50"]) / 0.84
    sl = sm["criticality"].map(SL_DEFAULT)
    z = norm.ppf(sl)
    ss = np.ceil(z * np.sqrt(L * sd_d ** 2 + (mu_d ** 2) * sdL ** 2))
    rop = np.ceil(mu_d * L + ss)
    eoq = np.ceil(np.sqrt(2 * (mu_d * 365) * 1_500_000 / (0.20 * sm["price"].clip(lower=1)))).clip(lower=1)
    params = pd.DataFrame({"service_level": sl, "safety_stock": ss, "rop": rop, "min_new": rop,
                           "order_qty": np.where(mu_d > 0, eoq, 0)})
    params["max_new"] = params["rop"] + params["order_qty"]
    stock = sm["stock"]
    insurance = (sm["pattern"].isin(["No demand"])) & (sm["criticality"] == "A")
    status = np.select(
        [(mu_d > 0) & (stock <= params["rop"]),
         insurance,
         (sm["pattern"] == "No demand") & (stock > 0),
         (mu_d > 0) & (stock > params["max_new"]) & (stock / (mu_d * 365).replace(0, np.nan) > 2),
         (mu_d > 0) & (stock > params["max_new"])],
        ["Stockout Risk", "Insurance Spare", "Dead", "Slow-moving", "Overstock"], "Healthy")
    action = pd.Series(status, index=sm.index).map({"Stockout Risk": "Reorder", "Insurance Spare": "Keep",
                                                     "Dead": "Review/Dispose", "Slow-moving": "Reduce",
                                                     "Overstock": "Reduce", "Healthy": "Maintain"})
    action[(sm["has_anomaly"] == 1) & (action == "Maintain")] = "Monitor"
    qty = np.where(action == "Reorder", (params["max_new"] - stock).clip(lower=0),
                   np.where(action.isin(["Reduce"]), (stock - params["max_new"]).clip(lower=0),
                            np.where(action == "Review/Dispose", stock, 0)))
    days_cover = (stock - ss).clip(lower=0) / mu_d.replace(0, np.nan)
    order_by = TODAY + pd.to_timedelta((days_cover - sm["lt_p80"]).clip(lower=0, upper=730).fillna(0), unit="D")
    value = np.where(action == "Reorder", -qty * sm["price"], qty * sm["price"])
    risk = (1 - d["need_prob_3m"].fillna(0) * (stock <= ss)).rsub(1).clip(0, 1)
    crit_w = sm["criticality"].map({"A": 4, "B": 3, "C": 2, "S": 1})
    recs = pd.DataFrame({"status": status, "action": action, "qty": qty,
                         "order_by_date": np.where(action == "Reorder", order_by.dt.strftime("%Y-%m-%d"), None),
                         "value_impact": value, "stockout_risk": risk,
                         "priority": crit_w * 25 * (action == "Reorder") + risk * 10, "tam_linked": (sm["pattern"] == "TAM-only").astype(int),
                         "reason": "Provisional recommendation (v0) - final logic pending"}, index=sm.index)

# ---------------------------------------------------------------- tulis DB
if os.path.exists(OUT):
    os.remove(OUT)
con = sqlite3.connect(OUT)
con.executescript(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")).read())

def put(table, df):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    df = df.reset_index() if df.index.name else df
    df = df[[c for c in cols if c in df.columns]].replace({np.nan: None})
    con.executemany(f"INSERT INTO {table} ({','.join(df.columns)}) VALUES ({','.join('?' * len(df.columns))})",
                    df.itertuples(index=False, name=None))
    print(f"{table:20s} {len(df):>7,} rows")

sm.index.name = "material_no"
put("parts", sm)
raw = series["raw"].reindex(sm.index)
an = mv[mv.is_anomaly].assign(month=lambda x: x["Posting Date"].dt.to_period("M")).groupby(["Material Number", "month"]).size()
dm = raw.stack().rename("qty").reset_index(); dm.columns = ["material_no", "month", "qty"]
dm = dm[dm.qty > 0]
dm["is_anomaly"] = [int((m, p) in an.index) for m, p in zip(dm.material_no, dm.month)]
dm["month"] = dm["month"].astype(str)
put("demand_monthly", dm)
fc = load("stat_fc")
if has("need_prob"):
    npb = load("need_prob")
    fc = fc.drop(columns=["need_prob"]).merge(npb, on=["material_no", "month"], how="left")
    fc["need_prob"] = fc["need_prob_gbm"]
    # forecast rutin = peluang (model global) x ukuran rata-rata saat dipakai
    fc["forecast"] = fc["need_prob"] * fc["material_no"].map(dist["size_mean"]).fillna(0)
    fc["lower"] = np.minimum(fc["lower"], fc["forecast"])
    fc["upper"] = np.maximum(fc["upper"], fc["forecast"])
if "pm_due" not in fc.columns:
    fc["pm_due"] = 0
put("forecasts", fc[fc.material_no.isin(sm.index)])
put("model_eval", load("model_eval").rename(columns={"mae": "mase"}))
params.index.name = "material_no"; recs.index.name = "material_no"
put("parameters", params)
put("recommendations", recs)
for t in ["maintenance_events", "event_parts", "backtest_monthly", "scenario_grid", "kit_rules", "suppliers"]:
    if has(t):
        put(t, load(t))
put("twin_groups", tw)
put("data_quality_log", load("dq_log"))
put("data_quality_score", load("dq_score"))

# KPI
val_a = sm.loc[sm.price_source == "actual", "stock_value"].sum()
rel = recs["action"].isin(["Reduce", "Review/Dispose"])
act = sm["price_source"] == "actual"
excess = recs.loc[rel, "value_impact"].sum()
excess_a = recs.loc[rel & act, "value_impact"].sum()
dead = recs.loc[recs.status == "Dead", "value_impact"].sum()
reorder_inv = -recs.loc[recs.action.isin(["Reorder", "Plan for TAM"]), "value_impact"].sum()
kp = [("inventory_value", sm.stock_value.sum(), "Total inventory value (incl. estimated prices)"),
      ("inventory_value_actual", val_a, "Inventory value, parts with actual PO price"),
      ("releasable_value", excess, "Releasable excess + dead stock (incl. estimated prices)"),
      ("releasable_value_actual", excess_a, "Releasable excess + dead stock, actual-price parts only"),
      ("releasable_share_actual", excess_a / val_a, "Share of actual-priced inventory value that can be released"),
      ("reorder_investment", reorder_inv, "Investment needed for reorders + TAM preparation"),
      ("dead_value", dead, "Dead stock value"),
      ("stock_items", int((sm.mrp == "Stock Item").sum()), "Stock items (after merging duplicates)"),
      ("parts_in_scope", len(sm), "Parts optimised (stock items + Non Stock conversion candidates)"),
      ("critical_stockout_risk", int(((recs.status == "Stockout Risk") & (sm.criticality == "A")).sum()), "Criticality-A parts at stockout risk"),
      ("critical_zero_stock", int(((sm.stock == 0) & (sm.criticality == "A") & (sm.total_issued > 0) & (sm.mrp == "Stock Item")).sum()), "Criticality-A parts used but zero stock"),
      ("reorder_count", int((recs.action == "Reorder").sum()), "Parts to reorder"),
      ("dead_count", int((recs.status == "Dead").sum()), "Dead stock items"),
      ("insurance_count", int((recs.status == "Insurance Spare").sum()), "Insurance spares kept"),
      ("twin_groups", int(tw.twin_group.nunique()), "Duplicate material groups merged"),
      ("lt_median", float(pp[pp.lt_valid].lead_time.median()), "Median lead time (days)"),
      ("late_rate", float(pp[pp.late_valid].is_late.mean()), "POs delivered after required date"),
      ("tam_share", float(load("forecast_headline")["tam_share"]), "Share of consumption during TAM window")]
if has("forecast_headline"):
    h = load("forecast_headline")
    kp += [("fc_improvement", float(h.improvement_vs_ma6), "Forecast error reduction vs moving average"),
           ("fc_bias_old", float(h.bias_ma6_naive), "Over-forecast bias, old method"),
           ("fc_bias_new", float(h.bias_selected_tam_aware), "Over-forecast bias, proposed policy")]
for extra in ("kpis_extra", "kpis_tam", "kpis_sourcing"):
    if has(extra):
        kp += [tuple(r) for r in load(extra).itertuples(index=False)]
if has("gbm_eval"):
    g = load("gbm_eval")
    kp += [("gbm_auc", g.at["roc_auc", "global_gbm"], "Need-probability model ROC-AUC"),
           ("gbm_auc_base", g.at["roc_auc", "historical_frequency"], "Baseline ROC-AUC (historical frequency)"),
           ("gbm_capture", g.at["top10pct_capture", "global_gbm"], "Share of real demand caught in top 10% risk scores")]
if has("lt_eval"):
    e = load("lt_eval")
    kp += [("lt_improvement_p80", e.at[0.8, "improvement"], "Lead-time P80 error reduction vs historical quantile"),
           ("lt_coverage_p80", e.at[0.8, "coverage_model"], "Share of orders arriving within predicted P80")]
if has("nlp_eval"):
    kp += [("nlp_accuracy", load("nlp_eval")["cv_accuracy"], "Notification classifier CV accuracy")]
if has("price_cv"):
    pc = load("price_cv")
    kp += [("price_within_2x", pc["knn_within_2x"], "Price model: estimates within 2x of true price")]
kp += [("stockout_risk_active", int((recs.stockout_risk > 0.2).sum()), "Parts with >20% chance of running out before replenishment")]
put("kpis", pd.DataFrame(kp, columns=["key", "value", "label"]))
con.commit(); con.close()
print("\nwritten", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB")

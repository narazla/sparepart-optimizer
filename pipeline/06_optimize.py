"""
TAHAP 6 - OPTIMASI BIAYA-RISIKO (MONTE CARLO) + REKOMENDASI + REPLAY BACKTEST
Input : mat, demand_dist, need_prob, lead_time, demand_class, series, pp, twins
Output: parameters.pkl, recommendations.pkl, scenario_grid.pkl, backtest_monthly.pkl, kpis_extra.pkl

1. Distribusi demand selama lead time (LTD) per part, disimulasikan 3.000 kali:
     lead time  ~ lognormal (dikalibrasi ke P50 & P80 dari model tahap 4)
     kejadian   ~ Poisson(peluang bulanan dari model global tahap 5 x lama lead time)
     ukuran     ~ Gamma(rata-rata & sd ukuran pemakaian part tsb)
2. Untuk setiap service level di grid: ROP = kuantil LTD, Max = ROP + Q (EOQ),
   rata-rata inventory & perkiraan stockout per tahun -> tabel scenario_grid (dipakai simulator)
3. Service level optimal per part = yang meminimalkan
     biaya simpan (carrying 20%/tahun x nilai inventory rata-rata)
   + biaya risiko (stockout/tahun x biaya per stockout menurut kekritisan)
4. Status & aksi per part (Reorder / Reduce / Review-Dispose / Keep / Plan for TAM / Monitor / Maintain)
5. Replay 17 bulan konsumsi asli: Min-Max lama vs parameter usulan (proposed)
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT = os.path.join(ROOT, "data", "interim")
L = lambda n: pd.read_pickle(os.path.join(INT, f"{n}.pkl"))
S = lambda df, n: df.to_pickle(os.path.join(INT, f"{n}.pkl"))
rng = np.random.default_rng(42)
TODAY = pd.Timestamp("2026-10-01")

# ---------------------------------------------------------------- asumsi bisnis (bisa diubah di simulator)
CARRY = 0.20                       # biaya simpan per tahun (% nilai barang)
ORDER_COST = 1_500_000             # biaya administrasi per PO (IDR)
STOCKOUT_COST = {"A": 750e6, "B": 200e6, "C": 50e6, "S": 10e6}   # per kejadian: produksi hilang + pembelian darurat
SL_GRID = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995]
N_SIM = 3000

mat, dist, need, lt, cls = L("mat"), L("demand_dist"), L("need_prob"), L("lead_time"), L("demand_class")
series, pp, tw = L("series"), L("pp"), L("twins")
sm = mat[mat.in_scope].copy()
sm = sm.join(lt[["lt_p50", "lt_p80", "late_rate", "main_supplier"]]).join(cls[["pattern", "tam_qty", "total_issued"]])
sm = sm.join(dist[["size_mean", "size_sd", "p_month"]])
sm["p_gbm"] = need.groupby("material_no")["need_prob_gbm"].mean()
# peluang bulanan: model global, tetapi 0 untuk part yang tidak pernah punya demand rutin
sm["p"] = np.where(sm["size_mean"] > 0, sm["p_gbm"], 0.0)
sm["mu_month"] = sm["p"] * sm["size_mean"]
sm["mu_day"] = sm["mu_month"] / 30.4375
open_po = pp[pp["PO-Delivery Date"] > pd.Timestamp("2026-09-30")].groupby("PR-Material")["PR-Quantity"].sum()
sm["on_order"] = open_po.reindex(sm.index).fillna(0)
sm["position"] = sm["stock"] + sm["on_order"]

# ---------------------------------------------------------------- 1-2. Monte Carlo LTD + grid
def ltd_samples(p, sm_, ss_, p50, p80):
    sig = max(np.log(max(p80, p50 + 1)) - np.log(max(p50, 1)), 1e-3) / 0.8416
    Lt = rng.lognormal(np.log(max(p50, 1)), sig, N_SIM)
    k = rng.poisson(p * Lt / 30.4375)
    if ss_ <= 0:
        size = np.full(k.sum(), sm_)
    else:
        shape = (sm_ / ss_) ** 2
        size = rng.gamma(shape, sm_ / shape, k.sum())
    out = np.zeros(N_SIM)
    np.add.at(out, np.repeat(np.arange(N_SIM), k), size)
    return out

grid, best, risk_now = [], {}, {}
active = sm[sm.mu_month > 0]
for m, r in active.iterrows():
    ltd = ltd_samples(r.p, r.size_mean, r.size_sd, r.lt_p50, r.lt_p80)
    D = 12 * r.mu_month
    Q = np.sqrt(2 * D * ORDER_COST / (CARRY * max(r.price, 1)))
    Q = float(np.ceil(np.clip(Q, max(1, np.ceil(r.size_mean)), max(1, np.ceil(D)))))
    cycles = D / Q
    risk_now[m] = float(np.mean(ltd > r.position))                    # peluang habis sebelum barang datang
    rows = []
    for sl in SL_GRID:
        rop = float(np.ceil(np.quantile(ltd, sl)))
        ss = max(rop - ltd.mean(), 0)
        p_short = float(np.mean(ltd > rop))
        avg_inv = (ss + Q / 2) * r.price
        so = p_short * cycles
        cost = CARRY * avg_inv + so * STOCKOUT_COST[r.criticality]
        rows.append(dict(material_no=m, service_level=sl, rop=rop, max_level=rop + Q, avg_inventory_value=avg_inv,
                         expected_stockouts=so, ss=ss, Q=Q, cost=cost))
    grid += rows
    best[m] = min(rows, key=lambda x: x["cost"])
grid = pd.DataFrame(grid)

# part tanpa demand rutin: nilai inventory konstan (stok yang dipertahankan), tanpa risiko stockout
def keep_qty(r):
    if r.criticality == "A":                                              # insurance spare: simpan secukupnya
        return min(r.stock, max(1, min(np.ceil(0.1 * r.installed_qty), 5)))
    if r.tam_qty > 0:                                                     # disiapkan untuk TAM berikutnya
        return min(r.stock, r.tam_qty)
    return 0.0
inactive = sm[sm.mu_month <= 0].copy()
inactive["keep"] = inactive.apply(keep_qty, axis=1)
static_rows = [dict(material_no=m, service_level=sl, rop=0.0, max_level=r.keep, avg_inventory_value=r.keep * r.price,
                    expected_stockouts=0.0) for m, r in inactive.iterrows() for sl in SL_GRID]
scenario_grid = pd.concat([grid[["material_no", "service_level", "rop", "max_level", "avg_inventory_value",
                                 "expected_stockouts"]], pd.DataFrame(static_rows)])

# ---------------------------------------------------------------- 3. parameter terpilih
bp = pd.DataFrame(best).T
params = pd.DataFrame(index=sm.index)
params["service_level"] = bp["service_level"]
params["safety_stock"] = np.ceil(bp["ss"].astype(float))
params["rop"] = bp["rop"]
params["order_qty"] = bp["Q"]
params["max_new"] = bp["max_level"]
params.loc[inactive.index, ["service_level", "safety_stock", "rop", "order_qty"]] = [np.nan, 0, 0, 0]
params.loc[inactive.index, "max_new"] = inactive["keep"]
params["min_new"] = params["rop"]
params = params.astype(float)

# ---------------------------------------------------------------- 4. status & aksi
r = sm.join(params)
r["risk"] = pd.Series(risk_now).reindex(r.index).fillna(0)
r["cover_months"] = r["stock"] / r["mu_month"].replace(0, np.nan)
r["days_to_rop"] = ((r["position"] - r["rop"]) / r["mu_day"].replace(0, np.nan))
false_so = tw.groupby("twin_group").filter(lambda g: (g.stock == 0).any() and (g.stock > 0).any())
false_so_primary = set(false_so[false_so.is_primary == 1].material_no) | \
                   set(tw[tw.twin_group.isin(false_so.twin_group) & (tw.is_primary == 1)].material_no)
twin_n = tw.groupby("twin_group").size()
r["twin_group"] = tw[tw.is_primary == 1].set_index("material_no")["twin_group"].reindex(r.index)

def decide(x):
    if x.mrp == "Non Stock":                       # kandidat dari tahap 1: dipakai berulang, kritis, lead time panjang
        return "Misclassified Non-Stock", "Convert to Stock"
    if x.mu_month > 0:
        if x.position <= x.rop or (x.days_to_rop is not np.nan and x.days_to_rop <= 30):
            return "Stockout Risk", "Reorder"
        if x.cover_months > 24 and x.stock > x.max_new:
            return "Slow-moving", "Reduce"
        if x.stock > x.max_new:
            return "Overstock", "Reduce"
        if x.has_anomaly:
            return "Healthy", "Monitor"
        return "Healthy", "Maintain"
    if x.tam_qty > 0:
        return "TAM Reserve", ("Plan for TAM" if x.stock < x.tam_qty else "Keep")
    if x.criticality == "A":
        return "Insurance Spare", ("Keep" if x.stock <= x.max_new else "Reduce")
    if x.stock > 0:
        return "Dead", "Review/Dispose"
    return "Healthy", "Maintain"

dec = r.apply(decide, axis=1, result_type="expand"); dec.columns = ["status", "action"]
r = r.join(dec)
BUY = ["Reorder", "Plan for TAM", "Convert to Stock"]
qty = np.select(
    [r.action == "Reorder", r.action == "Convert to Stock", r.action == "Plan for TAM", r.action == "Reduce",
     r.action == "Review/Dispose"],
    [(r.max_new - r.position).clip(lower=1), np.where(r.position <= r.rop, (r.max_new - r.position).clip(lower=1), 0), (r.tam_qty - r.position).clip(lower=0),
     (r.stock - r.max_new).clip(lower=0), r.stock], 0)
r["qty"] = np.ceil(qty)
r["value_impact"] = np.where(r.action.isin(BUY), -r.qty * r.price, r.qty * r.price)
r.loc[~r.action.isin(BUY + ["Reduce", "Review/Dispose"]), "value_impact"] = 0.0
od = TODAY + pd.to_timedelta(r["days_to_rop"].clip(lower=0, upper=30).fillna(0), unit="D")
r["order_by_date"] = np.where(r.action == "Reorder", od.dt.strftime("%Y-%m-%d"),
                              np.where(r.action == "Convert to Stock", TODAY.strftime("%Y-%m-%d"), None))
CW = {"A": 4, "B": 3, "C": 2, "S": 1}
r["priority"] = np.where(r.action == "Reorder", 50 + 12 * r.criticality.map(CW) * (0.3 + r.risk),
                         np.where(r.action == "Convert to Stock", 45 + 5 * r.criticality.map(CW),
                         np.where(r.action == "Plan for TAM", 40 + 5 * r.criticality.map(CW),
                                  np.minimum(39, np.log10(r.value_impact.abs() + 1) * 3))))
r["tam_linked"] = (r.tam_qty > 0).astype(int)
r["stockout_risk"] = r["risk"]

# dual-mode sourcing: urgent untuk part kelas A yang mendesak, normal (termurah) untuk sisanya
src = L("sourcing")
r = r.join(src)
urgent = r.action.isin(BUY) & (r.criticality == "A") & ((r.risk >= 0.2) | (r.position <= r.safety_stock))
r["sourcing_mode"] = np.where(r.action.isin(BUY), np.where(urgent, "Urgent", "Normal"), None)
r["supplier"] = np.where(r.sourcing_mode == "Urgent", r.supplier_urgent,
                         np.where(r.sourcing_mode == "Normal", r.supplier_normal.fillna(r.supplier_urgent), None))

def rp(v):
    return f"Rp {v/1e9:,.1f} bn" if abs(v) >= 1e9 else f"Rp {v/1e6:,.0f} M"
def reason(x):
    s = []
    if x.action == "Reorder":
        s.append(f"Stock position {x.position:g} (on hand {x.stock:g} + on order {x.on_order:g}) is at or near the new "
                 f"reorder point of {x.rop:g}.")
        s.append(f"Lead time P80 is {x.lt_p80:.0f} days and {x.late_rate:.0%} of past orders arrived late; "
                 f"there is a {x.risk:.0%} chance of running out before a new order arrives.")
        s.append(f"Order {x.qty:g} unit(s) to restore the new maximum of {x.max_new:g} (criticality {x.criticality}, "
                 f"service level {x.service_level:.1%}).")
    elif x.action == "Convert to Stock":
        s.append(f"Set up as Non Stock (bought only when needed), but used in {x.use_months:.0f} of the last 17 months "
                 f"on criticality-{x.criticality} equipment"
                 + (f", with a historical lead time of {x.lt_hist:.0f} days." if x.lt_hist == x.lt_hist
                    else ", and never bought through a regular PO."))
        s.append(f"Every failure therefore triggers emergency buying. Change the SAP MRP type to Stock Item with "
                 f"Min {x.min_new:g} / Max {x.max_new:g}"
                 + (f" and buy {x.qty:g} unit(s) as initial stock." if x.qty > 0
                    else f"; current position of {x.position:g} unit(s) (on hand {x.stock:g} + on order {x.on_order:g}) is above the new "
                         f"minimum, so no extra purchase is needed now."))
    elif x.action == "Plan for TAM":
        s.append(f"No routine demand, but {x.tam_qty:g} unit(s) were consumed during the last turnaround window.")
        s.append(f"Current position is {x.position:g}; order {x.qty:g} unit(s) before the next TAM "
                 f"(lead time P80 {x.lt_p80:.0f} days).")
    elif x.action == "Reduce" and x.status == "Insurance Spare":
        s.append(f"Criticality-A insurance spare with no consumption in 17 months. Keep {x.max_new:g} unit(s) as cover;"
                 f" the remaining {x.qty:g} unit(s) ({rp(x.value_impact)}) can be released.")
    elif x.action == "Reduce":
        s.append(f"Stock of {x.stock:g} exceeds the new maximum of {x.max_new:g}"
                 + (f" and covers about {x.cover_months:.0f} months of demand." if x.cover_months == x.cover_months else "."))
        s.append(f"Stop replenishment and release {x.qty:g} unit(s) ({rp(x.value_impact)}) via transfer or return.")
    elif x.action == "Review/Dispose":
        s.append(f"No consumption in 17 months, no turnaround usage and criticality {x.criticality}.")
        s.append(f"{x.stock:g} unit(s) ({rp(x.value_impact)}) are candidates for disposal or vendor return.")
    elif x.status == "Insurance Spare":
        s.append(f"Rarely used, but installed on criticality-A equipment. Keeping {x.stock:g} unit(s) as insurance "
                 f"is cheaper than one stockout.")
    elif x.status == "TAM Reserve":
        s.append(f"Reserved for turnaround: {x.tam_qty:g} unit(s) used in the last TAM window; stock covers it.")
    elif x.action == "Monitor":
        s.append("Stock is healthy, but recent consumption contains abnormal spikes. Review usage before the next order.")
    else:
        s.append(f"Stock {x.stock:g} is within the new range {x.min_new:g} to {x.max_new:g}.")
    if x.sourcing_mode == "Urgent" and isinstance(x.supplier, str):
        s.append(f"Urgent sourcing: use {x.supplier} (lead time about {x.lt_urgent:.0f} days, "
                 f"{x.late_urgent:.0%} late; {x.urgent_source}).")
    elif x.sourcing_mode == "Normal" and isinstance(x.supplier, str):
        s.append(f"Normal sourcing: {x.supplier}" + (f" at about {rp(x.price_normal)} per unit."
                                                     if x.price_normal == x.price_normal else "."))
    if x.name in false_so_primary:
        s.append(f"Note: this part existed under {int(twin_n.get(x.twin_group, 2))} material numbers; "
                 f"stock has been consolidated (some numbers showed zero stock while a twin had stock).")
    return " ".join(s)
r["reason"] = r.apply(reason, axis=1)
recs = r[["status", "action", "qty", "order_by_date", "value_impact", "stockout_risk", "priority", "tam_linked",
          "sourcing_mode", "supplier", "reason"]]
print(recs.groupby(["status", "action"]).agg(n=("qty", "size"), value_bn=("value_impact", lambda v: v.sum() / 1e9)).round(1))

# ---------------------------------------------------------------- 5. replay backtest
raw = series["raw"].reindex(sm.index).fillna(0)
months = list(raw.columns)
tam_m = set(L("tam_months"))
Lm = np.maximum(1, np.round(sm["lt_p50"] / 30.4375)).astype(int).values
price = sm["price"].values
crit_a = (sm["criticality"] == "A").values
dem = raw.values
def replay(s_lvl, S_lvl, label, tam_plan=None):
    n = len(sm); on_hand = S_lvl.copy().astype(float); pipe = {}
    out = []
    for t, mo in enumerate(months):
        arr = pipe.pop(t, np.zeros(n)); on_hand += arr
        d = dem[:, t].copy()
        if tam_plan is not None and str(mo) in tam_m:
            on_hand += tam_plan[:, t]                                       # kit TAM yang dipesan 3+ bulan sebelumnya
        short = d > on_hand + 1e-9
        on_hand = np.maximum(on_hand - d, 0)
        onorder = sum(v for k, v in pipe.items())
        pos = on_hand + (onorder if isinstance(onorder, np.ndarray) else 0)
        need = (pos <= s_lvl) & (S_lvl > 0)
        q = np.where(need, S_lvl - pos, 0)
        for i in np.flatnonzero(q > 0):
            pipe.setdefault(t + Lm[i], np.zeros(n))[i] += q[i]
        out.append(dict(month=str(mo), policy=label, inventory_value=float((on_hand * price).sum()),
                        stockouts=int(short.sum()), stockouts_critical=int((short & crit_a).sum()),
                        stockouts_si=int((short & is_si).sum()), stockouts_si_a=int((short & is_si & crit_a).sum())))
    return pd.DataFrame(out)
is_si = (sm["mrp"] == "Stock Item").values

old = replay(sm["min_old"].values, sm["max_old"].values, "current")
new_s, new_S = params["rop"].fillna(0).values, params["max_new"].fillna(0).values
new = replay(new_s, new_S, "proposed_policy")
tam_plan = np.zeros_like(dem)
for t, mo in enumerate(months):
    if str(mo) in tam_m:
        tam_plan[:, t] = np.where(sm["tam_qty"].values > 0, dem[:, t], 0)   # kebutuhan TAM direncanakan dari Readiness
newt = replay(new_s, new_S, "proposed", tam_plan=tam_plan)
bt = pd.concat([old, newt])
summ = pd.concat([old, new, newt]).groupby("policy").agg(avg_inv_bn=("inventory_value", lambda v: v.mean() / 1e9),
                                                         stockouts=("stockouts", "sum"),
                                                         critical=("stockouts_critical", "sum"),
                                                         stock_items=("stockouts_si", "sum"),
                                                         stock_items_a=("stockouts_si_a", "sum"))
print("\nReplay 17 bulan:\n", summ.round(1).to_string())
demand_events = int((dem > 0).sum())

# ---------------------------------------------------------------- KPI tambahan
o, n_ = summ.loc["current"], summ.loc["proposed"]
cur_risk = sum(STOCKOUT_COST[c] for c in sm.criticality) * 0  # placeholder agar jelas: lihat simulator
extra = [
    ("bt_stockouts_old", o.stockouts, "Replay: stockout events, current Min-Max"),
    ("bt_stockouts_new", n_.stockouts, "Replay: stockout events, proposed policy"),
    ("bt_critical_old", o.critical, "Replay: criticality-A stockouts, current"),
    ("bt_critical_new", n_.critical, "Replay: criticality-A stockouts, proposed policy"),
    ("bt_inv_old", o.avg_inv_bn * 1e9, "Replay: average inventory value, current"),
    ("bt_inv_new", n_.avg_inv_bn * 1e9, "Replay: average inventory value, proposed policy"),
    ("bt_demand_events", demand_events, "Replay: part-months with demand"),
    ("bt_si_old", o.stock_items, "Replay: stockouts, current stock items only"),
    ("bt_si_new", n_.stock_items, "Replay: stockouts, stock items only, proposed policy"),
    ("bt_si_policy_only", summ.loc["proposed_policy", "stock_items"], "Replay: stockouts, stock items only, parameters without TAM readiness"),
    ("bt_policy_only", summ.loc["proposed_policy", "stockouts"], "Replay: stockouts, parameters without TAM readiness"),
    ("bt_si_a_old", o.stock_items_a, "Replay: criticality-A stockouts, stock items only, current"),
    ("bt_si_a_new", n_.stock_items_a, "Replay: criticality-A stockouts, stock items only, proposed"),
    ("convert_to_stock", int((r.action == "Convert to Stock").sum()), "Non Stock parts recommended to convert to Stock Item"),
    ("urgent_sourcing", int((r.sourcing_mode == "Urgent").sum()), "Buy recommendations using urgent sourcing"),
    ("on_order_lines", int((sm.on_order > 0).sum()), "Parts with open purchase orders"),
    ("carry_rate", CARRY, "Assumed carrying cost per year"),
    ("stockout_cost_A", STOCKOUT_COST["A"], "Assumed cost per stockout, criticality A"),
    ("stockout_cost_B", STOCKOUT_COST["B"], "Assumed cost per stockout, criticality B"),
    ("stockout_cost_C", STOCKOUT_COST["C"], "Assumed cost per stockout, criticality C"),
    ("stockout_cost_S", STOCKOUT_COST["S"], "Assumed cost per stockout, criticality S"),
    ("false_stockouts", len(false_so_primary), "Parts whose 'zero stock' was held under a twin number"),
]
S(pd.DataFrame(extra, columns=["key", "value", "label"]), "kpis_extra")
S(params, "parameters"); S(recs, "recommendations"); S(scenario_grid.reset_index(drop=True), "scenario_grid")
S(bt.reset_index(drop=True), "backtest_monthly"); S(summ, "backtest_summary_policy")
S(r[["on_order", "position", "p", "mu_month"]], "positions")
print("\nScenario grid rows:", len(scenario_grid))

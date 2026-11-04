"""
TAHAP 7 - KIT PART (ASSOCIATION RULES) + TAM & SHUTDOWN READINESS
Input : mv, mp, bom, mat, lead_time, positions, demand_class
Output: kit_rules.pkl, maintenance_events.pkl, event_parts.pkl

A. Association rules: setiap maintenance order = satu "keranjang" part yang dikeluarkan bersama.
   support = % order yang berisi A & B, confidence = P(B | A), lift = confidence / P(B).
   Dipakai untuk melengkapi daftar part sebuah pekerjaan ("biasanya juga butuh ...").
B. Event maintenance 12 bulan ke depan dari Mplan (siklus >= 1 bulan) -> FuncLoc -> part di BoM.
   Kebutuhan dihitung untuk part aus (wear parts) sebesar jumlah terpasang di FuncLoc tsb,
   ditambah part dari kit rules. Readiness = % part yang stoknya (on hand + on order) mencukupi.
C. Skenario TAM: lingkup TAM Jan-2026 (part yang dipakai pada window TAM) sebagai event perencanaan.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INT = os.path.join(ROOT, "data", "interim")
L = lambda n: pd.read_pickle(os.path.join(INT, f"{n}.pkl"))
S = lambda df, n: df.to_pickle(os.path.join(INT, f"{n}.pkl"))
mv, mp, bom, mat, lt, pos, cls, fl = L("mv"), L("mp"), L("bom"), L("mat"), L("lead_time"), L("positions"), L("demand_class"), L("fl")
TODAY = pd.Timestamp("2026-10-01"); HORIZON = TODAY + pd.DateOffset(months=12)
sm = mat[mat.mrp == "Stock Item"]

# ================================================================ A. association rules
baskets = mv.groupby("Order")["Material Number"].apply(lambda x: sorted(set(x)))
baskets = baskets[baskets.map(len) >= 2]
n_orders = mv["Order"].nunique()
item_cnt = Counter(m for b in mv.groupby("Order")["Material Number"].apply(set) for m in b)
pair_cnt = Counter(p for b in baskets for p in combinations(b, 2))
rules = []
for (a, b), c in pair_cnt.items():
    if c < 3:
        continue
    for x, y in ((a, b), (b, a)):
        conf = c / item_cnt[x]
        lift = conf / (item_cnt[y] / n_orders)
        if conf >= 0.3 and lift >= 2:
            rules.append(dict(material_a=x, material_b=y, support=c / n_orders, confidence=conf, lift=lift))
kit = pd.DataFrame(rules)
print(f"Orders {n_orders:,} | multi-part orders {len(baskets):,} | rules {len(kit):,} "
      f"| parts with a kit {kit.material_a.nunique() if len(kit) else 0}")
S(kit, "kit_rules")

# ================================================================ B. event maintenance
WEAR = {"GASKET", "SEAL", "O-RING", "FILTER ELEMENT", "PACKING", "BEARING", "STUDBOLT", "SPRING", "TRIM SET",
        "SEAT RING", "COUPLING", "IMPELLER", "BELT", "MECHANICAL SEAL", "LAMP", "BRUSH"}
ev = mp[(mp.next_due >= TODAY) & (mp.next_due < HORIZON) & (mp.cycle_months >= 1)].copy()
ev = ev.merge(fl[["Functional Location", "site", "equip_class"]], on="Functional Location", how="left")
b = bom[bom["Material No"].isin(sm.index)][["Functional Loc.", "Material No", "Installed Qty"]] \
    .groupby(["Functional Loc.", "Material No"], as_index=False)["Installed Qty"].sum()
b["category"] = b["Material No"].map(sm["category"])
b = b[b["category"].isin(WEAR)]
avail = pos["position"]
lt80 = lt["lt_p80"]

events, eparts = [], []
eid = 0
kit_by_a = kit.groupby("material_a") if len(kit) else None
for row in ev.itertuples(index=False):
    parts = b[b["Functional Loc."] == row[0]]
    if parts.empty:
        continue
    eid += 1
    due = row.next_due
    req = {int(m): float(np.ceil(q)) for m, q in zip(parts["Material No"], parts["Installed Qty"])}
    src = {m: "bom" for m in req}
    if kit_by_a is not None:
        for m in list(req):
            if m in kit_by_a.groups:
                for kb in kit_by_a.get_group(m).sort_values("lift", ascending=False).head(3).material_b:
                    if kb in sm.index and kb not in req:
                        req[int(kb)], src[int(kb)] = 1.0, "kit"
    short = 0
    for m, q in req.items():
        a = float(avail.get(m, 0))
        ob = due - pd.Timedelta(days=float(lt80.get(m, 60)))
        sf = max(q - a, 0)
        short += sf > 0
        eparts.append(dict(event_id=eid, material_no=m, required_qty=q, available_qty=a, shortfall=sf,
                           order_by_date=ob.strftime("%Y-%m-%d"), source=src[m]))
    events.append(dict(event_id=eid, funcloc=row[0], funcloc_desc=row[1], site=row.site, system=row.equip_class,
                       criticality=row[2], task=row[5], cycle_months=row.cycle_months,
                       due_date=due.strftime("%Y-%m-%d"), n_parts=len(req), n_short=short,
                       readiness=1 - short / len(req)))

# ================================================================ C. skenario TAM
tam_parts = cls[(cls.tam_qty > 0)].index.intersection(sm.index)
tam_due = pd.Timestamp("2027-01-15")       # tanggal skenario; di aplikasi bisa diubah planner
eid += 1
short = 0
for m in tam_parts:
    q = float(cls.at[m, "tam_qty"]); a = float(avail.get(m, 0)); sf = max(q - a, 0); short += sf > 0
    eparts.append(dict(event_id=eid, material_no=int(m), required_qty=q, available_qty=a, shortfall=sf,
                       order_by_date=(tam_due - pd.Timedelta(days=float(lt80.get(m, 60)))).strftime("%Y-%m-%d"),
                       source="tam"))
events.append(dict(event_id=eid, funcloc="TAM-SCENARIO", funcloc_desc="Turnaround scenario: repeat of Jan-2026 TAM scope",
                   site="PX+PY", system="Turnaround", criticality="A", task="Turnaround maintenance (TAM)",
                   cycle_months=None, due_date=tam_due.strftime("%Y-%m-%d"), n_parts=len(tam_parts), n_short=short,
                   readiness=1 - short / max(len(tam_parts), 1)))
events, eparts = pd.DataFrame(events), pd.DataFrame(eparts)
S(events, "maintenance_events"); S(eparts, "event_parts")
e = events[events.funcloc != "TAM-SCENARIO"]
print(f"Maintenance events (12 mo, with wear parts): {len(e):,} | parts lines {len(eparts):,}")
print(f"Events fully ready {(e.readiness == 1).mean():.0%} | events with shortage {(e.n_short > 0).sum():,}")
t = events.iloc[-1]
print(f"TAM scenario: {t.n_parts} parts, {t.n_short} short, readiness {t.readiness:.0%}")
late = eparts[(eparts.shortfall > 0) & (pd.to_datetime(eparts.order_by_date) < TODAY)]
print(f"Shortages already past order-by date (cannot arrive in time with P80 lead time): {len(late):,}")
S(pd.DataFrame([("events_12m", len(e), "Maintenance events in next 12 months (with wear parts)"),
                ("events_at_risk", int((e.n_short > 0).sum()), "Events with at least one part short"),
                ("tam_parts", int(t.n_parts), "Parts in TAM scope"),
                ("tam_short", int(t.n_short), "TAM parts short"),
                ("tam_readiness", float(t.readiness), "TAM readiness"),
                ("kit_rules", len(kit), "Co-usage rules learned from maintenance orders"),
                ("late_shortages", len(late), "Shortages already past order-by date")],
               columns=["key", "value", "label"]), "kpis_tam")

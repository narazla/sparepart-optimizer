"""Buat spareparts_dummy.db dengan skema asli + data acak, supaya pengembang aplikasi bisa mulai
sebelum hasil model siap. Jalankan: python make_dummy_db.py"""
import sqlite3, random, os
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output", "spareparts_dummy.db")
random.seed(7)
if os.path.exists(OUT):
    os.remove(OUT)
con = sqlite3.connect(OUT)
con.executescript(open(os.path.join(HERE, "schema.sql")).read())

N = 300
cats = ["BEARING", "VALVE", "GASKET", "SEAL", "COUPLING", "TRANSMITTER", "STUDBOLT", "O-RING"]
systems = ["CRACKED GAS COMPRESSION SYSTEM", "ETHYLENE REFRIGERATION SYSTEM", "QUENCH & DILUTION STEAM SYSTEM",
           "BUTADIENE EXTRACTION", "COOLING TOWER", "BOILER"]
patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Sporadic", "No demand"]
statuses = {"Stockout Risk": "Reorder", "Overstock": "Reduce", "Slow-moving": "Monitor", "Dead": "Review/Dispose",
            "Insurance Spare": "Keep", "Healthy": "Maintain"}
hist = [f"{y}-{m:02d}" for y, m in [(2025, m) for m in range(5, 13)] + [(2026, m) for m in range(1, 10)]]
fut = ["2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03"]
today = date(2026, 10, 1)

mats = []
for i in range(N):
    mno = 4100000000 + i * 37
    cat = random.choice(cats)
    price = random.choice([3e4, 2e5, 1.5e6, 8e6, 4e7])
    stock = random.choice([0, 0, 1, 2, 5, 10, 40])
    mn, mx = random.choice([(0, 0), (1, 2), (2, 5), (5, 12)])
    pat = random.choice(patterns)
    mats.append(mno)
    con.execute("INSERT INTO parts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        mno, f"{cat},DUMMY PART {i}", cat, "EA", random.choice("AAABBCS"), random.choice(["PX", "PY", "PX+PY"]),
        random.choice(systems), random.randint(1, 6), random.randint(1, 8), stock, mn, mx, price,
        random.choice(["actual", "estimated"]), stock * price, None, pat, round(random.uniform(1, 17), 2),
        round(random.uniform(0, 2), 2), random.randint(0, 17), random.randint(0, 200), "2026-08-14",
        random.randint(14, 90), random.randint(30, 160), round(random.random(), 2), "DUMMY SUPPLIER PT",
        random.randint(0, 12), random.randint(0, 1)))
    for m in hist:
        q = random.choice([0, 0, 0, 1, 2, 5])
        if q:
            con.execute("INSERT INTO demand_monthly VALUES (?,?,?,?)", (mno, m, q, int(random.random() < .05)))
    for m in fut:
        f = round(random.uniform(0, 3), 2)
        con.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)", (mno, m, f, max(0, f - 1), f + 2,
                                                                      round(random.random(), 2), int(random.random() < .15)))
    for mdl in ["MA", "SBA", "TSB", "GBM"]:
        con.execute("INSERT INTO model_eval VALUES (?,?,?,?,?)", (mno, mdl, round(random.uniform(.5, 1.5), 2),
                                                                 round(random.uniform(-.5, .5), 2), int(mdl == "SBA")))
    sl = random.choice([.9, .93, .95, .97, .99])
    rop = random.randint(0, 8)
    con.execute("INSERT INTO parameters VALUES (?,?,?,?,?,?,?)", (mno, sl, max(0, rop - 2), rop, rop + 4, 4, rop))
    st = random.choice(list(statuses))
    con.execute("INSERT INTO recommendations VALUES (?,?,?,?,?,?,?,?,?,?)", (
        mno, st, statuses[st], random.randint(0, 10), str(today + timedelta(days=random.randint(0, 90))),
        random.choice([-1, 1]) * price * random.randint(1, 5), round(random.random(), 2), round(random.random() * 100, 1),
        int(random.random() < .2), "Dummy reason: stock below new reorder point; lead time P80 95 days."))
    for s in [.8, .85, .9, .93, .95, .97, .99, .995]:
        con.execute("INSERT INTO scenario_grid VALUES (?,?,?,?,?,?)", (mno, s, rop * s, rop * s + 4,
                                                                     price * (rop * s + 2), round((1 - s) * 3, 3)))

for e in range(1, 61):
    n = random.randint(2, 8); short = random.randint(0, n)
    con.execute("INSERT INTO maintenance_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
        e, f"PX_DUMMY-{e}", f"DUMMY EQUIPMENT {e}", "PX", random.choice(systems), random.choice("ABCS"),
        "Shell & tube HE tube cleaning", random.choice([6, 12, 60]), str(today + timedelta(days=random.randint(0, 365))),
        n, short, round(1 - short / n, 2)))
    for mno in random.sample(mats, n):
        req = random.randint(1, 4); av = random.randint(0, 4)
        con.execute("INSERT INTO event_parts VALUES (?,?,?,?,?,?,?)", (e, mno, req, av, max(0, req - av),
                                                                     str(today + timedelta(days=random.randint(0, 200))),
                                                                     random.choice(["bom", "kit"])))
for m in hist:
    for pol, base in [("current", 9e11), ("proposed", 7e11)]:
        con.execute("INSERT INTO backtest_monthly VALUES (?,?,?,?,?)", (m, pol, base * random.uniform(.95, 1.05),
                                                                        random.randint(5, 30) if pol == "current" else random.randint(1, 10),
                                                                        random.randint(1, 10) if pol == "current" else random.randint(0, 3)))
for g in range(1, 11):
    for k, mno in enumerate(random.sample(mats, 3)):
        con.execute("INSERT OR IGNORE INTO twin_groups VALUES (?,?,?,?,?)", (g, mno, f"DUMMY TWIN {g}", random.randint(0, 5), int(k == 0)))
for _ in range(40):
    a, b = random.sample(mats, 2)
    con.execute("INSERT OR IGNORE INTO kit_rules VALUES (?,?,?,?,?)", (a, b, .01, round(random.random(), 2), round(random.uniform(1, 20), 1)))
for s in range(20):
    con.execute("INSERT INTO suppliers VALUES (?,?,?,?,?,?)", (100000 + s, f"DUMMY SUPPLIER {s} PT", random.randint(5, 300),
                                                                random.randint(20, 90), round(random.random(), 2), random.randint(30, 95)))
for i, (src, iss, n, act) in enumerate([("PR PO", "Exact duplicate rows", 5420, "Removed"),
                                        ("Movement Material", "Exact duplicate rows", 863, "Removed"),
                                        ("Mitem & Mplan", "Inconsistent cycle text", 28, "Standardized")], 1):
    con.execute("INSERT INTO data_quality_log VALUES (?,?,?,?,?,?)", (i, src, iss, n, act, "dummy"))
for src in ["BoM", "Movement Material", "Notif History", "FuncLoc List", "PR PO", "Mitem & Mplan"]:
    t = random.randint(5000, 20000); c = int(t * random.uniform(.5, .99))
    con.execute("INSERT INTO data_quality_score VALUES (?,?,?,?)", (src, t, c, round(100 * c / t, 1)))
for k, v, l in [("inventory_value", 9.1e11, "Total inventory value"), ("excess_value", 3.2e11, "Releasable excess"),
                ("dead_value", 1.4e11, "Dead stock value"), ("critical_stockout_risk", 95, "Critical parts at stockout risk"),
                ("reorder_this_week", 42, "Parts to order this week"), ("lt_median", 42, "Median lead time (days)"),
                ("late_rate", .6, "POs delivered late")]:
    con.execute("INSERT INTO kpis VALUES (?,?,?)", (k, v, l))
con.commit(); con.close()
print("written", OUT)

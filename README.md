# Sparely — AI & Data Pipeline

**Spare Part Intelligence & Inventory Optimization** · CALIBER 2026 Final Case

This repository contains the **AI and data pipeline** behind the platform. It turns six raw SAP-style Excel extracts into a single SQLite database (`output/spareparts.db`) holding cleaned data, forecasts, optimised stock parameters, recommendations, turnaround readiness and scenario data. The web app reads this database; it never recomputes the models.

**Main platform (web app):** https://github.com/narazla/Sparely

---

## Why this exists

Spare parts at a naphtha-cracker complex are planned reactively with static Min–Max levels. The result: critical parts are missing when equipment fails or a turnaround starts, while slow-moving parts tie up working capital. This pipeline answers the two key questions of the case:

1. **How can demand for irregular, uncertain spare-part consumption become more predictable?**
2. **How can inventory be optimised dynamically, balancing cost, availability and operational risk?**

## Key results (17 months of baseline data, May 2025 – Sep 2026)

| Area | Result |
| --- | --- |
| Data quality | 47% of PR/PO rows were exact duplicates; 597 material numbers were really 133 parts (duplicate material master) |
| False stockouts | 66 parts showed zero stock under one number while a twin number held stock |
| Turnaround effect | 37% of all consumption happened in the 3-month window around the Jan-2026 turnaround (TAM); 220 parts are used only during TAM |
| Forecast | 11% lower total-demand error than a 6-month moving average; over-forecast bias cut from +61% to +43% |
| Lead time | P80 lead-time error 62% lower than historical per-part quantiles |
| Need probability | ROC-AUC 0.93 (baseline 0.91); top 10% of risk scores capture 80% of actual demand |
| Notification NLP | 94% cross-validated accuracy (Failure / Repair-Replace / Routine-Support) |
| Replay vs current Min–Max | Stockout events 938 → 150 (−84%); criticality-A stockouts 795 → 88 (−89%), incl. turnaround readiness. Parameters alone: −59% stockouts |
| Turnaround readiness | Repeat of Jan-2026 TAM scope: 655 parts, 226 short today (65% ready) |

> All results come from the dummy baseline dataset provided by the organisers. See [Limitations](#limitations).

---

## Pipeline overview

```text
data/raw/*.xlsx  (BoM, Movement Material, Notif History, FuncLoc List, PR PO, Mitem & Mplan)
      │
      ▼
01_cleaning.py ──► 02_forecast.py ──► 04_leadtime_nlp.py ──► 05_global_model.py
                                                                    │
                    09_export.py ◄── 07_tam_readiness.py ◄── 06_optimize.py
                         │
                         ▼
                output/spareparts.db  ──►  Next.js web app (read-only)
```

| Stage | Script | What it does | Method |
| --- | --- | --- | --- |
| 1 | `01_cleaning.py` | Removes exact duplicates, standardises 28 maintenance-cycle spellings, validates dates, merges duplicate material masters, builds the material master (criticality, site, equipment class), estimates missing prices, flags abnormal consumption, writes a data-quality log | Entity resolution (TF-IDF + attribute check), kNN price model, Isolation Forest |
| 2 | `02_forecast.py` | Classifies demand patterns, separates turnaround demand from routine demand, backtests five models, selects the best model per pattern | ADI / CV² (Syntetos–Boylan), MA6, SES, Croston, SBA, TSB |
| 3 | `04_leadtime_nlp.py` | Predicts P50/P80 lead time per part and supplier reliability; classifies notification text | Quantile gradient boosting; TF-IDF + logistic regression (weak supervision) |
| 4 | `05_global_model.py` | One model across all parts predicting the probability that a part is needed next month | Gradient boosting classifier on a part × month panel |
| 5 | `06_optimize.py` | Simulates lead-time demand, picks the cost-optimal service level per part, sets new Min / Max / safety stock / reorder point, generates actions and plain-language reasons, builds the scenario grid, replays 17 months vs current Min–Max | Monte Carlo simulation, cost–risk optimisation |
| 6 | `07_tam_readiness.py` | Learns which parts are used together, schedules maintenance events for the next 12 months, scores readiness, builds a turnaround scenario | Association rules (support / confidence / lift) |
| 7 | `09_export.py` | Writes all tables and KPIs to SQLite following `schema.sql` | — |

---

## How each AI component works

### 1. Data cleaning & entity resolution
- **Exact duplicates removed:** PR PO 5,420 rows, Movement Material 863, BoM 230.
- **Duplicate material master:** candidates found with character n-gram TF-IDF similarity (≥ 0.90), then confirmed only if every comma-separated attribute matches (order-insensitive). 2,207 look-alike pairs with different attributes (e.g. `1/2IN` vs `12IN`, `P/N FE-1` vs `FE-2`) are deliberately kept separate.
- **Missing prices:** 2,575 stock items have no purchase history. Prices are estimated by k-nearest neighbours on description text (5-fold CV: 56% of estimates within 2× of the true price vs 50% for category median). Estimated prices are labelled `estimated`.
- **Abnormal consumption:** Isolation Forest + ≥ 5× part median flags 446 transactions; spikes are capped for training and shown as *Monitor* alerts.

### 2. Demand forecasting (turnaround-aware)
- Consumption is aggregated monthly per part and classified as Smooth, Erratic, Intermittent, Lumpy, Sporadic, TAM-only or No demand.
- Nov-2025 to Jan-2026 is treated as the **turnaround window** (37% of consumption, 2–3× normal maintenance orders). Routine forecasts are trained without it; turnaround needs are planned through readiness instead.
- **Backtest:** train May-25 → Mar-26, test Apr-26 → Sep-26. Metrics: scaled MAE, bias and total-demand error (MAPE is avoided because of zeros).

### 3. Lead time (quantile regression)
- Gradient boosting with quantile loss predicts P50 and P80 lead time from supplier, purchasing group, PR type, material category, criticality, quantity, price, requested lead time and month.
- **Backtest:** train on PRs before Apr-26, test after. P80 coverage 75% (target 80%).
- Supplier reliability score = 60% on-time rate + 40% lead-time stability.

### 4. Notification NLP
- Keyword rules label 93% of notifications; a TF-IDF + logistic regression model labels the rest. N2 notifications (all *Emergency*) are always *Failure*.
- Failure + Repair notifications on the functional locations where a part is installed become a feature for the global model.

### 5. Global need-probability model
- A single gradient boosting classifier trained on all parts (26k part-months) so that parts with short histories borrow strength from similar parts.
- Most important features: historical frequency, **number of functional locations where the part is installed**, equipment class, category, recent failure notifications.

### 6. Cost–risk optimisation
- Lead-time demand is simulated 3,000 times per part: lead time ~ lognormal (calibrated to P50/P80), demand events ~ Poisson, sizes ~ Gamma.
- For each service level (80% → 99.5%): reorder point = quantile of lead-time demand, Max = ROP + EOQ, expected inventory value and expected stockouts per year.
- The chosen service level minimises **carrying cost + expected stockouts × stockout cost by criticality**.
- Open purchase orders (delivery after 30 Sep 2026) count as stock on order.

### 7. Turnaround & maintenance readiness
- Maintenance items with cycles ≥ 1 month and a due date in the next 12 months are linked to wear parts installed on the same functional location, plus parts that association rules say are usually issued together.
- Readiness = share of required parts covered by stock on hand + on order; order-by date = due date − P80 lead time.

### Recommendation logic

| Status | Action | Rule |
| --- | --- | --- |
| Stockout Risk | Reorder | Stock + on order ≤ new reorder point, or reaches it within 30 days |
| Overstock / Slow-moving | Reduce | Stock above new Max (slow-moving if cover > 24 months) |
| TAM Reserve | Plan for TAM / Keep | No routine demand but used during the last turnaround |
| Insurance Spare | Keep / Reduce | No demand, criticality A; keep a small cover, release the rest |
| Dead | Review / Dispose | No demand, no turnaround use, not criticality A |
| Healthy | Maintain / Monitor | Within range; *Monitor* if abnormal spikes were detected |

---

## Prerequisites

- **Python 3.11+**
- ~200 MB free disk space
- Optional: **LightGBM**. The scripts use LightGBM automatically if it is installed; otherwise they fall back to scikit-learn `HistGradientBoosting` (same histogram-based gradient boosting family).

## Run locally

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r pipeline/requirements.txt
# optional: pip install lightgbm

# 3. Put the six baseline files in data/raw/
#    BoM.xlsx, Movement_Material.xlsx, Notif_History.xlsx,
#    FuncLoc_list.xlsx, PR_PO.xlsx, Mitem__Mplan.xlsx

# 4. Run the full pipeline (~1 minute)
bash pipeline/run_all.sh
```

The database is written to `output/spareparts.db`. Intermediate files go to `data/interim/` (git-ignored).

Run a single stage (each stage reads the outputs of the previous ones):

```bash
python pipeline/06_optimize.py
python pipeline/09_export.py
```

Create the dummy database (same schema, random data) for front-end development:

```bash
python pipeline/make_dummy_db.py      # -> output/spareparts_dummy.db
```

## Run on Google Colab

```python
# upload this repo (or clone it) and the six Excel files into data/raw/
!pip install lightgbm
!bash pipeline/run_all.sh
```

## Use the database in the web app

1. Copy `output/spareparts.db` into the web app, e.g. `app/data/spareparts.db`.
2. Read it server-side (API routes) with `better-sqlite3` in read-only mode.
3. Make sure the file is included in the Vercel deployment (`outputFileTracingIncludes` in `next.config`).
4. The database is regenerated, never migrated: change the schema in `pipeline/schema.sql`, rerun the pipeline, replace the file.

### Tables

| Table | Rows | Content |
| --- | --- | --- |
| `parts` | 3,305 | One row per stock item after merging duplicates: criticality, site, equipment class, stock, old Min/Max, price, pattern, lead time, failure signal |
| `demand_monthly` | 4,493 | Monthly consumption with anomaly flag |
| `forecasts` | 19,830 | 6-month forecast, 80% interval, need probability, PM due flag |
| `model_eval` | 3,320 | Backtest error per part and model; `selected = 1` for the model in use |
| `parameters` | 3,305 | New service level, safety stock, ROP (= Min), Max, order quantity |
| `recommendations` | 3,305 | Status, action, quantity, order-by date, value impact, stockout risk, priority, reason |
| `maintenance_events` | 1,493 | Maintenance events in the next 12 months + turnaround scenario, readiness |
| `event_parts` | 6,258 | Parts required per event, shortfall, order-by date, source (`bom` / `kit` / `tam`) |
| `backtest_monthly` | 34 | 17-month replay: `current` vs `proposed` policy |
| `scenario_grid` | 26,440 | Per part × service level: ROP, Max, average inventory value, expected stockouts (for the scenario simulator) |
| `twin_groups` | 597 | Duplicate material numbers merged into one part |
| `kit_rules` | 187 | Parts usually issued together (support, confidence, lift) |
| `suppliers` | 124 | Median lead time, late rate, reliability score |
| `data_quality_log` | 18 | Every cleaning action: source, issue, rows affected, action, reason |
| `data_quality_score` | 6 | Clean-row share per source file |
| `kpis` | 49 | Headline numbers for the dashboard and deck |

---

## Assumptions (all editable)

| Assumption | Value | Where |
| --- | --- | --- |
| Criticality order | A > B > C > S (confirmed by organisers) | `01_cleaning.py` |
| Carrying cost | 20% of item value per year | `06_optimize.py` |
| Ordering cost | Rp 1.5 M per purchase order | `06_optimize.py` |
| Stockout cost per event | A Rp 750 M · B Rp 200 M · C Rp 50 M · S Rp 10 M | `06_optimize.py` |
| FX rates | USD 16,300 · EUR 17,700 · SGD 12,600 · JPY 105 | `01_cleaning.py` |
| Turnaround window | Nov-2025 to Jan-2026 | `02_forecast.py` |
| Snapshot date | 1 Oct 2026 (data ends 30 Sep 2026) | all stages |
| Turnaround scenario date | 15 Jan 2027 (planning placeholder) | `07_tam_readiness.py` |

## Limitations

- The baseline dataset is **dummy data**; results show the method, not real plant performance.
- 2,575 of 3,305 stock items have **estimated prices** (56% within 2× in validation). Rupiah totals that include estimated prices should be read as indicative; actual-price figures are reported separately in `kpis`.
- Only 17 months of history: yearly seasonality cannot be learned.
- Consumption records only fulfilled issues, so stockouts in the replay are **simulated**.
- The replay with turnaround readiness assumes the turnaround scope is known in advance; it is an upper bound. The parameters-only result (−59% stockouts) is the conservative figure.
- Maintenance plans are not linked to spare parts directly; the link goes through functional location → BoM, and only 56% of planned functional locations have BoM data. PM due dates did not add predictive power in the global model.
- The notification → order → material link appears random in the dummy data, so the NLP classes could not be validated against parts issued.
- Stockout costs are assumptions, not company figures.

## Repository structure

```text
.
├── pipeline/
│   ├── 01_cleaning.py
│   ├── 02_forecast.py
│   ├── 04_leadtime_nlp.py
│   ├── 05_global_model.py
│   ├── 06_optimize.py
│   ├── 07_tam_readiness.py
│   ├── 09_export.py
│   ├── make_dummy_db.py
│   ├── run_all.sh
│   ├── schema.sql
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── raw/          # six baseline Excel files (not public)
│   └── interim/      # generated, git-ignored
├── output/
│   ├── spareparts.db
│   └── spareparts_dummy.db
├── .gitignore
└── README.md
```

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| Data | pandas, NumPy, openpyxl |
| Machine learning | scikit-learn (TF-IDF, logistic regression, Isolation Forest, kNN, HistGradientBoosting), optional LightGBM |
| Simulation & statistics | NumPy, SciPy |
| Storage | SQLite |

---

Built for the **CALIBER 2026 Final Case — Spare Part Intelligence & Inventory Optimization**.

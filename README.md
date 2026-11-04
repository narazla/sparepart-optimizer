# Sparely — AI & Data Pipeline

*Just enough, just in time.*

**Spare Part Intelligence & Inventory Optimization** · CALIBER 2026 Final Case

This repository contains the **AI and data pipeline** behind Sparely. It turns six raw SAP-style Excel extracts into a single SQLite database (`output/spareparts.db`) holding cleaned data, forecasts, optimised stock parameters, recommendations, turnaround readiness and scenario data.

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
| Turnaround effect | 37% of all consumption happened in the 3-month window around the Jan-2026 turnaround (TAM); 220+ parts are used only during TAM |
| Non-Stock misclassification | 119 Non Stock parts (criticality A/B, used repeatedly, long lead time) recommended to become Stock Items; 1,636 Non Stock materials hold unplanned stock |
| Forecast | 12% lower total-demand error than a 6-month moving average; over-forecast bias cut from +63% to +44% |
| Lead time | P80 lead-time error 62% lower than historical per-part quantiles; 60% of POs arrive after the requested date |
| Need probability | ROC-AUC 0.92 (baseline 0.91); top 10% of risk scores capture 76% of actual demand |
| Notification NLP | 94% cross-validated accuracy (Failure / Repair-Replace / Routine-Support) |
| Holdout replay vs current Min–Max (Apr–Sep 2026, parameters built from data up to Mar 2026) | Stockouts 301 → 175 (−42%); criticality-A stockouts 270 → 100 (−63%); fill rate 79% → 88% |
| Turnaround readiness | Repeat of Jan-2026 TAM scope: 750 parts, 276 short today (63% ready) |
| Sourcing | No material number was ever bought from more than one supplier; urgent buys get a faster same-category candidate supplier to be qualified by Procurement |

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
| 1 | `01_cleaning.py` | Removes exact duplicates, standardises 28 maintenance-cycle spellings, validates dates, merges duplicate material masters, builds the material master (criticality, site, equipment class, purchasing group), estimates missing prices, flags abnormal consumption and misclassified Non Stock parts, writes a data-quality log | Entity resolution (TF-IDF + attribute check), kNN price model, Isolation Forest |
| 2 | `02_forecast.py` | Classifies demand patterns, separates turnaround demand from routine demand, backtests five models, selects the best model per pattern | ADI / CV² (Syntetos–Boylan), MA6, SES, Croston, SBA, TSB |
| 3 | `04_leadtime_nlp.py` | Predicts P50/P80 lead time per part, scores supplier reliability, selects normal and urgent suppliers; classifies notification text | Quantile gradient boosting; TF-IDF + logistic regression (weak supervision) |
| 4 | `05_global_model.py` | One model across all parts predicting the probability that a part is needed next month; saves holdout predictions for the replay | Gradient boosting classifier on a part × month panel |
| 5 | `06_optimize.py` | Simulates lead-time demand, picks the cost-optimal service level per part, sets new Min / Max / safety stock / reorder point, generates actions (incl. Convert to Stock) with sourcing mode and plain-language reasons, builds the scenario grid, runs a leak-free holdout replay vs current Min–Max | Monte Carlo simulation, cost–risk optimisation |
| 6 | `07_tam_readiness.py` | Learns which parts are used together, applies a replacement factor to BoM quantities, schedules maintenance events for the next 12 months, scores readiness, builds a turnaround scenario | Association rules (support / confidence / lift) |
| 7 | `09_export.py` | Writes all tables and KPIs to SQLite following `schema.sql` | — |

---

## How each AI component works

### 1. Data cleaning & entity resolution
- **Exact duplicates removed:** PR PO 5,420 rows, Movement Material 863, BoM 230.
- **Duplicate material master:** candidates found with character n-gram TF-IDF similarity (≥ 0.90), then confirmed only if every comma-separated attribute matches (order-insensitive). 2,207 look-alike pairs with different attributes (e.g. `1/2IN` vs `12IN`, `P/N FE-1` vs `FE-2`) are deliberately kept separate.
- **Missing prices:** 2,575 stock items have no purchase history. Prices are estimated by k-nearest neighbours on description text (5-fold CV: 56% of estimates within 2× of the true price vs 50% for category median). Estimated prices are labelled `estimated`.
- **Abnormal consumption:** Isolation Forest + ≥ 5× part median flags 446 transactions; spikes are capped for training and shown as *Monitor* alerts.

### 2. Non-Stock to Stock conversion
- A Non Stock material is flagged when it is criticality A or B, used in at least 2 of 17 months, and has a historical lead time above 42 days (median of all POs) or no regular PO at all.
- 119 parts qualify. They are optimised like stock items and receive a recommendation to change the SAP MRP type to Stock Item with new Min/Max; 80 also need initial stock.

### 3. Demand forecasting (turnaround-aware)
- Consumption is aggregated monthly per part and classified as Smooth, Erratic, Intermittent, Lumpy, Sporadic, TAM-only or No demand.
- Nov-2025 to Jan-2026 is treated as the **turnaround window** (37% of consumption, 2–3× normal maintenance orders). Routine forecasts are trained without it; turnaround needs are planned through readiness instead.
- **Backtest:** train May-25 → Mar-26, test Apr-26 → Sep-26 on 751 parts. Metrics: scaled MAE, bias and total-demand error (MAPE is avoided because of zeros). SES wins for all routine patterns; TSB is used for TAM-only and No demand.

### 4. Lead time (quantile regression)
- Gradient boosting with quantile loss predicts P50 and P80 lead time from supplier, purchasing group, PR type, material category, criticality, quantity, price, requested lead time and month.
- **Backtest:** train on PRs before Apr-26, test after. P80 coverage 75% (target 80%).
- Supplier reliability score = 60% on-time rate + 40% lead-time stability.

### 5. Dual-mode sourcing
- **Normal:** cheapest known supplier for the part (median unit price).
- **Urgent:** supplier with the shortest effective lead time = median lead time × (1 + late rate). For single-supplier parts, a supplier from the same category (≥ 5 POs) is suggested when it is at least 20% faster.
- Urgent mode is applied to criticality-A buys with ≥ 20% chance of running out or stock below safety stock.
- Per original material number, no part has more than one supplier in the data. Different suppliers only appear for 36 merged duplicate parts bought under different numbers, so urgent suppliers are candidates that Procurement must qualify.

### 6. Notification NLP
- Keyword rules label 93% of notifications; a TF-IDF + logistic regression model labels the rest. N2 notifications (all *Emergency*) are always *Failure*.
- Failure + Repair notifications on the functional locations where a part is installed become a feature for the global model.

### 7. Global need-probability model
- A single gradient boosting classifier trained on all parts (27k part-months) so that parts with short histories borrow strength from similar parts.
- Most important features: historical frequency, **number of functional locations where the part is installed**, equipment class, category, recent failure notifications.

### 8. Cost–risk optimisation
- Lead-time demand is simulated 3,000 times per part: lead time ~ lognormal (calibrated to P50/P80), demand events ~ Poisson, sizes ~ Gamma.
- For each service level (80% → 99.5%): reorder point = quantile of lead-time demand, Max = ROP + EOQ, expected inventory value and expected stockouts per year.
- The chosen service level minimises **carrying cost + expected stockouts × stockout cost by criticality**.
- Open purchase orders (delivery after 30 Sep 2026) count as stock on order.

### 9. Holdout replay (proof)
- Parameters for both policies are built **only from data up to March 2026**: need probability from the global model trained up to March 2026, usage sizes from routine months before April 2026, and lead times from POs created before April 2026.
- Both policies are then tested on actual consumption from **April to September 2026**, six months the model never saw. Each month stock is reduced by demand, an order is placed when the position reaches Min, and goods arrive after the lead time. Both policies start at their own Max on 1 April 2026.

| Metric (Apr–Sep 2026) | Current Min–Max | Sparely | Change |
| --- | --- | --- | --- |
| Stockouts (part-months with unmet demand) | 301 | 175 | −42% |
| Criticality-A stockouts | 270 | 100 | −63% |
| Fill rate (part-months fully served) | 79.4% | 88.0% | +8.6 pts |
| Criticality-A stockouts, stock items only | 172 | 92 | −47% |

### 10. Turnaround & maintenance readiness
- Maintenance items with cycles ≥ 1 month and a due date in the next 12 months are linked to parts installed on the same functional location, plus parts that association rules say are usually issued together.
- **Replacement factor (alpha):** consumables and sealing parts (gasket, O-ring, filter element, packing, grease, oil) = 100%; other parts = historical issued quantity ÷ installed quantity (capped at 1, category median when no history; median 0.33). Requirement = installed quantity × alpha; lines below 0.5 units are ignored.
- Readiness = share of required parts covered by stock on hand + on order; order-by date = due date − P80 lead time.

### Recommendation logic

| Status | Action | Rule |
| --- | --- | --- |
| Misclassified Non-Stock | Convert to Stock | Non Stock, criticality A/B, used in 2+ months, lead time > 42 days or no PO |
| Stockout Risk | Reorder | Stock + on order ≤ new reorder point, or reaches it within 30 days |
| Overstock / Slow-moving | Reduce | Stock above new Max (slow-moving if cover > 24 months) |
| TAM Reserve | Plan for TAM / Keep | No routine demand but used during the last turnaround |
| Insurance Spare | Keep / Reduce | No demand, criticality A; keep a small cover, release the rest |
| Dead | Review / Dispose | No demand, no turnaround use, not criticality A |
| Healthy | Maintain / Monitor | Within range; *Monitor* if abnormal spikes were detected |

Every buy recommendation carries a sourcing mode (Normal / Urgent) and a recommended supplier. Every part carries a purchasing group (department owner) from PO history, or inferred from its category.

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
| `parts` | 3,424 | One row per part in scope (3,305 stock items + 119 Non Stock conversion candidates): criticality, site, equipment class, MRP type, purchasing group, stock, old Min/Max, price, pattern, lead time, normal and urgent supplier |
| `demand_monthly` | 4,905 | Monthly consumption with anomaly flag |
| `forecasts` | 20,544 | 6-month forecast, 80% interval, need probability, PM due flag |
| `model_eval` | 3,755 | Backtest error per part and model; `selected = 1` for the model in use |
| `parameters` | 3,424 | New service level, safety stock, ROP (= Min), Max, order quantity |
| `recommendations` | 3,424 | Status, action, quantity, order-by date, value impact, stockout risk, priority, sourcing mode, supplier, reason |
| `maintenance_events` | 2,294 | Maintenance events in the next 12 months + turnaround scenario, readiness |
| `event_parts` | 9,488 | Parts required per event, replacement factor, shortfall, order-by date, source (`bom` / `kit` / `tam`) |
| `backtest_monthly` | 12 | Holdout replay Apr–Sep 2026: `current` vs `proposed` policy |
| `scenario_grid` | 27,392 | Per part × service level: ROP, Max, average inventory value, expected stockouts (for the scenario simulator) |
| `twin_groups` | 597 | Duplicate material numbers merged into one part |
| `kit_rules` | 187 | Parts usually issued together (support, confidence, lift) |
| `suppliers` | 124 | Median lead time, late rate, reliability score |
| `data_quality_log` | 20 | Every cleaning action: source, issue, rows affected, action, reason |
| `data_quality_score` | 6 | Clean-row share per source file |
| `kpis` | ~60 | Headline numbers for the dashboard and deck |

---

## Assumptions (all editable)

| Assumption | Value | Where |
| --- | --- | --- |
| Criticality order | A > B > C > S (confirmed by organisers) | `01_cleaning.py` |
| Purchasing group meaning | Department that owns the item (confirmed by organisers) | `01_cleaning.py` |
| Non Stock conversion rule | Criticality A/B, used in ≥ 2 months, lead time > 42 days or no PO | `01_cleaning.py` |
| Carrying cost | 20% of item value per year | `06_optimize.py` |
| Ordering cost | Rp 1.5 M per purchase order | `06_optimize.py` |
| Stockout cost per event | A Rp 750 M · B Rp 200 M · C Rp 50 M · S Rp 10 M | `06_optimize.py` |
| Urgent sourcing trigger | Criticality A and (≥ 20% stockout risk or stock below safety stock) | `06_optimize.py` |
| Replay split | Train up to Mar 2026, test Apr–Sep 2026 | `05_global_model.py`, `06_optimize.py` |
| Replacement factor | Consumables 100%; others historical issued ÷ installed | `07_tam_readiness.py` |
| FX rates | USD 16,300 · EUR 17,700 · SGD 12,600 · JPY 105 | `01_cleaning.py` |
| Turnaround window | Nov-2025 to Jan-2026 | `02_forecast.py` |
| Snapshot date | 1 Oct 2026 (data ends 30 Sep 2026) | all stages |
| Turnaround scenario date | 15 Jan 2027 (planning placeholder) | `07_tam_readiness.py` |

## Limitations

- The baseline dataset is **dummy data**; results show the method, not real plant performance.
- 2,575 of 3,305 stock items have **estimated prices** (56% within 2× in validation). Rupiah totals that include estimated prices should be read as indicative; actual-price figures are reported separately in `kpis`.
- Only 17 months of history: yearly seasonality cannot be learned.
- Consumption records only fulfilled issues, so stockouts in the replay are **simulated**. Both policies start at their own Max on 1 Apr 2026, so existing excess stock is not counted.
- The replay tests routine demand only: the turnaround window (Nov-2025 to Jan-2026) falls in the training period. An earlier 17-month in-sample replay overstated results and was replaced by this holdout test.
- Maintenance plans are not linked to spare parts directly; the link goes through functional location → BoM, and only 56% of planned functional locations have BoM data. PM due dates did not add predictive power in the global model.
- No material number was ever bought from more than one supplier; urgent-sourcing suggestions are same-category candidates that Procurement still needs to qualify.
- Only 730 parts have purchasing-group data from POs; the rest are inferred from their category.
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
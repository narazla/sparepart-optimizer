# Spare Part Intelligence — AI & Data Pipeline

Builds `output/spareparts.db` (SQLite) from the six CALIBER 2026 baseline files.
The Next.js app only **reads** this database.

## Run

```bash
pip install -r requirements.txt      # lightgbm is optional; scikit-learn HistGradientBoosting is used if absent
# put BoM.xlsx, Movement_Material.xlsx, Notif_History.xlsx, FuncLoc_list.xlsx, PR_PO.xlsx, Mitem__Mplan.xlsx in data/raw/
bash run_all.sh                      # ~1 minute
```

## Stages

| Script | What it does | AI / method |
|---|---|---|
| `01_cleaning.py` | Removes exact duplicates (PR PO 47%), standardises 28 Mplan cycle spellings, validates dates, merges duplicate material masters, builds the material master, estimates missing prices, flags abnormal consumption | Entity resolution (TF-IDF + attribute check), kNN price model, Isolation Forest |
| `02_forecast.py` | Demand pattern classification (ADI / CV²), separates turnaround (TAM) demand from routine demand, backtests MA6 / SES / Croston / SBA / TSB, selects model per pattern | Statistical forecasting, time-based backtest |
| `04_leadtime_nlp.py` | Predicts P50/P80 lead time per part, supplier reliability; classifies notification text into Failure / Repair / Routine | Quantile gradient boosting; TF-IDF + logistic regression (weak supervision) |
| `05_global_model.py` | One model across all parts predicting the probability a part is needed next month | Gradient boosting classifier (panel), permutation importance |
| `06_optimize.py` | Monte Carlo lead-time demand, cost-optimal service level, new Min/Max/SS/ROP, actions & reasons, scenario grid, 17-month replay vs current Min-Max | Prescriptive optimisation |
| `07_tam_readiness.py` | Parts co-used in maintenance orders, 12-month maintenance events, readiness score, TAM scenario | Association rules |
| `09_export.py` | Writes all tables + KPIs to SQLite (`schema.sql`) | — |

## Key assumptions (editable)
- Criticality order A > B > C > S (confirmed by organisers).
- Carrying cost 20%/year; ordering cost Rp 1.5 M per PO; stockout cost per event A Rp 750 M, B Rp 200 M, C Rp 50 M, S Rp 10 M.
- Fixed FX: USD 16,300; EUR 17,700; SGD 12,600; JPY 105.
- TAM window = Nov-2025 to Jan-2026 (37% of consumption, 2-3x maintenance orders), aligned with the Jan-2026 turnaround.
- Data snapshot date 1 Oct 2026; POs due after 30 Sep 2026 are treated as on order.

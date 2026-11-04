-- Spare Part Intelligence database schema (SQLite)
-- Dibuat ulang setiap kali pipeline Python dijalankan. Aplikasi Next.js hanya MEMBACA.
-- Konvensi: tanggal = TEXT 'YYYY-MM-DD', bulan = TEXT 'YYYY-MM', uang = REAL dalam IDR.

-- Satu baris per stock item (material yang sudah digabung dengan kembarannya)
CREATE TABLE parts (
  material_no        INTEGER PRIMARY KEY,
  description        TEXT NOT NULL,
  category           TEXT,              -- kata pertama deskripsi: BEARING, VALVE, ...
  unit               TEXT,
  criticality        TEXT,              -- A (tertinggi), B, C, S (terendah)
  site               TEXT,              -- PX / PY / PX+PY
  main_system        TEXT,              -- sistem proses utama, mis. 'CRACKED GAS COMPRESSION SYSTEM'
  n_funcloc          INTEGER,           -- jumlah FuncLoc tempat part terpasang
  installed_qty      REAL,
  stock              REAL,
  min_old            REAL,
  max_old            REAL,
  price              REAL,              -- IDR per unit
  price_source       TEXT,              -- 'actual' | 'estimated'
  stock_value        REAL,              -- stock * price
  twin_group         INTEGER,           -- NULL jika tidak punya kembaran
  pattern            TEXT,              -- Smooth | Erratic | Intermittent | Lumpy | Sporadic | No demand
  adi                REAL,
  cv2                REAL,
  demand_months      INTEGER,           -- jumlah bulan dengan pemakaian (dari 17)
  total_issued       REAL,
  last_issue_date    TEXT,
  lt_p50             REAL,              -- lead time hari
  lt_p80             REAL,
  late_rate          REAL,              -- 0..1
  main_supplier      TEXT,
  notif_12m          INTEGER,           -- notifikasi kerusakan di FuncLoc terkait, 12 bulan terakhir
  has_anomaly        INTEGER            -- 0/1
);

-- Konsumsi bulanan (histori)
CREATE TABLE demand_monthly (
  material_no  INTEGER,
  month        TEXT,
  qty          REAL,
  is_anomaly   INTEGER,
  PRIMARY KEY (material_no, month)
);

-- Forecast 6 bulan ke depan
CREATE TABLE forecasts (
  material_no  INTEGER,
  month        TEXT,
  forecast     REAL,
  lower        REAL,
  upper        REAL,
  need_prob    REAL,                    -- peluang ada pemakaian di bulan itu
  pm_due       INTEGER,                 -- 1 jika ada PM terjadwal di FuncLoc terkait
  PRIMARY KEY (material_no, month)
);

-- Hasil backtest per part per model
CREATE TABLE model_eval (
  material_no  INTEGER,
  model        TEXT,                    -- MA | ETS | Croston | SBA | TSB | GBM
  mase         REAL,
  bias         REAL,
  selected     INTEGER,                 -- 1 = model yang dipakai
  PRIMARY KEY (material_no, model)
);

-- Parameter stok baru
CREATE TABLE parameters (
  material_no    INTEGER PRIMARY KEY,
  service_level  REAL,
  safety_stock   REAL,
  rop            REAL,                  -- = Min baru
  max_new        REAL,
  order_qty      REAL,
  min_new        REAL
);

-- Rekomendasi aksi
CREATE TABLE recommendations (
  material_no     INTEGER PRIMARY KEY,
  status          TEXT,                 -- Stockout Risk | Overstock | Slow-moving | Dead | Insurance Spare | Healthy
  action          TEXT,                 -- Reorder | Reduce | Review/Dispose | Keep | Monitor | Maintain | Use twin stock
  qty             REAL,                 -- qty pesan (Reorder) atau qty dilepas (Reduce)
  order_by_date   TEXT,
  value_impact    REAL,                 -- + modal dilepas, - investasi tambahan
  stockout_risk   REAL,                 -- 0..1
  priority        REAL,                 -- untuk urutan default
  tam_linked      INTEGER,
  reason          TEXT                  -- kalimat "Why this recommendation"
);

-- Event maintenance 12 bulan ke depan
CREATE TABLE maintenance_events (
  event_id      INTEGER PRIMARY KEY,
  funcloc       TEXT,
  funcloc_desc  TEXT,
  site          TEXT,
  system        TEXT,
  criticality   TEXT,
  task          TEXT,
  cycle_months  REAL,
  due_date      TEXT,
  n_parts       INTEGER,
  n_short       INTEGER,
  readiness     REAL                    -- 0..1
);

CREATE TABLE event_parts (
  event_id       INTEGER,
  material_no    INTEGER,
  required_qty   REAL,
  available_qty  REAL,
  shortfall      REAL,
  order_by_date  TEXT,
  source         TEXT,                  -- 'bom' | 'kit'
  PRIMARY KEY (event_id, material_no)
);

-- Replay historis: kebijakan lama vs usulan (proposed)
CREATE TABLE backtest_monthly (
  month                TEXT,
  policy               TEXT,            -- 'current' | 'proposed'
  inventory_value      REAL,
  stockouts            INTEGER,
  stockouts_critical   INTEGER,
  PRIMARY KEY (month, policy)
);

-- Grid untuk Scenario Simulator: satu baris per part per service level
CREATE TABLE scenario_grid (
  material_no          INTEGER,
  service_level        REAL,            -- 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995
  rop                  REAL,
  max_level            REAL,
  avg_inventory_value  REAL,            -- IDR
  expected_stockouts   REAL,            -- kejadian per tahun
  PRIMARY KEY (material_no, service_level)
);

-- Master material ganda
CREATE TABLE twin_groups (
  twin_group    INTEGER,
  material_no   INTEGER,
  description   TEXT,
  stock         REAL,
  is_primary    INTEGER,                -- nomor yang dipertahankan
  PRIMARY KEY (twin_group, material_no)
);

-- Kit part (association rules)
CREATE TABLE kit_rules (
  material_a   INTEGER,
  material_b   INTEGER,
  support      REAL,
  confidence   REAL,
  lift         REAL,
  PRIMARY KEY (material_a, material_b)
);

CREATE TABLE suppliers (
  supplier_code  INTEGER PRIMARY KEY,
  name           TEXT,
  n_po           INTEGER,
  lt_median      REAL,
  late_rate      REAL,
  reliability    REAL                   -- 0..100
);

CREATE TABLE data_quality_log (
  id             INTEGER PRIMARY KEY,
  source         TEXT,
  issue          TEXT,
  rows_affected  INTEGER,
  action         TEXT,                  -- Removed | Standardized | Flagged | Merged | Kept
  reason         TEXT
);

CREATE TABLE data_quality_score (
  source      TEXT PRIMARY KEY,
  total_rows  INTEGER,
  clean_rows  INTEGER,
  score       REAL                      -- 0..100
);

-- KPI ringkas untuk Command Center
CREATE TABLE kpis (
  key    TEXT PRIMARY KEY,
  value  REAL,
  label  TEXT
);

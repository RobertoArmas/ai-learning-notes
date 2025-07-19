# transform_dataset.py
# -----------------------------------------------------------
# Convierte demand_global_superstore_real.csv al formato final requerido
# y simula interest_rate_pct, competitor_price_index y regulation_tariff_flag
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 1. Leer CSV original ------------------------------------------
IN_PATH  = Path("output/demand_global_superstore_real.csv")
OUT_PATH = Path("output/demand_global_superstore_transformed.csv")

df = pd.read_csv(IN_PATH, parse_dates=["date"])

# ---------- 2. Renombrar columnas existentes ------------------------------
df = df.rename(columns={
    "product_cat":   "product_code",
    "usd_fx":        "exchange_rate_to_usd"
})

# ---------- 3. Simular columnas faltantes ---------------------------------
rng = np.random.default_rng(42)          # semilla para reproducibilidad

# 3.1 interest_rate_pct  ≈ inflación + 1 p.p. ± ruido(σ=0.6)
df["interest_rate_pct"] = (
    df["inflation_rate_pct"] + 1 + rng.normal(0, 0.6, len(df))
).clip(lower=0).round(2)

# 3.2 competitor_price_index  (baseline 1 ±5 %, ajustado por macro-categoría)
cat_factor = {
    "Technology": 1.05,
    "Furniture":  0.95,
    "Office":     1.00,
}
def infer_category(code: str) -> str:
    return code.split("-")[0].title() if isinstance(code, str) else "Office"

df["competitor_price_index"] = (
    df["product_code"].map(infer_category).map(lambda c: cat_factor.get(c, 1.0))
    * rng.normal(1.0, 0.05, len(df))
).round(3)

# 3.3 regulation_tariff_flag  (5 % de probabilidad)
df["regulation_tariff_flag"] = rng.binomial(1, 0.05, len(df))

# ---------- 4. Seleccionar y reordenar ------------------------------------
cols_out = [
    "date", "country", "product_code", "units_sold",
    "gdp_growth_pct", "inflation_rate_pct", "interest_rate_pct",
    "exchange_rate_to_usd", "competitor_price_index",
    "regulation_tariff_flag", "holiday_event_flag"
]
df = df[cols_out]

# ---------- 5. Guardar CSV final ------------------------------------------
OUT_PATH.parent.mkdir(exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("✅ CSV transformado:", OUT_PATH.resolve())
print(df.head())

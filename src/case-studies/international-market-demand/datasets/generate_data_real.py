import pandas as pd, numpy as np, wbdata, pycountry, chardet, holidays
from pathlib import Path

# ---------- 1. Cargar Global Superstore --------------------
csv_file = "Global_Superstore2.csv"
enc = chardet.detect(open(csv_file, "rb").read(100_000))["encoding"]

# 🆕  Usa date_format="%d-%m-%Y"  (formato del archivo ejemplo)
sales = (pd.read_csv(
            csv_file,
            encoding=enc,
            parse_dates=["Order Date"],
            date_format="%d-%m-%Y",      # ← parsea DD-MM-YYYY sin deprecaciones
            on_bad_lines="warn")
         .rename(columns={"Country": "country",
                          "Order Date": "date",
                          "Quantity": "units_sold",
                          "Category": "product_cat"})
         [["date", "country", "product_cat", "units_sold", "Sales"]])

# La columna ya es datetime64[ns]  ➜  .dt funciona
sales["date"] = sales["date"].dt.to_period("M").dt.to_timestamp()
monthly = (sales.groupby(["date", "country", "product_cat"])
                 .agg(units_sold=("units_sold", "sum"),
                      revenue_usd=("Sales", "sum"))
                 .reset_index())

# ---------- 2. ISO-3 limpio --------------------------------
countries_iso3 = {c.name: c.alpha_3 for c in pycountry.countries}
monthly["iso3"] = monthly["country"].map(countries_iso3)\
                                    .replace({"XKX":"SRB","REU":"FRA","ANT":"NLD","ALA":"FIN"})

wb_valid = {c["id"] for c in wbdata.get_countries()}
good_iso3 = [c for c in monthly["iso3"].dropna().unique() if c in wb_valid]

# ---------- 3. PIB e inflación -----------------------------
indicators = {"NY.GDP.MKTP.KD.ZG":"gdp_growth_pct","FP.CPI.TOTL.ZG":"inflation_rate_pct"}
def wbdata_batch(ind, iso_list, batch=50):
    return pd.concat([wbdata.get_dataframe(ind, country=iso_list[i:i+batch])
                      for i in range(0, len(iso_list), batch)])
macro_raw = (wbdata_batch(indicators, good_iso3).reset_index()
             .rename(columns={"date":"year"}))
macro_raw["year"] = pd.to_datetime(macro_raw["year"].astype(str)+"-01-01")
macro = (macro_raw.merge(pd.DataFrame({"month":range(1,13)}), how="cross"))
macro["date"] = pd.to_datetime(dict(year=macro["year"].dt.year,
                                    month=macro["month"], day=1))
macro = macro.drop(columns="month")

# ---------- 4. FX SIMULADO (robusto) -----------------------
def synthetic_fx(cur):                    # valores arbitrarios/fijos
    base={"USD":1,"EUR":0.9,"GBP":0.8,"CAD":1.3,"AUD":1.5}
    if cur in base: return base[cur]
    rng=np.random.default_rng(abs(hash(cur))%2**32)
    return round(rng.normal(1.2,0.25),2)

fx_rows=[]
for country,iso3 in monthly[["country","iso3"]].drop_duplicates().values:
    try:
        ctry=pycountry.countries.get(alpha_3=iso3)
        curr_obj=pycountry.currencies.get(numeric=ctry.numeric)
        cur=curr_obj.alpha_3 if curr_obj else "USD"
    except (LookupError,AttributeError):
        cur="USD"
    rate=synthetic_fx(cur)
    for d in monthly["date"].unique():
        fx_rows.append({"country":country,"date":d,"usd_fx":rate})
fx=pd.DataFrame(fx_rows)

# ---------- 5. Feriados (robusto y sin FutureWarning) ------
def holiday_flag(df):
    rows = []
    for country, iso3 in df[["country", "iso3"]].drop_duplicates().values:

        # 1️⃣  Obtén el ISO-2 (alpha_2) requerido por `holidays`
        try:
            iso2 = pycountry.countries.get(alpha_3=iso3).alpha_2
        except (LookupError, AttributeError):
            continue                         # país raro → omitimos

        # 2️⃣  Crea calendario solo si el país existe en `holidays`
        try:
            hol = holidays.country_holidays(iso2, years=range(2011, 2016))
        except (NotImplementedError, AttributeError):
            continue                         # no soportado → omitimos

        # 3️⃣  Convierte el dict a DatetimeIndex (evita FutureWarning)
        hol_dates = pd.to_datetime(list(hol.keys()))
        tmp = pd.DataFrame({"date": pd.date_range("2011-01-01", "2015-12-31", freq="MS")})
        tmp["holiday_event_flag"] = tmp["date"].isin(hol_dates.normalize()).astype(int)
        tmp["country"] = country
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)

holiday = holiday_flag(monthly)

# ---------- 6. Unión y guardado ----------------------------
dataset = (
    monthly
    .merge(macro,   on=["country", "date"], how="left")   # <── aquí
    .merge(fx,      on=["country", "date"], how="left")
    .merge(holiday, on=["country", "date"], how="left")
    .sort_values(["date", "country", "product_cat"])
    .reset_index(drop=True)
    .fillna(method="ffill")
)

out = Path("output/demand_global_superstore_real.csv")
out.parent.mkdir(exist_ok=True)
dataset.to_csv(out, index=False)

print("✅ CSV generado:", out)
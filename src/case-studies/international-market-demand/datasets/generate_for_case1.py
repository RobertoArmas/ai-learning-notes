import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define date range, countries, and products
dates = pd.date_range(start="2020-01-01", end="2024-12-01", freq="MS")  # Monthly start frequency
countries = ["USA", "MEX", "DEU"]  # United States, Mexico, Germany
products = ["A", "B", "C"]  # Three example product lines

# Helper dictionaries for country/product-specific baselines and macro assumptions
baseline_sales = {"USA": 500, "MEX": 300, "DEU": 400}
product_adjustment = {"A": 1.0, "B": 0.8, "C": 1.2}

gdp_growth_mean = {"USA": 2.0, "MEX": 2.5, "DEU": 1.8}
inflation_mean = {"USA": 2.5, "MEX": 4.0, "DEU": 1.7}
interest_rate_mean = {"USA": 3.0, "MEX": 7.0, "DEU": 2.0}
exchange_rate_mean = {"USA": 1.0, "MEX": 0.055, "DEU": 1.1}  # USD baseline

rows = []

for date in dates:
    month = date.month
    # Simple seasonality factor (sinusoidal: peak mid-year, dip start/end)
    season_factor = 1 + 0.2 * np.sin((month - 1) / 12 * 2 * np.pi)

    for country in countries:
        for product in products:
            # Generate macroeconomic variables (add small Gaussian noise)
            gdp_growth = np.random.normal(gdp_growth_mean[country], 0.5)
            inflation = np.random.normal(inflation_mean[country], 0.3)
            interest_rate = np.random.normal(interest_rate_mean[country], 0.4)
            exchange_rate = np.random.normal(exchange_rate_mean[country], 0.01)

            # Competitive pricing index (1.0 baseline ±5%)
            competitor_price_index = np.random.normal(1.0, 0.05)

            # Regulatory/tariff impact flag (10% chance of a tariff/regulation shock)
            regulation_tariff_flag = np.random.choice([0, 1], p=[0.9, 0.1])

            # Holiday/event flag (peak season: November & December)
            holiday_event_flag = 1 if month in [11, 12] else 0

            # Synthetic demand generation (units sold)
            base_units = baseline_sales[country] * product_adjustment[product] * season_factor
            macro_multiplier = 1 + (gdp_growth / 100)
            random_noise = np.random.normal(1.0, 0.1)

            units_sold = int(base_units * macro_multiplier * random_noise)

            rows.append(
                {
                    "date": date,
                    "country": country,
                    "product_code": product,
                    "units_sold": units_sold,
                    "gdp_growth_pct": round(gdp_growth, 2),
                    "inflation_rate_pct": round(inflation, 2),
                    "interest_rate_pct": round(interest_rate, 2),
                    "exchange_rate_to_usd": round(exchange_rate, 4),
                    "competitor_price_index": round(competitor_price_index, 3),
                    "regulation_tariff_flag": regulation_tariff_flag,
                    "holiday_event_flag": holiday_event_flag,
                }
            )

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
file_path = "output/demanda_productos_ejemplo.csv"
df.to_csv(file_path, index=False)
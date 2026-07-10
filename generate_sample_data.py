"""
generate_sample_data.py
------------------------
Creates a synthetic 'electricity.csv' dataset with the same schema used
in the original notebook (all features min-max scaled between 0 and 1):

    Temperature, Humidity, WindSpeed, HourOfDay, ElectricityDemand

This lets anyone clone the repo and immediately train/run the app even
without the original private dataset. Swap this file out for your own
real dataset (same column names) any time.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

temperature = np.random.rand(N)
humidity = np.random.rand(N)
wind_speed = np.random.rand(N)
hour_of_day = np.random.rand(N)

# Synthetic but plausible relationship:
# demand rises with temperature (AC load) and time-of-day peaks,
# and drops slightly with humidity/wind (comfort effect), plus noise.
electricity_demand = (
    0.45 * temperature
    + 0.25 * np.sin(hour_of_day * np.pi)
    - 0.15 * humidity
    + 0.10 * wind_speed
    + np.random.normal(0, 0.05, N)
)

# scale demand to 0-1 range like the rest of the columns
electricity_demand = (electricity_demand - electricity_demand.min()) / (
    electricity_demand.max() - electricity_demand.min()
)

df = pd.DataFrame(
    {
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
        "HourOfDay": hour_of_day,
        "ElectricityDemand": electricity_demand,
    }
)

df.to_csv("electricity.csv", index=False)
print(f"Saved synthetic dataset with {len(df)} rows to electricity.csv")

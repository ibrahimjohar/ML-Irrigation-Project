# ======================================================
# FULL GEOSPATIAL + SYNTHETIC AGRICULTURAL ML PIPELINE
# ======================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------
# 1. Load Shapefile
# ------------------------------------------------------
shapefile_path = r"C:\Ibrahim\Personal\University Stuff\Machine Learning\Project\ML Irrigation Project\data\geospatial\pakistan_shapefile\gadm41_PAK_2.shp"

gdf = gpd.read_file(shapefile_path)
print("Shapefile Loaded:", gdf.shape)

# Keep only needed columns
gdf = gdf[['GID_2', 'NAME_2', 'geometry']]


# ------------------------------------------------------
# 2. Generate Synthetic Agricultural Data
# ------------------------------------------------------
np.random.seed(42)

districts = gdf["NAME_2"].unique()
years = np.arange(2000, 2021)   # 21 years of data

synthetic_rows = []

for district in districts:
    for year in years:

        # Generate synthetic but realistic values
        rainfall = np.random.uniform(50, 500)              # mm
        temperature = np.random.uniform(12, 35)            # °C
        soil_moisture = np.random.uniform(10, 50)          # %
        crop_yield = 20 + 0.05 * rainfall - 0.2 * (temperature - 20) + np.random.uniform(-5, 5)
        irrigation = max(0, 100 - (rainfall / 5) + np.random.uniform(-10, 10))

        synthetic_rows.append([
            district,
            year,
            rainfall,
            temperature,
            soil_moisture,
            irrigation,
            crop_yield
        ])

df = pd.DataFrame(synthetic_rows, columns=[
    "District", "Year", "Rainfall", "Temperature", "Soil_Moisture",
    "Irrigation_Required", "Crop_Yield"
])

print("Synthetic Dataset Created:", df.shape)
print(df.head())


# ------------------------------------------------------
# 3. Merge Geospatial Data With Synthetic Data
# ------------------------------------------------------
geo_merged = gdf.merge(df, left_on="NAME_2", right_on="District", how="left")
print("Merged Dataset:", geo_merged.shape)
print(geo_merged.head())


# ------------------------------------------------------
# 4. Train ML Model to Predict Irrigation Need
# ------------------------------------------------------
X = df[["Rainfall", "Temperature", "Soil_Moisture"]]
y = df["Irrigation_Required"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

df["Predicted_Irrigation"] = model.predict(X)
geo_merged = geo_merged.merge(df[["District", "Year", "Predicted_Irrigation"]],
                              on=["District", "Year"], how="left")


# ------------------------------------------------------
# 5. Plot Geospatial Irrigation Heatmap
# ------------------------------------------------------
latest_year = df["Year"].max()
map_data = geo_merged[geo_merged["Year"] == latest_year]

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
map_data.plot(column="Predicted_Irrigation",
              cmap="YlOrRd",
              legend=True,
              edgecolor="black",
              ax=ax)

ax.set_title(f"Predicted Irrigation Requirement Across Districts ({latest_year})",
             fontsize=16)

plt.show()


# ------------------------------------------------------
# 6. Save Outputs (Optional)
# ------------------------------------------------------
geo_merged.to_file("synthetic_geo_irrigation.shp")
df.to_csv("synthetic_irrigation_dataset.csv", index=False)

print("\n--- PROCESS COMPLETE ---")
print("Generated:")
print("• synthetic_irrigation_dataset.csv")
print("• synthetic_geo_irrigation.shp (for GIS tools)")

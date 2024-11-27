import geopandas as gpd

file_path = "fl_2016/fl_2016.shp"  

gdf = gpd.read_file(file_path)


print(gdf.head())

print(gdf.columns)

import matplotlib.pyplot as plt

print(gdf.head())  # Shows the first few rows of the data
print(gdf.columns) 
print(gdf.iloc[0])  # Displays the data for the first precinct

#gdf.plot()

#plt.show()


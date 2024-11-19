import geopandas as gpd

file_path = "fl_2016/fl_2016.shp"  

gdf = gpd.read_file(file_path)


print(gdf.head())

print(gdf.columns)

import matplotlib.pyplot as plt

gdf.plot()
plt.show()
import geopandas as gpd
import folium

# Load the shapefile
file_path = "fl_2016/fl_2016.shp"
gdf = gpd.read_file(file_path)

# Create a folium map centered on Florida
m = folium.Map(location=[27.9944024, -81.7602544], zoom_start=6)

# Add GeoJSON layer with tooltips
for _, row in gdf.iterrows():
    # Extract district geometry and other attributes
    geometry = row['geometry']
    votes_info = f"District: {row['DISTRICT']}<br>Votes: {row['VOTES']}<br>Party: {row['PARTY']}"
    
    # Convert geometry to GeoJSON format and add it to the map
    folium.GeoJson(
        data=geometry,
        tooltip=folium.Tooltip(votes_info),  # Display info on hover
    ).add_to(m)

# Save map to HTML and display
m.save("interactive_map.html")
print("Map saved as interactive_map.html")

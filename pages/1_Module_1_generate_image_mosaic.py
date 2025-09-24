import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
from src.src_modul_1 import Reflectance_Data, Reflectance_Stats
import tempfile
import zipfile
import os
import ee
import datetime
ee.Authenticate()
ee.Initialize()

#title of the module
st.title("Search and Generate Landsat Image Mosaic")
#module name
markdown = """
This module allows users to search and generate a Landsat image mosaic for a specified area of interest (AOI) and time range using Google Earth Engine (GEE) data.
"""
#set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

#User input, AOI upload
st.subheader("Upload Area of Interest (Shapefile)")
st.markdown("currently the platform only support shapefile in .zip format")
uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type=["zip"])
aoi = None
#define AOI upload function
if uploaded_file:
    # Extract the uploaded zip file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # write uploaded bytes to disk (required before reading zip)
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the .shp file in the extracted files (walk subfolders)
        shp_files = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp"):
                    shp_files.append(os.path.join(root, fname))

        if len(shp_files) == 0:
            st.error("No .shp file found in the uploaded zip.")
        else:
            gdf = gpd.read_file(shp_files[0])
            st.success("AOI uploaded!")
            # Convert to EE geometry
            aoi = ee.FeatureCollection(gdf.__geo_interface__)
            #Show a small preview map centered on AOI
            st.text("Area of interest preview:")
            centroid = gdf.geometry.centroid.iloc[0]
            preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
            preview_map.add_geojson(gdf.__geo_interface__, layer_name="AOI")
            preview_map.to_streamlit(height=300)

#User input, search criteria
st.subheader("Specify Imagery Search Criteria")
st.text("Enter the acquisition date range, cloud cover percentage, and Landsat mission type. " \
"Current platform support Landsat 5-9 Collection 2 Level 2 Analysis Ready Data (ARD), excluding the thermal bands. Landsat mission avaliability is as follows:")
st.markdown("1. Landsat 5 Thematic Mapper (1984 - 2012)")
st.markdown("2. Landsat 7 Enhanced Thematic Mapper Plus (1999 - 2021)")
st.markdown("3. Landsat 8 Operational Land Imager (2013 - present)")
st.markdown("4. Landsat 9 Operational Land Imager-2 (2021 - present)")
#specified the avaliable sensor type
sensor_type = ['L5_SR', 'L7_SR', 'L8_SR', 'L9_SR']
#create a selection box for sensor type

# Map friendly names to internal codes
sensor_dict = {
    "Landsat 5 Thematic Mapper": "L5_SR",
    "Landsat 7 Enhanced Thematic Mapper Plus": "L7_SR",
    "Landsat 8 Operational Land Imager": "L8_SR",
    "Landsat 9 Operational Land Imager-2": "L9_SR"
}
sensor_names = list(sensor_dict.keys())

#User can select the sensor type here
selected_sensor_name = st.selectbox("Select Landsat Sensor:", sensor_names, index=2)
optical_data = sensor_dict[selected_sensor_name]  # This is what you pass to your backend
# Set default dates
default_start = datetime.date(2020, 1, 1)
default_end = datetime.date(2020, 12, 31)
start_date_dt = st.date_input("Start Date:", default_start)
end_date_dt = st.date_input("End Date:", default_end)
#Convert the date format to be compatible with GEE
start_date = start_date_dt.strftime("%Y-%m-%d")
end_date = end_date_dt.strftime("%Y-%m-%d")
#cloud cover slider
cloud_cover = st.slider("Maximum Cloud Cover (%):", 0, 100, 30)

#Search the landsat imagery
if st.button("Search Landsat Imagery") and aoi:
    reflectance = Reflectance_Data()
    collection, meta = reflectance.get_optical_data(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        optical_data=optical_data,
        cloud_cover=cloud_cover,
        verbose=False,
        compute_detailed_stats=False
    )
    stats = Reflectance_Stats()
    detailed_stats = stats.get_collection_statistics(collection, compute_stats=True, print_report=True)
    st.success(f"Found {detailed_stats['total_images']} images.")
        # Debug: check collection size (safe server-side call)
    try:
        coll_size = int(collection.size().getInfo())
    except Exception as e:
        st.error(f"Failed to query collection size: {e}")
        coll_size = 0
    st.write(f"Collection size: {coll_size}")

    if coll_size == 0:
        st.warning("No images found for the selected criteria.")

    # --- Format and display the report as Markdown ---
    summary_md = f"""
    ### Landsat Imagery Search Summary

    - **Total Images Found:** {detailed_stats.get('total_images', 'N/A')}
    - **Date Range of Images:** {detailed_stats.get('date_range', 'N/A')}
    - **Unique WRS Tiles:** {detailed_stats.get('unique_tiles', 'N/A')}
    - **Scene IDs:** {', '.join(detailed_stats.get('Scene_ids', [])) if detailed_stats.get('Scene_ids') else 'N/A'} 
    - **Image acquisition dates:** {', '.join(detailed_stats.get('individual_dates', [])) if detailed_stats.get('individual_dates') else 'N/A'}
    - **Average Scene Cloud Cover:** {detailed_stats.get('cloud_cover', {}).get('mean', 'N/A')}%
    - **Date Range:** {detailed_stats.get('date_range', 'N/A')}
    - **Cloud Cover Range:** {detailed_stats.get('cloud_cover', {}).get('min', 'N/A')}% - {detailed_stats.get('cloud_cover', {}).get('max', 'N/A')}%
    """
    st.markdown(summary_md)
    # Optionally, display the full stats as a table
    #st.subheader("Detailed Statistics") {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}
    #st.write(detailed_stats)
    if detailed_stats['total_images'] > 0:
        #visualization parameters
        vis_params = {
        'min': 0,
        'max': 0.4,
        'gamma': [0.95, 1.1, 1],
        'bands':['NIR', 'RED', 'GREEN']
        }
        #Create and image composite/mosaic
        composite = collection.median().clip(aoi)
        # Store in session state for use in other modules
        st.session_state['composite'] = composite
        # Display the image using geemap
        centroid = gdf.geometry.centroid.iloc[0]
        m = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
        m.addLayer(composite, vis_params, 'Landsat Mosaic')
        m.addLayer(collection, vis_params, 'Landsat Collection', shown=False)
        m.add_geojson(gdf.__geo_interface__, layer_name="AOI")
        m.to_streamlit(height=600)
else:
    st.info("Upload an AOI and specify search criteria to begin.")
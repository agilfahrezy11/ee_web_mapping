import streamlit as st
import leafmap.foliumap as leafmap
import geopandas as gpd
import ee
ee.Initialize()
from src.src_modul_1 import Reflectance_Data, Reflectance_Stats
import tempfile
import zipfile
import os
import io
import sys

#module name
markdown = """
Module 1: Generate Image Mosaic
"""
#set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

#User input, AOI upload
st.subheader("Step 1: Upload Area of Interest (Shapefile)")
st.markdown("currently the platform only support shapefile in .zip format")
uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type=["zip"])
aoi = None
#define AOI upload function
if uploaded_file:
    # Extract the uploaded zip file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        # Find the .shp file in the extracted files
        shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if len(shp_files) == 0:
            st.error("No .shp file found in the uploaded zip.")
        else:
            gdf = gpd.read_file(shp_files[0])
            st.success("AOI uploaded successfully!")
            st.write(gdf)
            # Convert to EE geometry
            aoi = ee.FeatureCollection(gdf.__geo_interface__)

#User input, search criteria
st.subheader("Step 2: Specify Search Criteria")
st.markdown("enter the acquisition date range, cloud cover percentage, and Landsat sensor type")
st.markdown( "current platform support Landsat 5, 7, 8 and 9 surface reflectance data. " \
"1. Landsat 5 (1984 - 2012) " \
"2. Landsat 7 (1999 - 2021)" \
"3. Landsat 8 (2013 - present)" \
"4. Landsat 9 (2021 - present)")
sensor_type = ['L5_SR', 'L7_SR', 'L8_SR', 'L9_SR']
optical_data = st.selectbox("Select Landsat Sensor:", sensor_type, index=2)
start_date = st.text_input("Start Date (YYYY-MM-DD or YYYY):", "2020-01-01")
end_date = st.text_input("End Date (YYYY-MM-DD or YYYY):", "2020-12-31")
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

    # --- Format and display the report as Markdown ---
    summary_md = f"""
    ### Landsat Imagery Search Summary

    - **Total Images Found:** {detailed_stats.get('total_images', 'N/A')}
    - **Mean Cloud Cover:** {detailed_stats.get('mean_cloud_cover', 'N/A')}%
    - **Date Range:** {detailed_stats.get('date_range', 'N/A')}
    - **Cloud Cover:** {detailed_stats.get('cloud_cover', 'N/A')}
    """
    st.markdown(summary_md)

    # Optionally, display the full stats as a table
    #st.subheader("Detailed Statistics")
    #st.write(detailed_stats)

    if detailed_stats['total_images'] > 0:
        # Create a median composite for visualization
        composite = collection.mosaic()
        map = leafmap.Map()
        map.add_ee_layer(composite, {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}, 'Landsat Mosaic')
        map.to_streamlit(height=500)
else:
    st.info("Upload an AOI and specify search criteria to begin.")


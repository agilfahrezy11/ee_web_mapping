import streamlit as st
from src.utils_shapefile_validation_conversion import shapefile_validator, EE_converter
import geemap.foliumap as geemap
import geopandas as gpd
import ee
import tempfile
import zipfile
import os
ee.Authenticate()
ee.Initialize()


#title of the module
st.title("Analzye the Separability of Region of Interest (ROI)")
st.markdown("This module is used so that user are able to perform separability analysis of the region of interest (ROI) data. " \
"The user must upload the training data in zip shape file format. The training data should contain the class ID and class names")
#module name
markdown = """
This module is designed to perform separability analysis of the training data.
"""
#set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

#User input, AOI upload
st.subheader("Upload Region of Interest (Shapefile)")
st.markdown("currently the platform only support shapefile in .zip format")
uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type=["zip"])
aoi = None
# At the top of your file
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
            try:
                # Read the shapefile
                gdf = gpd.read_file(shp_files[0])
                st.success("ROI loaded successfully!")
                validate = shapefile_validator(verbose=False)
                converter = EE_converter(verbose=False)
                st.markdown("ROI table preview:")
                st.write(gdf)
                # Validate and fix geometry
                gdf_cleaned = validate.validate_and_fix_geometry(gdf, geometry="mixed")
                
                if gdf_cleaned is not None:
                    # Convert to EE geometry safely
                    aoi = converter.convert_roi_gdf(gdf_cleaned)
                    
                    if aoi is not None:
                        st.success("ROI conversion completed!")
                        
                        # Show a small preview map centered on AOI
                        # Store in session state
                        st.session_state['training_data'] = aoi
                        st.session_state['training_gdf'] = gdf_cleaned
                        st.text("Region of Interest distribution:")
                        centroid = gdf_cleaned.geometry.centroid.iloc[0]
                        preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
                        preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="AOI")
                        preview_map.to_streamlit(height=500)
                    else:
                        st.error("Failed to convert ROI to Google Earth Engine format")
                else:
                    st.error("Geometry validation failed")
                    
            except Exception as e:
                st.error(f"Error reading shapefile: {e}")
                st.info("Make sure your shapefile includes all necessary files (.shp, .shx, .dbf, .prj)")
import streamlit as st
import pandas as pd
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
st.markdown("This module is used so that user are able to perform separability analysis of the region of interest (ROI) data." \
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
            st.success("ROI uploaded!")
            st.markdown("ROI table preview:")
            st.write(gdf)
            # Convert to EE FeatureCollection (supports Point, MultiPolygon, etc.)
            features = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                props = row.drop('geometry').to_dict()
                if geom.geom_type == "MultiPoint":
                    for pt in geom.geoms:
                        try:
                            features.append(ee.Feature(pt.__geo_interface__, props))
                        except Exception as e:
                            st.error(f"Error converting MultiPoint geometry: {e}")
                else:
                    try:
                        features.append(ee.Feature(geom.__geo_interface__, props))
                    except Exception as e:
                        st.error(f"Error converting geometry: {e}")
            if features:
                aoi = ee.FeatureCollection(features)
                # Show a small preview map centered on AOI
                st.text("Region of Interest data preview:")
                # Prepare geometry for map preview (flatten MultiPoint to Points)
                preview_geoms = []
                for geom in gdf.geometry:
                    if geom.geom_type == "MultiPoint":
                        preview_geoms.extend(list(geom.geoms))
                    else:
                        preview_geoms.append(geom)
                preview_gdf = gpd.GeoDataFrame(geometry=preview_geoms, crs=gdf.crs)
                centroid = preview_gdf.geometry.centroid.iloc[0]
                preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
                preview_map.add_geojson(preview_gdf.__geo_interface__, layer_name="ROI")
                preview_map.to_streamlit(height=500)
            else:
                st.error("No valid features found in shapefile.")

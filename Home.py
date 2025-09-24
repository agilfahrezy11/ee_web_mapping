import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

# Customize the sidebar
markdown = """
An working example module 1 and 3 of Epistem land cover mapping platform. Adapted from:
<https://github.com/opengeos/streamlit-map-template>
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

# Customize page title
st.title("Epistem land cover mapping platform demo")

st.markdown(
    """
    This multipage platform aims to demonstrate a working example of epistem's first module, aim to generate a Landsat imagery for certain area of interest.
    """
)

st.header("Instructions")

markdown = """
1. Define the area of interest by drawing a rectangle on the map or upload a shapefile (zip).
2. Specified the acqusition year of date. If you just type the year, the map will automatically filtered image from January 1 all the way to December 31.
3. Specified the cloud cover percentage and sensor type (currently support Landsat 5 TM - Landsat 9 OLI2).
4. click run to generate the mosaic image.
"""

st.markdown(markdown)

m = leafmap.Map(minimap_control=True)
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)

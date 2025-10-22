import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, Polygon, MultiPolygon
from src.module_helpers import init_gee
init_gee()
import ee
from datetime import datetime
import os
import folium
import matplotlib.pyplot as plt
from IPython.display import display
from shapely.geometry import mapping
import random
import geemap.foliumap as geemap

# --- System response 3.1 ---
class InputCheck:
    """
    System respons 3.1: Prequisite check
    """

    def ValidateVariable(*variable_names):
        """
        Validate that all specified variables exist in Streamlit session state.
        
        Args:
            *variable_names: Variable names as strings to check in st.session_state
        
        Returns:
            bool: True if all variables exist, raises error otherwise
            
        Usage:
            InputCheck.ValidateVariable('LULCTable', 'ClippedImage', 'AOI')
        """
        st.write("### Validating Required Variables")
        
        missing_variables = []
        validation_results = []
        #change from eval() to streamlit session state since the input here are generated from other module
        for var_name in variable_names:
            if var_name in st.session_state:
                var_value = st.session_state[var_name]
                var_type = type(var_value).__name__
                validation_results.append({
                    'Variable': var_name,
                    'Status': '✅ EXISTS',
                    'Type': var_type
                })
            else:
                validation_results.append({
                    'Variable': var_name,
                    'Status': '❌ NOT DEFINED',
                    'Type': 'N/A'
                })
                missing_variables.append(var_name)
        
        #Display results in a table
        results_df = pd.DataFrame(validation_results)
        st.dataframe(results_df, use_container_width=True)
        
        if missing_variables:
            error_msg = f"Missing required variables: {', '.join(missing_variables)}"
            st.error(f"❌ {error_msg}")
            st.stop()  # Streamlit way to stop execution
        else:
            st.success("✅ All required variables are present. Continuing execution...")
            return True
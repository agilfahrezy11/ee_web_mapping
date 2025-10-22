import streamlit as st
from src.module_helpers import init_gee, shapefile_validator, EE_converter
init_gee()
from src.src_modul_7 import Thematic_Accuracy_Assessment
import pandas as pd
import geemap.foliumap as geemap
import tempfile
import zipfile
import os
import geopandas as gpd
import plotly.express as px

#Page configuration
st.set_page_config(
    page_title="Thematic Accuracy Assessment",
    page_icon="logos/logo_epistem_crop.png",
    layout="wide"
)

#Initialize accuracy assessment manager
@st.cache_resource
def get_accuracy_manager():
    return Thematic_Accuracy_Assessment()

manager = get_accuracy_manager()

# Page header
st.title("Thematic Accuracy Assessment")
st.divider()

st.markdown("""
Evaluate the thematic accuracy of your land cover classification from Module 6 using independent validation data. 
In order to run this module you need an ground reference data containing class ID and class name similar to a ROI
The accuracy of land cover map is evaluated using a confusion matrix, with the following key metrics

- **Overall Accuracy** with confidence intervals
- **Kappa Coefficient** for agreement assessment  
- **F1-Score** for class-level performance
""")

st.markdown("---")

#This module wont run if classification result from module 6 is not avaliable
def check_prerequisites():
    """Check if required data from previous modules is available"""
    if 'classification_result' not in st.session_state or st.session_state.classification_result is None:
        st.error("❌ No classification result found from Module 6.")
        st.warning("Please complete Module 6 first to generate a land cover classification map.")
        st.stop()
    else:
        st.success("✅ Classification map loaded from Module 6")
        return st.session_state.classification_result

#Initialize the functions
lcmap = check_prerequisites()
#function to upload the ground reference data (similar to module 3, but in a function)
def process_shapefile_upload(uploaded_file):
    """Process uploaded shapefile and convert to Earth Engine format"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded file
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find shapefile
            shp_files = []
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        shp_files.append(os.path.join(root, fname))

            if not shp_files:
                return False, "No .shp file found in the uploaded zip.", None, None

            # Read and process shapefile
            gdf = gpd.read_file(shp_files[0])
            
            #Validate and clean geometry using the helper module 
            validator = shapefile_validator(verbose=False)
            converter = EE_converter(verbose=False)
            
            gdf_cleaned = validator.validate_and_fix_geometry(gdf, geometry="mixed")
            
            if gdf_cleaned is None:
                return False, "Geometry validation failed", None, None
            
            # Convert to Earth Engine format using helper module
            ee_data = converter.convert_roi_gdf(gdf_cleaned)
            
            if ee_data is None:
                return False, "Failed to convert to Google Earth Engine format", None, None
            
            return True, "Validation data processed successfully", ee_data, gdf_cleaned
            
    except Exception as e:
        return False, f"Error processing shapefile: {str(e)}", None, None

#similar to module 1 but wrap in function
def render_validation_upload():
    """Render validation data upload section"""
    st.subheader("Step 1: Upload Ground Reference Data")
    st.info("📁 Upload a **.zip shapefile** containing your independent validation samples with class IDs.")

    uploaded_file = st.file_uploader("Choose a zipped shapefile (.zip)", type=["zip"])

    if uploaded_file:
        with st.spinner("Processing validation data..."):
            success, message, ee_data, gdf_cleaned = process_shapefile_upload(uploaded_file)
            
            if success:
                st.success(f"✅ {message}")
                
                # Store in session state
                st.session_state['validation_data'] = ee_data
                st.session_state['validation_gdf'] = gdf_cleaned
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(gdf_cleaned.head(), use_container_width=True)
                
                # Show map preview
                st.markdown("**📍 Validation Points Distribution:**")
                centroid = gdf_cleaned.geometry.centroid.iloc[0]
                preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=8)
                preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="Validation Points")
                preview_map.to_streamlit(height=500)
                
            else:
                st.error(f"❌ {message}")
                if "Make sure your shapefile includes" not in message:
                    st.info("💡 Ensure your shapefile includes all necessary files (.shp, .shx, .dbf, .prj)")

#run validation data upload
render_validation_upload()
#Function for definining user input for accuracy assessment
def user_input_for_accuracy_assessment():
    """Render accuracy assessment configuration and execution"""
    st.divider()
    st.subheader("Step 2: Configure and Run Assessment")

    if "validation_data" not in st.session_state or st.session_state.validation_data is None:
        st.warning("⚠️ Please upload your validation data first.")
        return

    gdf_cleaned = st.session_state.get('validation_gdf')
    if gdf_cleaned is None:
        st.error("Validation data not properly loaded.")
        return

    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        class_prop = st.selectbox(
            "Class ID Field:",
            options=gdf_cleaned.columns.tolist(),
            index=gdf_cleaned.columns.get_loc("CLASS_ID") if "CLASS_ID" in gdf_cleaned.columns else 0,
            help="Field containing numeric class identifiers (e.g., 1, 2, 3, 4)"
        )
    
    with col2:
        scale = st.number_input(
            "Pixel Size (meters):",
            min_value=10,
            max_value=1000,
            value=30,
            help="Spatial resolution for sampling the classified map"
        )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        confidence = st.slider(
            "Confidence Level for Accuracy Intervals:",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )

    # Run assessment
    if st.button("🎯 Evaluate Map Accuracy", type="primary", use_container_width=True):
        with st.spinner("Running thematic accuracy assessment..."):
            success, results = manager.run_accuracy_assessment(
                lcmap=lcmap,
                validation_data=st.session_state.validation_data,
                class_property=class_prop,
                scale=scale,
                confidence=confidence
            )

            if success:
                st.session_state["accuracy_results"] = results
                st.success("✅ Thematic accuracy assessment completed!")
                st.rerun()
            else:
                st.error(f"❌ Assessment failed: {results.get('error', 'Unknown error')}")

#run user inpur function
user_input_for_accuracy_assessment()

#Function to display accuracy assessment
def render_accuracy_results():
    """Render accuracy assessment results"""
    #if not initialize yet
    if "accuracy_results" not in st.session_state:
        return
    #Then initialze the session state to store the accuracy
    results = st.session_state["accuracy_results"]
    
    if 'error' in results:
        st.error(f"❌ {results['error']}")
        return

    st.divider()
    st.subheader("Accuracy Assessment Results")

    #Prepared to display the key result
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy", 
            f"{results['overall_accuracy']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Kappa Coefficient", 
            f"{results['kappa']:.3f}"
        )
    
    with col3:
        ci = results['overall_accuracy_ci']
        confidence_pct = int(results['confidence_level'] * 100)
        st.metric(
            f"{confidence_pct}% Confidence Interval", 
            f"{ci[0]*100:.1f}% - {ci[1]*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Sample Size", 
            f"{results['n_total']} points"
        )

    # Class-level metrics table
    st.markdown("---")
    st.subheader("Class-Level Performance")

    df_metrics = pd.DataFrame({
        "Class ID": range(len(results['producer_accuracy'])),
        "Producer's Accuracy (%)": [round(v * 100, 2) for v in results['producer_accuracy']],
        "User's Accuracy (%)": [round(v * 100, 2) for v in results['user_accuracy']],
        "F1-Score (%)": [round(v * 100, 2) for v in results["f1_scores"]],
    })
    
    st.dataframe(df_metrics, use_container_width=True)

    # Confusion matrix visualization
    st.markdown("---")
    st.subheader("🔄 Confusion Matrix")

    cm_array = results["confusion_matrix"]
    n_classes = len(cm_array)
    
    cm_df = pd.DataFrame(
        cm_array,
        columns=[f"Predicted {i}" for i in range(n_classes)],
        index=[f"Actual {i}" for i in range(n_classes)]
    )
    
    # Create heatmap
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix (Actual vs Predicted Classes)"
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Export results option
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create downloadable results
        results_summary = {
            'Overall_Accuracy_Percent': results['overall_accuracy'] * 100,
            'Kappa_Coefficient': results['kappa'],
            'Confidence_Interval_Lower': results['overall_accuracy_ci'][0] * 100,
            'Confidence_Interval_Upper': results['overall_accuracy_ci'][1] * 100,
            'Sample_Size': results['n_total'],
            'Scale_Meters': results['scale']
        }
        
        results_df = pd.DataFrame([results_summary])
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results Summary",
            data=csv_data,
            file_name="accuracy_assessment_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download detailed metrics
        detailed_csv = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Class Metrics",
            data=detailed_csv,
            file_name="class_level_accuracy.csv",
            mime="text/csv",
            use_container_width=True
        )

def render_navigation():
    """Render navigation options"""
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Module 6", use_container_width=True):
            st.switch_page("pages/4_Module_6_Classification and LULC Creation.py")
    
    with col2:
        st.info("💡 Return to Module 6 to improve your classification model if needed")

# Render results and navigation
render_accuracy_results()
render_navigation()
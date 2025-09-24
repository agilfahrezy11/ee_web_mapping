import streamlit as st
import pandas as pd

st.title("Define Land Cover Land Use Classification Scheme")
#module name
markdown = """
In this module, users can define their own land cover land use (LCLU) classification scheme by specifying unique class ID, names and corresponding color codes. The defined scheme can be saved as a CSV file for future use in land cover classification tasks.
"""
#markdown description
st.markdown("User must define the desired land cover land use classification scheme. " \
"The class ID is a unique identifier for each land cover type, the class name is a descriptive label, and the color code is used for visual representation on maps. " \
"Users can add multiple classes, edit existing ones, and delete any unnecessary entries. Once the classification scheme is finalized, it can be downloaded as a CSV file for easy integration into other modules")

#set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

#Functionality to add, edit, delete classes and download as CSV
if "classes" not in st.session_state:
    st.session_state["classes"] = []
if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

st.subheader("Add a new class")

with st.form("add_class_form"):
    class_id = st.text_input("Class ID")
    class_name = st.text_input("Class Name")
    color_code = st.color_picker("Color Code")
    submitted = st.form_submit_button("Add Class")
    if submitted:
        if class_id and class_name and color_code:
            st.session_state["classes"].append({
                "Class ID": class_id,
                "Class Name": class_name,
                "Color Code": color_code
            })
        else:
            st.warning("Please fill in all fields.")

# Edit class form
if st.session_state["edit_index"] is not None:
    idx = st.session_state["edit_index"]
    edit_class = st.session_state["classes"][idx]
    with st.form("edit_class_form"):
        new_id = st.text_input("Edit Class ID", edit_class["Class ID"])
        new_name = st.text_input("Edit Class Name", edit_class["Class Name"])
        new_color = st.color_picker("Edit Color Code", edit_class["Color Code"])
        save_edit = st.form_submit_button("Save Changes")
        cancel_edit = st.form_submit_button("Cancel")
        if save_edit:
            st.session_state["classes"][idx] = {
                "Class ID": new_id,
                "Class Name": new_name,
                "Color Code": new_color
            }
            st.session_state["edit_index"] = None
        elif cancel_edit:
            st.session_state["edit_index"] = None

# Display the table of classes with edit/delete options
if st.session_state["classes"]:
    st.subheader("Classification Scheme")
    for i, row in enumerate(st.session_state["classes"]):
        cols = st.columns([2, 3, 2, 1, 1])
        cols[0].write(row["Class ID"])
        cols[1].write(row["Class Name"])
        cols[2].write(row["Color Code"])
        if cols[3].button("Edit", key=f"edit_{i}"):
            st.session_state["edit_index"] = i
        if cols[4].button("Delete", key=f"delete_{i}"):
            st.session_state["classes"].pop(i)
            st.session_state["edit_index"] = None
            st.rerun()
    #Visualize in panda dataframe and download as CSV
    df = pd.DataFrame(st.session_state["classes"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classification Scheme as CSV",
        data=csv,
        file_name="classification_scheme.csv",
        mime="text/csv"
    )
else:
    st.info("No classes added yet. Please add at least one class.")


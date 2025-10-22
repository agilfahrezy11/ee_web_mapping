import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any


class LULC_Scheme_Manager:
    """
    Module 2: Land Cover Classification Scheme Manager
    Backend processing for classification scheme management
    """
    
    def __init__(self):
        """Initialize session state variables for LULC scheme management"""
        self._init_session_state()
    #adapted from def init 
    #use st.session state to store result, instead of using self.
    #line 14 - 18
    def _init_session_state(self) -> None:
        """Initialize all required session state variables"""
        session_vars = {
            'lulc_classes': [],
            'lulc_next_id': 1,
            'lulc_edit_mode': False,
            'lulc_edit_idx': None,
            'csv_temp_classes': []
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    #Adapted from widget based layout (jupyter notebook) to streamlit session
    #@ is decorator or wrapper, so that when called, it does not need parenthesis
    #it provide clean access to each variable in streamlit session state
    #The original code (faza) use jupyter notebook UI, which is incompatible with streamlit
    @property
    def classes(self) -> List[Dict[str, Any]]:
        """Get classes from session state"""
        return st.session_state.lulc_classes
    #adapter from line 46-51 (self.class_id_input)
    @classes.setter
    def classes(self, value: List[Dict[str, Any]]) -> None:
        """Set classes in session state"""
        st.session_state.lulc_classes = value
    
    @property 
    def next_id(self) -> int:
        """Get next ID from session state"""
        return st.session_state.lulc_next_id
    
    @next_id.setter
    def next_id(self, value: int) -> None:
        """Set next ID in session state"""
        st.session_state.lulc_next_id = value

    def validate_class_input(self, class_id: Any, class_name: str) -> tuple[bool, Optional[str]]:
        """Validate class input parameters"""
        #Validate class_id type
        try:
            class_id = int(class_id)
        except (ValueError, TypeError):
            return False, "Class ID must be a valid number!"
        
        #Validate class name
        class_name = class_name.strip()
        if not class_name:
            return False, "Class name cannot be empty!"
        
        #Check if ID already exists (only for new classes)
        if not st.session_state.lulc_edit_mode:
            if any(c['ID'] == class_id for c in self.classes):
                return False, f"Class ID {class_id} already exists!"
        
        return True, None
    #adapted from line 184, but tailored with streamlit compability    
    def add_class(self, class_id: Any, class_name: str, color_code: str) -> tuple[bool, str]:
        """Add or update a class in the classification scheme"""
        # Validate input
        is_valid, error_msg = self.validate_class_input(class_id, class_name)
        if not is_valid:
            return False, error_msg
        
        class_id = int(class_id)
        class_name = class_name.strip()
        
        class_data = {
            'ID': class_id,
            'Class Name': class_name,
            'Color Code': color_code
        }
        
        # Update existing class
        if st.session_state.lulc_edit_mode and st.session_state.lulc_edit_idx is not None:
            self.classes[st.session_state.lulc_edit_idx] = class_data
            success_msg = f"Class '{class_name}' (ID: {class_id}) updated successfully!"
            self._reset_edit_mode()
        else:
            # Add new class
            self.classes.append(class_data)
            success_msg = f"Class '{class_name}' (ID: {class_id}) added successfully!"
        
        self._sort_and_update_next_id()
        return True, success_msg

    #functions for manual input options
    def _reset_edit_mode(self) -> None:
        """Reset edit mode session state"""
        st.session_state.lulc_edit_mode = False
        st.session_state.lulc_edit_idx = None
    
    def _sort_and_update_next_id(self) -> None:
        """Sort classes by ID and update next available ID"""
        self.classes = sorted(self.classes, key=lambda x: x['ID'])
        
        if self.classes:
            self.next_id = max([c['ID'] for c in self.classes]) + 1
        else:
            self.next_id = 1
    #adapted from line 229 onward
    def edit_class(self, idx: int) -> Optional[Dict[str, Any]]:
        """Set class for editing mode"""
        if 0 <= idx < len(self.classes):
            st.session_state.lulc_edit_mode = True
            st.session_state.lulc_edit_idx = idx
            return self.classes[idx]
        return None
    #adapted from line 247
    def delete_class(self, idx: int) -> tuple[bool, str]:
        """Delete a class from the scheme"""
        if 0 <= idx < len(self.classes):
            class_to_delete = self.classes[idx]
            del self.classes[idx]
            success_msg = f"Class '{class_to_delete['Class Name']}' (ID: {class_to_delete['ID']}) deleted successfully!"
            return True, success_msg
        return False, "Invalid class index"
    
    def cancel_edit(self) -> None:
        """Cancel edit mode"""
        self._reset_edit_mode()

    #adapted from line 266 - 320
    #Change so that csv is more tolaratable 

    def process_csv_upload(self, df: pd.DataFrame, id_col: str, name_col: str) -> tuple[bool, str]:
        """Process CSV upload - validate and prepare for color assignment"""
        try:
            class_list = []
            used_ids = set()

            for _, row in df.iterrows():
                class_id = row[id_col]
                class_name = row[name_col]

                # Skip empty rows
                if pd.isna(class_id) or pd.isna(class_name):
                    continue

                # Validate and convert class_id
                try:
                    class_id = int(class_id)
                except (ValueError, TypeError):
                    return False, f"Invalid Class ID format: {class_id}. Must be a number."

                # Check for duplicates
                if class_id in used_ids:
                    return False, f"Duplicate Class ID found: {class_id}"
                used_ids.add(class_id)

                class_list.append({
                    "ID": class_id,
                    "Class Name": str(class_name).strip(),
                    "Color Code": "#2e8540"  # Default color
                })

            st.session_state.csv_temp_classes = class_list
            return True, f"Successfully loaded {len(class_list)} classes from CSV"

        except Exception as e:
            return False, f"Error processing CSV: {str(e)}"
    
    def finalize_csv_upload(self, color_assignments: List[str]) -> tuple[bool, str]:
        """Finalize CSV upload with user-assigned colors"""
        try:
            # Update colors based on user assignments
            for i, class_data in enumerate(st.session_state.csv_temp_classes):
                if i < len(color_assignments):
                    class_data["Color Code"] = color_assignments[i]
            
            # Save to main classes and sort
            self.classes = st.session_state.csv_temp_classes.copy()
            self._sort_and_update_next_id()
            
            # Clear temporary storage
            st.session_state.csv_temp_classes = []
            
            return True, f"Classification scheme created with {len(self.classes)} classes"
            
        except Exception as e:
            return False, f"Error finalizing CSV upload: {str(e)}"
    def get_csv_data(self) -> Optional[bytes]:
        """Generate CSV data for download"""
        if not self.classes:
            return None
        
        df = self.get_dataframe()
        return df.to_csv(index=False).encode('utf-8')
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the classification scheme as a normalized DataFrame"""
        if not self.classes:
            return pd.DataFrame(columns=["ID", "Land Cover Class", "Color Palette"])

        df = pd.DataFrame(self.classes)

        # Normalize column names
        column_mapping = {
            "ID": "ID",
            "Class ID": "ID", 
            "Class Name": "Land Cover Class",
            "Land Cover Class": "Land Cover Class",
            "Color": "Color Palette",
            "Color Code": "Color Palette",
            "Color Palette": "Color Palette"
        }

        # Apply column renaming
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure consistent column order
        expected_columns = ["ID", "Land Cover Class", "Color Palette"]
        available_columns = [col for col in expected_columns if col in df.columns]
        
        return df[available_columns]

    #Adapted from line 407
    def load_default_scheme(self, scheme_name: str) -> tuple[bool, str]:
        """Load a predefined classification scheme"""
        default_schemes = self.get_default_schemes()
        
        if scheme_name not in default_schemes:
            return False, f"Unknown scheme: {scheme_name}"
        
        self.classes = default_schemes[scheme_name].copy()
        self._sort_and_update_next_id()
        
        return True, f"Loaded {scheme_name} with {len(self.classes)} classes"
    #Add RESTORE+ classification scheme
    @staticmethod
    def get_default_schemes() -> Dict[str, List[Dict[str, Any]]]:
        """Return available default classification schemes"""
        return {
            "RESTORE+ Project": [
                {'ID': 1, 'Class Name': 'Natural Forest', 'Color Code': "#0E6D0E"},
                {'ID': 2, 'Class Name': 'Agroforestry', 'Color Code': "#F08306"},
                {'ID': 3, 'Class Name': 'Monoculture Plantation', 'Color Code': "#38E638"},
                {'ID': 4, 'Class Name': 'Grassland or Savanna', 'Color Code': "#80DD80"},
                {'ID': 5, 'Class Name': 'Shrub', 'Color Code': "#5F972A"},
                {'ID': 6, 'Class Name': 'Paddy Field', 'Color Code': "#777907"},
                {'ID': 7, 'Class Name': 'Cropland (Palawija, Horticulture)', 'Color Code': "#E8F800"},
                {'ID': 8, 'Class Name': 'Settlement', 'Color Code': "#F81D00"},
                {'ID': 9, 'Class Name': 'Cleared Land', 'Color Code': "#E9B970"},
                {'ID': 10, 'Class Name': 'Waterbody', 'Color Code': "#1512F3"},
            ]
        }
    
    @staticmethod
    def auto_detect_csv_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Auto-detect ID and Name columns in CSV"""
        columns_lower = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]
        
        def find_column(keywords: List[str]) -> Optional[str]:
            for keyword in keywords:
                for i, col in enumerate(columns_lower):
                    if keyword in col:
                        return df.columns[i]
            return None
        
        id_col = find_column(["id", "classid", "kode"])
        name_col = find_column(["classname", "class", "kelas", "name"])
        
        return id_col, name_col 
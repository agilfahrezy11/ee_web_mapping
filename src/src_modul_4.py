import ee
import pandas as pd
import numpy as np
from src.module_helpers import init_gee
init_gee()

class sample_quality:
    """
    class which contain several functions used to conduct training data/sample analysis 
    
    """
    def __init__(self, training_data, image, class_property, region, class_name_property=None):
        """
        Initialize the tools for conducting the analysis
        Args:
            training_data: ee.FeatureCollection - Training polygons/points
            image: ee.Image - The image to extract spectral values
            class_property: str - Property/columen containing class ID (unique)
            region: ee.Geometry - Optional region to limit analysis
            class_name_property: str - Property/column containing class name
        """
        self.training_data = training_data
        self.image = image
        self.class_property = class_property
        self.class_name_property = class_name_property
        self.region = region
        self.band_names = self.image.bandNames().getInfo()
        self.class_mapping = None
    def get_display_property(self):
        """
        Helper functions to determine which properties (column in the training data) to display
        """
        return self.class_name_property if self.class_name_property else self.class_property
    def class_renaming(self):
        """
        Function to mapped between class ID and class names
        """
        if self.class_mapping is None and self.class_name_property:
            try:
                #mapped the combination between ID and names
                pairs = self.training_data.distinct([self.class_property, self.class_name_property])
                pair_info = pairs.getInfo()
                mapping = {}
                for features in pair_info['features']:
                    prop = features['properties']
                    class_id = prop[self.class_property]
                    class_name = prop[self.class_name_property]
                    # Convert float to int for consistent mapping keys
                    if isinstance(class_id, (int, float)):
                        mapping[int(class_id)] = class_name
                    else:
                        mapping[class_id] = class_name
                return mapping
            except Exception as e:
                print(f"Warning: Could not create class mapping: {e}")
                return {}
        return self.class_mapping or {}
    def add_class_names(self, df):
        """
        Add class names to dataframe based on class ids
        """ 
        id_column = self.class_property
        class_name = self.class_name_property  # Add this line
        mapping = self.class_renaming()
        if mapping:
            df[class_name] = df[id_column].astype(float).astype(int).map(lambda x: mapping.get(x, f"Class {x}"))
            cols = df.columns.tolist()
            if class_name in cols and id_column in cols:  
                id_idx = cols.index(id_column)
                cols.remove(class_name)  
                cols.insert(id_idx + 1, class_name) 
                df = df[cols]
        return df

    #Basic statistic of the training data:
    #Note that this process applied to training data before pixel value is extracted
    def sample_stats(self):
        """
        Get basic statistics about the training dataset. 
        """
        try:
            # Perfom in server side to minimize computational load
            class_counts = self.training_data.aggregate_histogram(self.class_property)
            total_samples = self.training_data.size()
            unique_classes = self.training_data.aggregate_array(self.class_property).distinct()
            # Get class names if available
            class_names = None
            if self.class_name_property:
                class_names = self.training_data.aggregate_histogram(self.class_name_property)            
            #return the dictionary
            results = ee.Dictionary({
                'class_counts': class_counts,
                'total_samples': total_samples,
                'unique_classes': unique_classes
            }).getInfo()
            if class_names:
                results['class_names'] = class_names            
            # Process results client-side
            class_counts_dict = results['class_counts']
            total_count = results['total_samples']
            classes_list = results['unique_classes']
            stats_dict = {
                'total_samples': total_count,
                'num_classes': len(classes_list),
                'class_counts': class_counts_dict,
                'classes': classes_list,
                'class_balance': {str(k): v/total_count for k, v in class_counts_dict.items()}
            }
            # Add class names mapping if available
            if 'class_names' in results:
                stats_dict['class_names'] = results['class_names']            
            return stats_dict
            
        except Exception as e:
            print(f"Error in get_basic_statistics: {e}")
            return None
    def get_sample_stats_df(self):
        """
        Get sample statistics as a formatted DataFrame
        """
        stats = self.sample_stats()
        if not stats:
            return pd.DataFrame()
        
        # Create DataFrame from class counts
        stats_data = []
        for class_id, count in stats['class_counts'].items():
            proportion = stats['class_balance'][str(class_id)]
            stats_data.append({
                'Class_ID': class_id,
                'Sample_Count': count,
                'Proportion': proportion,
                'Percentage': proportion * 100
            })
        
        df = pd.DataFrame(stats_data)
        df = df.sort_values('Class_ID').reset_index(drop=True)
        df = df.rename(columns={'Class_ID': self.class_property})
        # Add class names if available
        df = self.add_class_names(df)
        
        # Format percentage column
        df['Percentage'] = df['Percentage'].round(2)
        df['Proportion'] = df['Proportion'].round(4)
        
        return df
    #This function extract the pixel for all of the samples. 
    def extract_spectral_values(self, scale=30, max_pixels_per_class=7000):
        """
        Extract spectral values for all training samples. 
        
        Args:
            scale: int - Scale for sampling (meters)
            max_pixels_per_class: int - Maximum pixels per class to prevent memory issues
            
        Returns:
            pandas.DataFrame - Spectral values with class labels
        """
        try:
            # Limit Sample Size to prevent computational time out
            def limit_samples_per_class(class_value):
                class_samples = self.training_data.filter(
                    ee.Filter.eq(self.class_property, class_value)
                )
                # Use ee.Algorithms.If for server-side conditional logic
                sample_count = class_samples.size()
                limited_samples = ee.Algorithms.If(
                    sample_count.gt(max_pixels_per_class),
                    class_samples.randomColumn('random', 42).sort('random').limit(max_pixels_per_class),
                    class_samples
                )
                return ee.FeatureCollection(limited_samples)
            
            # Get unique classes
            unique_classes = self.training_data.aggregate_array(self.class_property).distinct()
            # Create balanced sample
            balanced_samples = ee.FeatureCollection(
                unique_classes.map(limit_samples_per_class)
            ).flatten()
            # Define properties to extract
            properties_to_extract = [self.class_property]
            if self.class_name_property:
                properties_to_extract.append(self.class_name_property)            
            # Extract the Pixel Value based on the training data
            training_sample = self.image.sampleRegions(
                collection=balanced_samples,
                properties=properties_to_extract,
                scale=scale,
                geometries=False,
                tileScale=16  # Increase tile scale to handle larger computations
            )
            # Single getInfo() call to get all data
            sample_data = training_sample.getInfo()
            if 'features' not in sample_data or len(sample_data['features']) == 0:
                print("Warning: No spectral data extracted. Check your training data quantity. Increase pixel size to reduce the size of sample")
                return pd.DataFrame()
            # Convert to panda
            df = pd.DataFrame([feat['properties'] for feat in sample_data['features']])
            
            # Data Cleaning
            # only keep the resulting 
            spectral_columns = [col for col in df.columns
                                if col !=self.class_property and col in self.band_names]
            # Convert only spectral columns to numeric
            for col in spectral_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')            
            df = df.dropna(subset=spectral_columns, how='all')
            print(f"Extracted spectral values for {len(df)} samples across {df[self.class_property].nunique()} classes")  
            return df
            
        except Exception as e:
            print(f"Error in extract_spectral_values: {e}")
            return pd.DataFrame() 
   #Calculate statistics of the training data after pixel value extraction               
    def sample_pixel_stats (self, df):
        """
        Compute statistics for sample extracted from the satellite imagery
        """
         #If the panda dataframe from previous step failed
        #if df.empty:
         #  return {}
         #Define the properties which include band pixel value   
        spectral_bands = [col for col in df.columns if col != self.class_property and col in self.band_names]
        classes = df[self.class_property].unique()
        properties_to_exclude = [self.class_property]
         #Define the properties which include band pixel value   
        if self.class_name_property:
            properties_to_exclude.append(self.class_name_property)        
        stats_byclass = {}
        #mapped them accross the class
        for class_name in classes:
            class_data = df[df[self.class_property] == class_name]
            stats_class ={}
            for band in spectral_bands:
                if band in class_data.columns:
                    band_values = class_data[band].dropna()
                    if len(band_values) > 0:
                        stats_class[band] = {
                            'mean': float(band_values.mean()),
                            'std' : float(band_values.std()),
                            'min': float(band_values.min()),
                            'max': float(band_values.max()),
                            'median': float(band_values.median()),
                            'count': int(len(band_values))
                        }
            stats_byclass[str(class_name)] = stats_class
        return stats_byclass
    def get_sample_pixel_stats_df(self, df):
        """
        Get pixel statistics as a formatted DataFrame
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe from extract_spectral_values
        statistic : str
            Which statistic to display ('mean', 'std', 'min', 'max', 'median', 'count')
        Returns:
        --------
        pandas.DataFrame
            Formatted statistics with class names if available
        """
        stats = self.sample_pixel_stats(df)
        if not stats:
            return pd.DataFrame()
        
        # Convert to long format DataFrame
        stats_data = []
        for class_id, bands in stats.items():
            for band, band_stats in bands.items():
                stats_data.append({
                    'Class_ID': int(class_id),
                    'Band': band,
                    'Mean': round(band_stats['mean'], 2),
                    'Std': round(band_stats['std'], 2),
                    'Min': round(band_stats['min'], 2),
                    'Max': round(band_stats['max'], 2),
                    'Median': round(band_stats['median'], 2),
                    'Count': band_stats['count']
                })
        
        result_df = pd.DataFrame(stats_data)
        result_df = result_df.sort_values(['Class_ID', 'Band']).reset_index(drop=True)
        result_df = result_df.rename(columns={'Class_ID': self.class_property})
        
        # Add class names if available
        result_df = self.add_class_names(result_df)
        
        return result_df
    #Calculate class separability using jeffries matusita distance
    #Note: This approach is not ideal for machine learning classifier, since JM distance works best with gaussian distribution
    def check_class_separability(self, df, method='JM'):
        """
        Calculate class separability.
        Available methods: 'JM' (Jeffries-Matusita), 'TD' (Transformed Divergence)
        """
        spectral_bands = [col for col in df.columns if col != self.class_property and col in self.band_names]
        classes = df[self.class_property].unique()

        if len(classes) < 2:
            print("Warning: Required at least 2 class for separability analysis")
            return {}

        separability_matrix = {}

        for i, class1 in enumerate(classes):
            separability_matrix[str(class1)] = {}
            for j, class2 in enumerate(classes):
                if i != j:
                    try:
                        data1 = df[df[self.class_property] == class1][spectral_bands].dropna().values
                        data2 = df[df[self.class_property] == class2][spectral_bands].dropna().values

                        if len(data1) == 0 or len(data2) == 0:
                            separability_matrix[str(class1)][str(class2)] = 0.0
                            continue

                        if method == 'JM':
                            sep = self._jeffries_matusita_distance(data1, data2)
                        elif method == 'TD':
                            sep = self.transform_divergence(data1, data2)
                        else:
                            raise ValueError("Unsupported method. Choose 'JM' or 'TD'.")

                        separability_matrix[str(class1)][str(class2)] = float(sep)

                    except Exception as e:
                        print(f"Error calculating separability for classes {class1}-{class2}: {e}")
                        separability_matrix[str(class1)][str(class2)] = 0.0
                else:
                    separability_matrix[str(class1)][str(class2)] = 0.0

        return separability_matrix

    def get_separability_df(self, df, method='JM'):
        """
        Get separability results as a DataFrame.
        method: 'JM' (Jeffries-Matusita) or 'TD' (Transformed Divergence)
        """
        separability = self.check_class_separability(df, method=method)
        if not separability:
            return pd.DataFrame()

        pairs = []
        class_mapping = self.class_renaming()
        metric_name = "JM_Distance" if method == 'JM' else "TD_Distance"

        for class1_id, class1_data in separability.items():
            for class2_id, value in class1_data.items():
                if value > 0:
                    # Use consistent key type for mapping
                    try:
                        class1_key = int(class1_id)
                    except ValueError:
                        class1_key = class1_id
                    try:
                        class2_key = int(class2_id)
                    except ValueError:
                        class2_key = class2_id

                    class1_name = class_mapping.get(class1_key, f"Class {class1_id}")
                    class2_name = class_mapping.get(class2_key, f"Class {class2_id}")
                    pairs.append({
                        'Class1_ID': class1_id,
                        'Class1_Name': class1_name,
                        'Class2_ID': class2_id,
                        'Class2_Name': class2_name,
                        metric_name: round(value, 3),
                        'Separability_Level': self.separability_level(value)
                    })

        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(metric_name).reset_index(drop=True)
        return pairs_df

    def lowest_separability(self, df, top_n=10, method='JM'):
        """
        Get the class pairs with the lowest separability value.
        Args:
            df: DataFrame with spectral values
            top_n: Number of lowest pairs to return
            method: 'JM' or 'TD'
        Returns:
            DataFrame of lowest separability pairs
        """
        separability_df = self.get_separability_df(df, method=method)
        if separability_df.empty:
            return pd.DataFrame()
        metric_name = "JM_Distance" if method == 'JM' else "TD_Distance"
        lowest_pairs = separability_df.head(top_n).copy()
        lowest_pairs['Interpretation'] = lowest_pairs[metric_name].apply(
            lambda x: self.separability_level(x, method=method)
        )
        return lowest_pairs

    def separability_level(self, value, method='JM'):
        """
        Categorize separability value for JM or TD distance.
        Args:
            value: Separability metric value
            method: 'JM' or 'TD'
        Returns:
            String interpretation
        """
        # Thresholds are for JM/TD (range 0–2)
        if value >= 1.8:
            return "Good Separability"
        elif 1.0 <= value < 1.8:
            return "Weak/marginal Separability"
        else:
            return "Class confusions"

    def sum_separability(self, df):
        """
        summarize the separability result
        """
        sum_df = self.get_separability_df(df)
        if sum_df.empty:
            return pd.DataFrame()
        summary = {
            'Total Pairs': len(sum_df),
            'Good Separability Pairs': len(sum_df[sum_df["Separability_Level"] == "Good Separability"]),
            'Weak Separability Pairs': len(sum_df[sum_df["Separability_Level"] == "Weak/marginal Separability"]),
            'Worst Separability Pairs': len(sum_df[sum_df["Separability_Level"] == "Class confusions"])
        }
        return pd.DataFrame(summary, index =[0])

    def _jeffries_matusita_distance(self, class1_data, class2_data):
        """Calculate Jeffries-Matusita distance between two classes"""
        try:
            if len(class1_data) < 2 or len(class2_data) < 2:
                return 0.0
                
            mean1 = np.mean(class1_data, axis=0)
            mean2 = np.mean(class2_data, axis=0)
            
            # Handle single band case
            if class1_data.shape[1] == 1:
                var1 = np.var(class1_data)
                var2 = np.var(class2_data)
                cov1 = np.array([[var1]]) if var1 > 0 else np.array([[1e-6]])
                cov2 = np.array([[var2]]) if var2 > 0 else np.array([[1e-6]])
            else:
                cov1 = np.cov(class1_data.T)
                cov2 = np.cov(class2_data.T)
                
                # Ensure covariance matrices are 2D
                if cov1.ndim == 0:
                    cov1 = np.array([[cov1]])
                if cov2.ndim == 0:
                    cov2 = np.array([[cov2]])
            
            # Add regularization for numerical stability
            reg_factor = 1e-6
            cov1 += np.eye(cov1.shape[0]) * reg_factor
            cov2 += np.eye(cov2.shape[0]) * reg_factor
            
            cov_mean = (cov1 + cov2) / 2
            diff_mean = mean1 - mean2
            
            # Calculate Bhattacharyya distance
            try:
                term1 = 0.125 * np.dot(np.dot(diff_mean.T, np.linalg.inv(cov_mean)), diff_mean)
            except np.linalg.LinAlgError:
                term1 = 0.125 * np.dot(np.dot(diff_mean.T, np.linalg.pinv(cov_mean)), diff_mean)
            
            try:
                det_cov_mean = np.linalg.det(cov_mean)
                det_cov1 = np.linalg.det(cov1)
                det_cov2 = np.linalg.det(cov2)
                
                if det_cov_mean <= 0 or det_cov1 <= 0 or det_cov2 <= 0:
                    term2 = 0.0
                else:
                    term2 = 0.5 * np.log(det_cov_mean / np.sqrt(det_cov1 * det_cov2))
            except:
                term2 = 0.0
            
            bhatt_dist = term1 + term2
            
            #Calculate Jeffries Matusita Distance
            jm_dist = 2 * (1 - np.exp(-bhatt_dist))
            
            return float(np.clip(jm_dist, 0, 2))
            
        except Exception as e:
            print(f"Error in JM distance calculation: {e}")
            return 0.0
        
    def transform_divergence(self, class1_data, class2_data):
        """Calculate Transform Divergence between two classes"""
        try:
            #Compute mean and covariance
            mean1 = np.mean(class1_data, axis=0)
            mean2 = np.mean(class2_data, axis=0)
            cov1 = np.cov(class1_data.T) + np.eye(class1_data.shape[1]) * 1e-6
            cov2 = np.cov(class2_data.T) + np.eye(class2_data.shape[1]) * 1e-6
            diff_mean  = mean1 - mean2
            #inverse covariance matrix
            try:
                inv_cov1 = np.linalg.inv(cov1)
            except np.linalg.LinAlgError:
                inv_cov1 = np.linalg.pinv(cov1)
            try:
                inv_cov2 = np.linalg.inv(cov2)
            except np.linalg.LinAlgError:
                inv_cov2 = np.linalg.pinv(cov2)
            #compute Divergence
            term1 = 0.5 * np.trace((cov1 - cov2) @ (inv_cov2 - inv_cov1))
            term2 = 0.5 * diff_mean.T @ (inv_cov1 + inv_cov2) @ diff_mean
            divergence = term1 + term2
            #Apply transformation
            td = 2 * (1 - np.exp(-divergence/8))
            return float(np.clip(td, 0, 2))
        except Exception as e:
            print(f"Error in Transform Divergence calculation: {e}")
            return 0.0
    #Plotting data distribution for evaluation
    #Data Histogram

    def print_analysis_summary(self, df):
        """
        Print a comprehensive analysis summary with all key results
        """
        # Class Separability
        print("\n\n1. Class Separability Summary:")
        print("-" * 40)
        sep_summary = self.sum_separability(df)
        if not sep_summary.empty:
            print(sep_summary.to_string(index=False))
        
        # 4. Most Problematic Class Pairs
        print("\n\n2. Lowest Separability Between Class:")
        print("-" * 40)
        lowest_pairs = self.lowest_separability(df, top_n=10)
        if not lowest_pairs.empty:
            display_cols = ['Class1_Name', 'Class2_Name', 'JM_Distance', 'Separability_Level', 'Interpretation']
            print(lowest_pairs[display_cols].to_string(index=False))
        
        print("\n" + "="*80)
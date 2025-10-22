import pandas as pd
import numpy as np
import ee
from src.module_helpers import init_gee
init_gee()

class FeatureExtraction:
    """
    Perform feature extraction as one of the input for land cover classification. Three types of split is presented here:
    Random Split: splitting the input data randomly based on specified split ratio 
    """
    def __init__(self):
        """
        Initializing the class function for feature extraction
        """
    ############################# 1. Single Random Split ###########################
    #extract pixel value for the labeled region of interest and partitioned them into training and testing data
    #This can be used if the training/reference data is balanced across class and required more fast result
    def random_split(self, image, roi, class_property, split_ratio = 0.6, pixel_size = 30, tile_scale=16):
        """
        Perform single random split and extract pixel value from the imagery
            Parameters:
                image = ee.Image
                aoi = area of interest, ee.FeatureCollection
                split_ratio = 
            Returns:
                tuple: (training_samples, testing_samples)
        """
        #create a random column
        roi_random = roi.randomColumn()
        #partioned the original training data
        training = roi_random.filter(ee.Filter.lt('random', split_ratio))
        testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
        #extract the pixel values
        training_pixels = image.sampleRegions(
                            collection=training,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        testing_pixels = image.sampleRegions(
                            collection=testing,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        print('Single Random Split Training Pixel Size:', training_pixels.size().getInfo())
        print('Single Random Split Testing Pixel Size:', testing_pixels.size().getInfo())
        return training_pixels, testing_pixels
    ############################## 2. Strafied Random Split ###########################
    # Conduct stratified train and test split, ideal for proportional split of the data
    def stratified_split (self, roi, image, class_prop, pixel_size= 30, train_ratio = 0.7, seed=0):
        """
        Used stratified random split to partitioned the original sample data into training and testing data used for model development
        Args:
            Split the region of interest using a stratified random approach, which use class label as basis for splitting
            roi: ee.FeatureCollection (original region of interest)
            class_prop: Class property (column) contain unique class ID
            tran_ratio: ratio for train-test split (usually 70% for training and 50% for testing)
        Return:
        ee.FeatureCollection, consist of training and testing data
        
        """
        #Define the unique class id using aggregate array
        classes = roi.aggregate_array(class_prop).distinct()
        #split the region of interest based on the class
        def split_class (c):
            subset = (roi.filter(ee.Filter.eq(class_prop, c))
                    .randomColumn('random', seed=seed))
            train = (subset.filter(ee.Filter.lt('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'training')))
            test = (subset.filter(ee.Filter.gte('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'testing')))
            return train.merge(test)
        #map the function for all the class
        split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
        #filter for training and testing
        train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
        test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
        print('Stratified Random Split Training Pixel Size:', train_fc.size().getInfo())
        print('Stratified Random Split Testing Pixel Size:', test_fc.size().getInfo())      
        #sample the image based stratified split data
        train_pix = image.sampleRegions(
                            collection=train_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
        test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16
        )
  
        return train_pix, test_pix

class Generate_LULC:
    def __init__(self):
        """
        Initialize the classification class to
        Perform classification to generate Land Cover Land Use Map. The parameters used in the classification should be the result of hyperparameter tuning
        """

    ############################# 1. Multiclass Classification ###########################
    def hard_classification(self, training_data, class_property, image, ntrees = 100, 
                                  v_split = None, min_leaf = 1, return_model = False,  seed=0):
        """
        Perform multiclass hard classification to generate land cover land use map
            Parameters:
            training data: ee.FeatureCollection, input sample data from feature extraction function (must contain pixel value)
            class_property (str): Column name contain land cover class id
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
        returns:
        ee.Image contain hard multiclass classification
        """
   # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used square root of total bands used in the classification
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        #Random Forest initialization
        clf = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed)
        model = clf.train(
            features=training_data,
            classProperty=class_property,
            inputProperties=image.bandNames()
        )
        #Implement the trained model to classify the whole imagery
        multiclass = image.classify(model)
        #Return model and classification, used for classification summary
        if return_model:
            return multiclass, model
        #if not needed, only return classification result
        else:
            return multiclass   
    ############################# 1. One-vs-rest (OVR) binary Classification ###########################
    def soft_classification(self, training_data, class_property, image, include_final_map=True,
                                ntrees = 100, v_split = None, min_leaf=1, seed=0, probability_scale = 100):
        """
        Implementation of one-vs-rest binary classification approach for multi-class land cover classification, similar to the work of
        Saah et al 2020. This function create probability layer stack for each land cover class. The final land cover map is created using
        maximum probability, via Argmax

        Parameters
            training_data (ee.FeatureCollection): The data which already have a pixel value from input covariates
            class_property (str): Column name contain land cover class id
            image (ee.Image): Image data
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
            probability scale = used to scaled up the probability layer

        Returns:
            ee.Image: Stacked probability bands + final classified map.
        """
        # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        
        # Get distinct classes ID from the training data. It should be noted that unique ID should be in integer, since 
        # float types tend to resulted in error during the band naming process 
        class_list = training_data.aggregate_array(class_property).distinct()
        
        #Define how to train one vs rest classification and map them all across the class
        def per_class(class_id):
            class_id = ee.Number(class_id)
            #Creating a binary features, 1 for a certain class and 0 for other (forest = 1, other = 0)
            binary_train = training_data.map(lambda ft: ft.set('binary', ee.Algorithms.If(
                            ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                )
                            ))
            #Build random forest classifiers, setting the outputmode to 'probability'. The probability mode will resulted in
            #one binary classification for each class. This give flexibility in modifying the final weight for the final land cover
            #multiprobability resulted in less flexibility in modifying the class weight
            #(the parameters required tuning)
            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed
            ).setOutputMode("PROBABILITY")
            #Train the model
            trained = classifier.train(
                features=binary_train,
                classProperty="binary",
                inputProperties=image.bandNames()
            )
            # Apply to the image and get the probability layer
            # (probability 1 represent the confidence of a pixel belonging to target class)
            prob_img = image.classify(trained).multiply(probability_scale).round().byte()
            #rename the bands
            #Ensure class_id is integer. 
            class_id_str = class_id.int().format()
            band_name = ee.String ('prob_').cat(class_id_str)

            return prob_img.rename(band_name)
        # Map over classes to get probability bands
        prob_imgs = class_list.map(per_class)
        prob_imgcol = ee.ImageCollection(prob_imgs)
        prob_stack = prob_imgcol.toBands()

        #if final map  is not needed, the functin will return prob bands only
        if not include_final_map:
            return prob_stack
        #final map creation using argmax
        print('Creating final classification using argmax')
        class_ids = ee.List(class_list)
        #find the mad prob in each band for each pixel
        #use an index image (0-based) indicating which class has highest probability
        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)

        #map the index to actual ID
        final_lc = max_prob_index.remap(ee.List.sequence(0, class_ids.size().subtract(1)),
                                        class_ids).rename('classification')
        #calculate confidence layer
        max_confidence = prob_stack.toArray().arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')
        #stack the final map and confidence
        stacked = prob_stack.addBands([final_lc, max_confidence])
        return stacked
        ############################# Feature importance ###########################
        #feature importance that can be used by hard or soft classification
    def get_feature_importance(self, trained_model):
        """
        Extract feature importance from a trained Random Forest model
        Parameters:
            trained_model: ee.Classifier - Trained Random Forest model (classification model must be enable)
        Returns:
            pandas.DataFrame containing model's feature importance
        """
        #Get model explanation from the trained model
        model_explanation = trained_model.explain().getInfo()
        #Extract feature importance
        if 'importance' not in model_explanation:
            raise ValueError("Feature importance not available in model explanation")
        importance_dict = model_explanation['importance']
        # Create DataFrame containing the importance
        importance_df = pd.DataFrame([
                {'Band': band, 'Importance': importance}
                for band, importance in importance_dict.items()
            ]).sort_values('Importance', ascending=False)
            # Normalize the value to percentage
        total_importance = importance_df['Importance'].sum()
        #get feature importance in the form of dataframe
        importance_df['Importance (%)'] = (importance_df['Importance'] / total_importance * 100).round(2)    
        # Reset index
        importance_df = importance_df.reset_index(drop=True)
        return importance_df 
    def evaluate_model(self, trained_model, test_data, class_property):
        """
        Perform model evaluation based on confusion matrix. This approach is similar to standard Remote Sensing accuracy assessment, but applied for trained model (ee.classifier)
        instead on classification result or map.
        Parameters:
            trained_model: ee.Classifier - Trained Random Forest model
            test_data: ee.FeatureCollection - Testing samples with pixel values
            class_property (str): Column name containing class labels
        
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        #Classify the testing data
        test_classified = test_data.classify(trained_model)
        
        #Get confusion matrix
        confusion_matrix = test_classified.errorMatrix(
            actual=class_property,
            predicted='classification'
        )
        #Get accuracy metrics
        #Producer accuracy / Recall (sensitivity)
        #User accuracy / Precision 
        # Get accuracy metrics from confusion matrix object
        overall_accuracy = confusion_matrix.accuracy().getInfo()
        kappa = confusion_matrix.kappa().getInfo()
        #here still used earth engine terminology
        producers_accuracy_ls = confusion_matrix.producersAccuracy().getInfo()
        #here still used earth engine terminology
        consumers_accuracy_ls = confusion_matrix.consumersAccuracy().getInfo()
        confusion_matrix_array = confusion_matrix.getInfo()

        #Flatten using numpy
        producers_accuracy = np.array(producers_accuracy_ls).flatten().tolist()
        consumers_accuracy = np.array(consumers_accuracy_ls).flatten().tolist()
        
        # Calculate F1 scores and geometric mean score
        #create and empty dict for storing the result
        #change the terminology from earth engine to machine learning
        f1_scores = []
        gmean_per_class = []
        for i in range(len(producers_accuracy)):
            recall_acc = producers_accuracy[i] #recall (machine learning terms)
            precision_acc = consumers_accuracy[i] #precision (machine learning terms)
            #calculate each class f1 score first
            if recall_acc + precision_acc > 0:
                f1 = 2 * (recall_acc * precision_acc) / (recall_acc + precision_acc)
            
            else:
                f1 = 0
            f1_scores.append(f1)
            
            #geometric mearn per class
            #equation: sqrt(recall * precision)
            if recall_acc > 0 and precision_acc > 0:
                gmean = np.sqrt(recall_acc * precision_acc)
            else:
                gmean = 0
            gmean_per_class.append(gmean)
        
        #Overall gmean
        #calculate the overall value of gmean from each class gmean
        #log transform are used for numerical stability (exp(mean(log(r1), log(r2), .... log(rn))))
        valid_gmeans = [g for g in gmean_per_class if g > 0]
        if valid_gmeans:
            overall_gmean = np.exp(np.mean(np.log(valid_gmeans)))
        else:
            overall_gmean = 0

        #Compile the accuracy metrics results
        accuracy_metrics = {
            'overall_accuracy': overall_accuracy,
            'kappa': kappa,
            'precision': consumers_accuracy,
            'recall': producers_accuracy,
            'confusion_matrix': confusion_matrix_array,
            'f1_scores': f1_scores,
            'gmean_per_class': gmean_per_class,
            'overall_gmean': overall_gmean
        }
        
        return accuracy_metrics
# Example of Dynamic Ensemble Classifiers for Land Use Land Cover Classification
## Introduction
Ensemble learning methods can be categorize into two main approaches, static and dynamic approach (H. Lu, Su, Hu, et al., 2022). The static method relied on single base classifiers, as discuss in the previous algorithms. The dynamic methods, or often refer as Dynamic Ensemble Selection (DES), select the base classifier on the fly according to new sample to be classified (Cruz et al., 2018; H. Lu, Su, Zheng, et al., 2022). The underlying rationale of DES is that each base classifier demonstrates expertise in specific local regions of the feature space (Cruz et al., 2020). Therefore, combining these "local experts" can enhance overall classification accuracy. 
## K-Nearest Neighbors Orcale
Ko et al (2008) introduced a novel DES-based selection process called K-Nearest Neighbor Oracles (KNORA). This method uses selection K nearest neighbor validation set, figure out which classifier correctly classified those neighbors and uses them as the ensemble for classifying the given pattern. Two KNORA framework were introduce in Ko et al (2008), KNORA-Union and KNORA-Eliminate. The KNORA-U employs a union rule, aggregating predictions from classifiers that correctly classify at least one instance within the local region (Elmi et al., 2023). In contrast, KNORA-E adopts a stricter criterion, selecting only classifiers that correctly classify all instances in the KNN set of the test sample. Implementation of DES for remote sensing image classification is found in Lu et al (2022) and Li et al (2022), with the application of detailed LULC mapping remains limited.
## Meta Learning for Dynamic Ensemble Selection
The selection of ensemble classifiers in dynamic ensemble selection (DES) is guided by estimates of the local accuracy or competence of base classifiers within a small region of the feature space (Cruz et al., 2014). In the KNORA framework, classifiers are selected based on their performance within the k-nearest neighbors of the query instance, with the aim of identifying the most competent classifiers in the local neighborhood. However, relying upon single criterion to measure level of competence of a base classifier is very prone to error (Cruz et al., 2014). To address this limitation, Cruz et al (2015) proposed a meta-learning framework for conducting the selection of the base classifiers. The meta learning perspective consider DES as a meta problem, in which different criteria is used to measure the level of competence for each base classifier. These criteria include various meta-features, such as measures of local accuracy in the region of competence, degree of confidence for the input sample, accuracy in the decision space, and etc (Cruz et al., 2015)
## Workflow
The implementation of Dynamic Ensemble Selection (DES) can be conducted using Deslib python library:
<br>
[![DESlib](https://img.shields.io/badge/DESlib-Docs-blue?style=flat-square&logo=python)](https://deslib.readthedocs.io/en/latest/)
<br>
The approach for this implementation, is similar to general machine learning workflow, with one notable difference. After splitting the original training data, the partitioned training data is further split into the training data for pool and the DES classiers. This approach is recommended by the authors of DESLIB, in which the DES classifiers, should be conducted by using different dataset from the pool classifiers. The following figure shows the workflow for implementation of dynamic ensemble classifier

<p align="center">
  <img src="DES_workflow.svg" width="400" alt="DES Workflow">
</p>

## Implementation for Land Use Land Cover Classification

below are the step by step guide for conducting DES implementation using DESLIB library, and jupyter notebook approach:
<br>
[![Open In NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](KNORAU.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/yourrepo/main?filepath=KNORAU.ipynb)

## Key Code Sections
#### 1. Importing Raster and Training Data
```python
#importing the raster data
raster_file = 'C:\Master of Remote Sensing\Python Code\EL_Research\Imagery\Multisource_Combine_clip.tif' #you could replace this with your own data
dataset = rasterio.open(raster_file)

# Inspect raster structure
L9_NS = dataset.read().transpose(1, 2, 0)  # Convert to (rows, cols, bands)
print(f"Raster shape: {L9_NS.shape}")

#adding the training data
sample = gpd.read_file('C:\Master of Remote Sensing\Python Code\Github_Repo_TassCap_Project\Data Source\TrainingSamples_rev21.shp') #Replace this with your own
sample.head()
# Class sample counts
print(sample['CLASS_NAME'].value_counts())
```

#### 2. Pixel Value Extraction
in this section, we are going to extract the pixels based on the training data. The core concept of machine learning classification can be summarize as follows:
Imagine your satellite image is like a giant puzzle where each tiny piece (pixel) needs to be identified. Here's how machine learning helps solve this puzzle:
  1. The Training Data 
  We mark sample areas on the image where we know the land cover type (forest, water, urban area, etc.)
  Each sample gives us:
  Features (X): The pixel's "fingerprint" (color values, elevation, etc.),
  Label (Y): The correct land cover type

  2. How the Computer Learns
  The algorithm looks for patterns like:
  "When pixels have high near-infrared values and medium greenness, they're usually forests"
  "When brightness is high and vegetation indices are low, it's likely urban areas"

  4. Making Predictions
  Once trained, the model can scan new pixels it has never seen and compare their features to patterns it learned

Here are the function for extracting the pixel for our model
```python
#Pixel Extraction function
def extract_pixels_from_shapefile(shapefile, raster):
    training_samples = []
    for index, row in shapefile.iterrows():
        geometry = [row['geometry']]
        id_class = row['New_ID'] #Uniqe Land Cover ID
        # Raster Masking based on input training data
        out_image, out_transform = mask(raster, geometry, crop=True)
        # Reshaping Raster (bands, height, width) 
        out_image = out_image.reshape(raster.count, -1).T  
        # Removing invalid pixels
        valid_pixels = out_image[~np.isnan(out_image).any(axis=1)]
        # Adding valid pixels ID
        for pixel in valid_pixels:
            training_samples.append((pixel, id_class))
    # Converts the tuple list into numpy array
    features = np.array([sample[0] for sample in training_samples])  # Extract pixel values (features)
    labels = np.array([sample[1] for sample in training_samples])    # Extract class labels
    return features, labels
```
Here we applied the function based on the raster and training data
```python
# Applying the previous function to get the pixel values
features, labels = extract_pixels_from_shapefile(sample, dataset)
# checking the shape of features and labels
print(features.shape)
print(labels.shape)
# masking nan values
nan_mask = np.isnan(features).any(axis=1) | np.isnan(labels)
features = features[~nan_mask]
labels = labels[~nan_mask]
# Counting each pixel samples
sample_new = pd.DataFrame({'class': labels})
print(sample_new['class'].value_counts())
```
#### 3. Training and Testing Split
This section provide example for partitioned the extracted pixels into traning and testing data. The proportion for each data vary according to literature. For this example, i used 80-20 proportion for training and testing data, since the partitioned training data will be further split into the training for the pool and DES classifiers
```python
from sklearn.model_selection import train_test_split
# Split data: 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
print('Testing data ', x_test.shape, y_test.shape)
# Split the aready partitioned training data for pool classifier (x_train), and DES classifier (x_dsel)
x_train, x_dsel, y_train, y_dsel = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3, random_state=42)
print("Training data for DES", x_dsel.shape, y_dsel.shape)
print('Training data for the pool classifiers', x_train.shape, y_train.shape)

#calculate the class distribution of pixels
un_class_train, class_count_train = np.unique(y_train, return_counts=True)
un_class_dsel, class_count_dsel = np.unique(y_dsel, return_counts=True)
un_class_test, class_count_test = np.unique(y_test, return_counts=True )
# Correct the union of unique classes for all datasets
all_classes = np.unique(np.concatenate([un_class_train, un_class_dsel, un_class_test]))

# Create the DataFrame for class pixel counts
class_pixel_count = pd.DataFrame({
    'Class': all_classes,
    'Pool Training Data': [
        class_count_train[un_class_train.tolist().index(cls)] if cls in un_class_train else 0
        for cls in all_classes
    ],
    'DES Training Data': [
        class_count_dsel[un_class_dsel.tolist().index(cls)] if cls in un_class_dsel else 0
        for cls in all_classes
    ],
    'Test Pixel': [
        class_count_test[un_class_test.tolist().index(cls)] if cls in un_class_test else 0
        for cls in all_classes
    ]
})
```
The pixel distribution based on this split are shown in the table below
### Dataset Distribution for Dynamic Ensemble Classification

| Class | Pool Training Data | DES Training Data | Test Pixels |
|-------|--------------------|-------------------|-------------|
| 0     | 622                | 267               | 222         |
| 1     | 85                 | 37                | 30          |
| 2     | 249                | 107               | 89          |
| 3     | 710                | 305               | 254         |
| 4     | 1947               | 834               | 695         |
| 5     | 85                 | 36                | 30          |
| 6     | 146                | 63                | 52          |
| 7     | 286                | 122               | 102         |
| 8     | 676                | 289               | 241         |
| 9     | 75                 | 32                | 27          |
| 10    | 1078               | 462               | 385         |
| 11    | 128                | 55                | 46          |
| 12    | 66                 | 28                | 24          |
| 13    | 1079               | 463               | 386         |
| 14    | 1013               | 434               | 362         |
| 15    | 636                | 273               | 227         |
| 16    | 615                | 264               | 220         |
| 17    | 904                | 387               | 323         |
| 18    | 321                | 138               | 115         |
| 19    | 1359               | 583               | 485         |
| 20    | 53                 | 22                | 19          |

**Total Samples:**
- Pool Classifiers Training: 12,133  
- DES Training: 5,201  
- Testing: 4,334  

## 4. Parameter Optimization
The success of machine learning algorithms often lied upon the utilization of several user defined parameters which needed to be optimized (or tuned) for a particular objective (Ramezan, 2022). In addition, tuning were crucial for bias reduction of a model predictive power, which tends to increase the model’s accuracy (Schratz et al., 2019). Various approach for parameter optimization has been proposed, ranging from grid search, randomized search, Bayesian optimization, and gradient optimization. The grid search approach is one of the widely used hyperparameter optimization technique (Perera et al., 2024). 
<br>
For the dynamic classifiers, the tuning process will be conducted for the base/pool classifiers, since optimization of the base classifier is crucial for the success of DES approach. In addition, the dynamic classifiers itself cannot implement a random grid search optimization as other ensemble classifier (Cruz et al., 2020). The pool classifiers could be homogenous or heterogenous, in both case diversity is expected (Britto et al., 2014). Therefore, for the pool classifiers, we used, Extremely Randomized Trees (ERT), Multilayer Perceptron (MLP), Extreme Gradient Boosting (XGB), and Histogram Gradient Boosting (HGBM) as the base classifiers for the DES framework. In addition to boosting and bagging, the introduction of neural network promote diversity in the pool classifiers. The diverse pool could improve the selection process, since different classifiers can made different mistake, which is beneficial in DES framework.

Below are the parameter grid i used for each pool classifiers used for DES classification. You could modified them according to your need
```python
#MLP Parameter Grid
mlp_param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128), (512, 256, 128)],
    'activation': ['tanh', 'relu'],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000, 1500, 2000],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.01]
}
mlp_model = MLPClassifier(solver='adam', early_stopping = True, random_state=42)
ERT_param = {
    'n_estimators' : [300, 500, 700, 900, 1100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
init_ET = ExtraTreesClassifier(random_state=42)
# Define the parameter grid for randomized search
xgb_param_grid = {
    'n_estimators': [500, 700, 900, 1100],
    'max_depth': [3, 6, 15, 20],
    'min_child_weight': [1, 3, 9, 13],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'subsample': [0.1, 0.5, 1.0],
    'colsample_bytree': [0.1, 0.5, 1.0]    
}
xgb_model = XGBClassifier(objective='multi:softprob', num_class = 21, eval_metric = 'mlogloss', random_state=42)
hgbm_param = {
    'learning_rate': [0.0001, 0.01, 0.1, 0.05, 0.2],
    'max_iter': [500, 700, 1000, 1500],  
    'max_depth': [None, 3, 9, 12, 15, 20],  
    'min_samples_leaf': [5, 10, 15, 20], 
    'max_bins': [32, 64, 255]
}
hgbm = HistGradientBoostingClassifier(random_state=42)
#List for storing the result

# Initialize a list to store results
results = []
import time
```
After finding the optimal parameters for each pool classifiers, we're going to explored some parameters in DES classiifers. As far as i know, GridSearchCV or RandomizedSearchCV are not compatible with Deslib library. This could change in the future, but for this part, we are going to utilized manual cross validation for finding optimal parameters. You could visit [DESlib documentation](https://deslib.readthedocs.io/en/latest/) for further reading regarding the parameter. There are several parameters for KNORA-E and METADES, for this example we are going to focus on two parameters
     <br>
     a. Number of k in k-Nearest Neighbors <br>
     The k refers to initial size of the region of competence used to estimate each classifier’s local accuracy. You could refer to this [article](https://doi.org/10.1016/j.patcog.2007.10.015) for further reading <br>
     b. Dynamic Frienemy Pruning (DFP) <br>
     If True, base classifiers that make identical predictions for every sample in the RoC are pruned before selection. You could refer to this [article](https://doi.org/10.1016/j.patcog.2017.06.030) for further reading <br>

Below are the python code for implementing manual cross validation for tuning the number of k
```python
#Function for Evaluating the number of k-neighbor in KNORAU
def get_models():
	models = dict()
	for n in range(2,22):
		models[str(n)] = KNORAE(k=n, DFP=True, voting='soft')
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, x_dsel, y_dsel, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
```
## 5. Implementing the Classification
For conducting the classification, we defined a list containing the best model acquired from parameter optimization. Then we fit the DES classifiers using the un-used partioned training data (x_dsel, y_dsel)
```python
#Provide the list for the pool classifiers
pool_clf = [best_xgb,  best_ert, best_hgb, best_mlp]
#K-Nearest Oracles-Eliminate
knorae = KNORAE(pool_classifiers=pool_clf, k=15, DFP=True, voting='soft', DSEL_perc = 0.1)
knorae.fit(x_dsel, y_dsel)
#METADES
metades = METADES(pool_classifiers=pool_clf, k=7, DFP=True, voting='soft', Hc=0.5, DSEL_perc = 0.1)
metades.fit(x_dsel, y_dsel)
```
## 6. Model Evaluation
Since my dataset contain imbalance data, we used several metrics for more comprehensive evaluation. The overall accuracy, F1-score, Geometric Mean, and Cross Entropy Loss will be calculated for the model, while taking account the class weight. The f1 score is the harmonic mean of precision and recall, and can be calculated using their average class value (Haibo He & Garcia, 2009). For G-mean, sensitivity and specificity are equal to true positive and true negative rates, respectively. In multiclass context, the G-mean can be calculated using average class value (Douzas et al., 2019).
<br>

$$F_1 = 2 \frac{Precision \times Recall}{Precision + Recall}$$

$$Gmean = \sqrt{Sensitivity \times Specificity}$$

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{n} \sum_{j=1}^{C} w_j \times y_{ij} \times \log(\hat{p}_{ij})$$

The CE Loss equation is shown in equation 5, in which this metric measure the dissimilarity between actual distribution and predicted probabilities (Yeung et al., 2022). The metric is computed by summing the negative logarithm of the predicted probabilities for the true classes log{{(p}_{ij})}, weighted by the refence class label (yij) and class weights (wj). The loss is then averaged over all samples. Intuitively, this metric penalizes predictions that assign low probabilities to the true classes, encouraging the model to output probabilities that closely align with the actual labels. Below are the implementation of model evaluation based on these metrics:

```python
from sklearn.utils.class_weight import compute_class_weight
# Classifiers and corresponding transformed test data
classifiers = {
    'KNORA-E': {'model': knorae, 'x_train': x_dsel, 'x_test': x_test},
    'METADES': {'model': metades, 'x_train': x_dsel, 'x_test': x_test},
}

# Store results
individual_accuracies = []

# Calculate class weights and evaluate classifiers
for name, clf_data in classifiers.items():
    clf = clf_data['model']
    x_train = clf_data['x_train']
    x_test = clf_data['x_test']

    # Calculate class weights based on RFECV-reduced training data
    unique_classes = np.unique(y_dsel)  # Assuming y_train is the same for all classifiers
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_dsel
    )
    class_weight_dict = dict(zip(unique_classes, weights))

    # Predict using the current classifier
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', sample_weight=[class_weight_dict[cls] for cls in y_test])
    w_logloss = log_loss(y_test, y_pred_proba, sample_weight=[class_weight_dict[cls] for cls in y_test])
    gmean = geometric_mean_score(y_test, y_pred, average='weighted', sample_weight=[class_weight_dict[cls] for cls in y_test])

    # Append results
    individual_accuracies.append((name, acc, f1, w_logloss, gmean))
    print(f'Accuracy of {name}: {acc:.4f}')
    print(f'F1 Score of {name}: {f1:.4f}')
    print(f'Log loss of {name}: {w_logloss:.4f}')
    print(f'Geometric Mean of {name}: {gmean:.4f}')

# Identify the worst-performing classifier
worst_classifier = min(individual_accuracies, key=lambda x: x[1])
print(f'\nWorst-performing classifier: {worst_classifier[0]} with accuracy: {worst_classifier[1]:.4f}')
print(f'F1 Score of worst-performing classifier: {worst_classifier[2]:.4f}')

# Convert results to DataFrame for easier visualization
columns = ['Classifier', 'Overall Accuracy', 'F1 Score', 'Log Loss', 'Geometric Mean']
result_df = pd.DataFrame(individual_accuracies, columns=columns)
print(result_df)
# Save the DataFrame
#result_df.to_csv('C:/Master of Remote Sensing/Python Code/HyperparameterTune_L9/DES_accuracy.csv', index=True)

```
## 7. Applied the model for the Classification
If the accuracy of each classifiers is sufficient, we can applied the model to conduct the classification for the whole dataset.
```python
import rasterio
from rasterio.transform import from_origin
#The classification funtion
def classify_raster(raster, model):
    # Reshape raster data into (pixels, bands)
    raster_data = raster.read().reshape((raster.count, -1)).T  # (pixels, bands)
    
    # Handle any invalid or missing data in the raster (e.g., NaN values)
    mask = np.isnan(raster_data).any(axis=1)  # Identify invalid pixels
    classified = np.full(raster_data.shape[0], -1, dtype=int)  # Initialize with a NoData value
    
    # Perform classification on valid pixels only
    classified[~mask] = model.predict(raster_data[~mask])
    
    # Reshape back to raster dimensions
    return classified.reshape((raster.height, raster.width))
def export_classified_map(classified_map, reference_raster, output_path, nodata_value=-1):
    # Copy metadata from reference raster
    meta = reference_raster.meta.copy()
    
    # Update metadata for classification output
    meta.update({
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,  # Single band
        "nodata": nodata_value,  # Define NoData value
        "compress": "lzw"
    })
    
    # Export the classified raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(classified_map.astype('int32'), 1)
```

## 8. Visualize the result
Aside from visualizing the result using GIS software, we could visualized the result using matplotlib, here are the example implementation:
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredText
# 1. Define legend and colors for classes
legend_21 = ['Settlement Build-up', 'Non-settlement Build-up', 'Volcanic Rock and Sand', 'Lowland Forest', 
             'Upland Forest', 'Mangrove Forest', 'Marine Sand', 'Herbs and Grassland', 'Production Forest', 
             'Aquaculture', 'Dryland Agriculture', 'Natural Fallow Land', 'Man-made Fallow Land', 'Rubber and Other Hardwood Plantation',
             'Ocean Waters','Oil Palm Plantatation','Tea Plantation', 'Bushes and Shrubs', 'Rivers', 'Wetland Agriculture',
             'Natural/Semi Natural Freshwaterbody']

legend_colors = ['#FF0000', '#e97421', '#7b3531', '#2e8b57', '#228b22', '#7fffd4', '#fdd9b5', 
                 '#c1e1c1', '#9acd32', '#ccccff', '#f5ff00', '#b5a642', '#cc8899', '#32cd32', 
                  '#1f52dc', '#808000', '#98d97d','#008080', '#87ceeb', '#afb325', '#0f3c5e']
#2. define the Original Raster data, and LULC data
rasters = {
    'Landsat 9': r'C:\Master of Remote Sensing\Python Code\Github_Repo_TassCap_Project\Data Source\Landsat9_Final_Addtwi.tif',
    'KNORA-E':   r"C:/Master of Remote Sensing/Python Code/EL_Research/NewOutput/New_KNORA-E.tif",
    'METADES': r"C:/Master of Remote Sensing/Python Code/EL_Research/NewOutput/New_METADES.tif"
}

cmap  = ListedColormap(legend_colors, name="lulc21") # Build a discrete Colormap from the predefined hex color code
bounds = np.arange(len(legend_colors)+1) - 0.5       #Bound the pixel values, with no gradient or shades
norm   = BoundaryNorm(bounds, cmap.N) #Asigning one color one id

# 3.  Creating Layout Plot
mpl.rcParams.update({
    'font.family'     : 'Times New Roman',
})
n_maps = len(rasters) #Images that will be plot
n_rows = 1  #number of row 
n_cols = int(np.ceil(n_maps / n_rows)) # minimum cols to fit everything

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows), #Canvas size in inches
                         sharex=True, sharey=True)
axes = axes.ravel() #Flatten the 2D Numpy Array for easier looping
# 4.  Create a loop through each raster
for ax, (title, path) in zip(axes, rasters.items()):
    with rasterio.open(path) as src:
        if src.count > 1 and title.startswith("Landsat"):   # multiband FCC
            # --- READ BANDS 5,4,3 (1‑based) ---
            nir  = src.read(5).astype("float32")
            red  = src.read(4).astype("float32")
            green= src.read(3).astype("float32")
            rgb  = np.stack([nir, red, green], axis=-1)

            # --- SIMPLE PERCENTILE STRETCH (2‑98%) ---
            p2, p98 = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
            rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

            ax.imshow(rgb)
        else:                                              # categorical LULC
            data = src.read(1)
            nodata_vals = {src.nodata, -1}
            nodata_vals = {v for v in nodata_vals if v is not None}
            mask = np.isin(data, list(nodata_vals))
            data = np.where(mask, np.nan, data)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_title(title,fontsize=16, fontweight= 'bold')
    ax.axis('off')

# hide any empty subplot (there will be one in a 2×3 grid)
for extra_ax in axes[n_maps:]:
    extra_ax.axis('off')

# 5.  Adding Legend for LULC Map
handles = [mpatches.Patch(color=c, label=lab)
           for lab, c in zip(legend_21, legend_colors)]

fig.legend(
    handles=handles,
    loc='center left', bbox_to_anchor=(0.97, 0.5),
    ncol=1,
    fontsize=13,
    frameon=False,
    handlelength=1.5,
    handleheight=1
)
plt.tight_layout(pad=1)
plt.show()
```
![image](https://github.com/user-attachments/assets/3e683352-4db4-4320-bf0b-51d7d233253b)

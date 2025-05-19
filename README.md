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

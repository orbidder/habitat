##SGNP Random Forest Land Classification##
#Owen R. Bidder#
#orbidder@berkeley.edu#

import pandas as pd
import rasterio
import fiona
from sklearn.model_selection import train_test_split
import os
import skimage.io as io

os.chdir('C:\\puma_db\env_data\Owen Habitat Map')

# quickly plot first band
from rasterio.plot import show

raster = rasterio.open('all_stack.bil')
show((raster, 2))
raster.close()

aspect = pd.Series()
dem = pd.Series()
rough = pd.Series()
slope = pd.Series()
tri = pd.Series()
ndvi = pd.Series()
flowdir = pd.Series()

person = pd.Series()
habitat = pd.Series()
coords = pd.Series()

# Read input shapefile with fiona and iterate over each feature

with fiona.open('train_points_combi.shp') as shp:
    for feature in shp:
        row = feature['properties']['id']
        per = feature['properties']['person']
        hab = feature['properties']['habitat']
        coo = feature['geometry']['coordinates']
        # Read pixel value at the given coordinates using Rasterio
        # NB: `sample()` returns an iterable of ndarrays.
        with rasterio.open('all_stack.bil') as src:
            val_aspect = [v for v in src.sample([coo])][0][0]
            val_dem = [v for v in src.sample([coo])][0][1]
            val_rough = [v for v in src.sample([coo])][0][2]
            val_slope = [v for v in src.sample([coo])][0][3]
            val_tri = [v for v in src.sample([coo])][0][4]
            val_ndvi = [v for v in src.sample([coo])][0][5]
            val_flowdir = [v for v in src.sample([coo])][0][6]
            # Update the pandas serie accordingly
            aspect.loc[row] = val_aspect
            dem.loc[row] = val_dem
            rough.loc[row] = val_rough
            slope.loc[row] = val_slope
            tri.loc[row] = val_tri
            ndvi.loc[row] = val_ndvi
            flowdir.loc[row] = val_flowdir
            person.loc[row] = per
            habitat.loc[row] = hab
            coords.loc[row] = coo

results = pd.concat((person.rename('person'), habitat.rename('habitat'), coords.rename('coords'),
                     aspect.rename('aspect'), dem.rename('dem'), rough.rename('rough'), slope.rename('slope'),
                     tri.rename('tri'), ndvi.rename('ndvi'), flowdir.rename('flowdir')), axis=1)
src.close()
# Write records into a CSV file
results.to_csv('annotated_points.csv')

# split results in to train and test
# set x (accl data) and y (classes)
x = results.iloc[:, 3:10]
y = results['habitat']

# Split accl_data to testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

# check training and testing have the same classes
set(y_train) == set(y_test)

## Import the Classifier.
from sklearn.ensemble import RandomForestClassifier

## Instantiate the model with 5 neighbors.
rf = RandomForestClassifier(n_estimators=600)
## Fit the model on the training data.
rf.fit(X_train, y_train.values.ravel())

## See how the model performs on the test data.
rf.score(X_test, y_test)

##Get Array of all predictions
y_pred = rf.predict(X_test)

##Calculate confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
cohen_kappa_score(y_test, y_pred)

# Get Feature importance
import pandas as pd

feature_imp = pd.Series(rf.feature_importances_, index=X_train.columns.values).sort_values(ascending=False)
feature_imp  # could drop flowdir and aspect?

##Plot a confusion matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##Get confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
class_names = y_test.unique()

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Predict habitat class in original raster

# Read Data
img_ds = io.imread('all_stack_wgs.tif')
img = np.array(img_ds, dtype='int16')

# Classification of array and save as image (7 refers to the number of bands in the stack)
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :7].reshape(new_shape)

# Predict over all cells and return to image
class_prediction = rf.predict(img_as_array)
class_prediction = class_prediction.reshape(img[:, :, 0].shape)

# now export your classificaiton (save as tif not bil)
pred_int = np.where(class_prediction == "Canyon", 1, np.where(class_prediction == "Meadow", 2, np.where(class_prediction == "Plain", 3, 999)))
io.imsave('predicted_habitat.tif', pred_int)



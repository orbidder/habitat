##SGNP Random Forest Land Classification##
#Owen R. Bidder#
#orbidder@berkeley.edu#

import pandas as pd
import rasterio
import fiona
import os

os.chdir('C:\\puma_db\env_data\Owen Habitat Map')

values = pd.Series()
person = pd.Series()
habitat = pd.Series()
coords = pd.Series()

# Read input shapefile with fiona and iterate over each feature
with fiona.open('habitat_points.shp') as shp:
    for feature in shp:
        row = feature['properties']['id']
        per = feature['properties']['person']
        hab = feature['properties']['habitat']
        coo = feature['geometry']['coordinates']
        # Read pixel value at the given coordinates using Rasterio
        # NB: `sample()` returns an iterable of ndarrays.
        with rasterio.open('all_stack.bil') as src:
            value = [v for v in src.sample([coo])][0][0]
        # Update the pandas serie accordingly
        values.loc[row] = value
        person.loc[row] = per
        habitat.loc[row] = hab
        coords.loc[row] = coo

results = pd.concat((person, habitat, coords, values), axis=1)
# print(results)
# Write records into a CSV file
#values.to_csv('annotated_points.csv')

# #split to training and testing set
#
# #Run RF model
#
# #Check accuracy
#
# #predict out to entire SGNP

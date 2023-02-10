import pandas as pd
import numpy as np
import joblib
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot

import sys
sys.path.append('/home/sr_user/alexia/rfv2/py_module/')
from data_display import display_predictions

images = "/work/users/flair-ign/test/D085_2019/Z9_NF/img/IMG_076387.tif"




# Load the test data

test_data = pd.read_csv("/work/users/flair-ign/alexia/dfs_test/IMG_061713.csv")
testd = test_data.drop(['i_rows','j_cols'], axis=1)
print(testd)



# Load the saved model
model = joblib.load("rfv2.sav")

# Make predictions
predictions = model.predict(testd)

# Save the predictions into a DataFrame
dfpred = pd.DataFrame(predictions, columns=['mask'])
print(dfpred)
# Concatenate testd and dfpred
result = pd.concat([testd, dfpred], axis=1)

# Print the result
print("Result:", result)
# save the Result
result.to_csv('result.csv', index=False)



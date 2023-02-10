import pandas as pd
import numpy as np
import joblib

# Load the test data
test_data = pd.read_csv("/work/users/flair-ign/alexia/dfs_test/IMG_061713.csv")
testd = test_data.drop(['i_rows','j_cols'], axis=1)


print(testd)



# Load the saved model
model = joblib.load("rfv2.sav")

# Make predictions
predictions = model.predict(testd)

# Print the result
print("Predictions:", predictions)

import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from SiPANN.import_nn import ImportLR

##### THESE ARE ALL THE THINGS YOU SHOULD NEED TO EDIT ##########
# Load Input and Output data here (replace np.zeros)
INPUT = np.zeros((100, 5))
OUTPUT = np.zeros((100, 3))
# Put the filename here
filename = "test"


# Set up and test the regression model
model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=4)),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)
model = model.fit(INPUT, np.real(OUTPUT))
MSE_REGRESION = mean_squared_error(model.predict(INPUT), np.real(OUTPUT))
print("Multivariate linear regression MSE: {:e}".format(MSE_REGRESION))

# Save so our ImportLR can use it
d = {
    "coef_": model.named_steps["linear"].coef_,
    "degree_": model.named_steps["poly"].degree,
}

n_features = model.named_steps["poly"].n_input_features_
n_out = 1 if len(d["coef_"].shape) == 1 else d["coef_"].shape[0]
d["s_data"] = (n_features, n_out)
pickle.dump(d, open(f"{filename}.pkl", "wb"))

## Check to make sure we're getting the same outputs for sklearn and Import LR
mine = ImportLR(f"{filename}.pkl")
print(np.allclose(model.predict(INPUT), mine.predict(INPUT)))

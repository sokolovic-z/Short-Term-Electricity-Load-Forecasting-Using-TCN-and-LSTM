# import libraries
import numpy as np
import pickle

from darts.timeseries import concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# load the raw dataset
ts = ElectricityConsumptionZurichDataset().load()
ts = ts.astype(np.float32)

# target & covariates selection
target = ts['Value_NE5'] # electricity consumption
covs = ts.drop_columns(['Value_NE5', 'Value_NE7']) # temp, humidity, windspeed, etc.

# splitting data into 70/15/15 ratio
_train, test = train_test_split(target, test_size=0.15)
train, val = train_test_split(_train, test_size=0.15 / 0.85)
_train_covs, test_covs = train_test_split(covs, test_size=0.15)
train_covs, val_covs = train_test_split(_train_covs, test_size=0.15 / 0.85)

# scaling target
scaler_target = Scaler(StandardScaler())
train_scaled = scaler_target.fit_transform(train)
val_scaled = scaler_target.transform(val)
test_scaled = scaler_target.transform(test)
# scaling covariates
scaler_covs = Scaler(StandardScaler())    
train_covs_scaled = scaler_covs.fit_transform(train_covs)
val_covs_scaled = scaler_covs.transform(val_covs)
test_covs_scaled = scaler_covs.transform(test_covs)
# concatenate for the full scaled series; we can feed this to model.fit()/predict() as Darts will extract the required covs for you
covs_scaled = concatenate([train_covs_scaled, val_covs_scaled, test_covs_scaled])

# save processed data
data = {
    "train_scaled": train_scaled,
    "val_scaled": val_scaled,
    "test_scaled": test_scaled,
    "covs_scaled": covs_scaled,
    "scaler_target": scaler_target,
    "scaler_covs": scaler_covs
}
with open("data/electricity_processed.pkl", "wb") as f:
    pickle.dump(data, f)

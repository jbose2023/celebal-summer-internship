import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save 'Id' for future use
train_ID = train['Id']
test_ID = test['Id']

# Drop 'Id' since it's not predictive
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# Store target and drop from train
y = train["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)

# Combine for preprocessing
all_data = pd.concat([train, test], axis=0)
print("Combined data shape:", all_data.shape)

# Missing values - Fill categorical with mode, numerical with median
cat_cols = all_data.select_dtypes(include=['object']).columns
num_cols = all_data.select_dtypes(exclude=['object']).columns

for col in cat_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

for col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# Log-transform skewed numeric features
skewed_feats = all_data[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
skewness = skewed_feats[abs(skewed_feats) > 0.75]
print(f"Skewed features: {len(skewness)}")

for feat in skewness.index:
    all_data[feat] = np.log1p(all_data[feat])

# One-hot encode categorical features
all_data = pd.get_dummies(all_data)
print("Data shape after encoding:", all_data.shape)

# Scale the features
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)

# Split back to train and test
X_train = all_data_scaled[:len(y)]
X_test = all_data_scaled[len(y):]

# Export processed data (optional)
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y.npy", y)

print("Preprocessing complete. Ready for modeling.")


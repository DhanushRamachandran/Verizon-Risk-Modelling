# data sampling
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
X = pd.read_csv("data/Preprocessed_Features.csv")
y = pd.read_csv(r"data/Verizon_Data.csv")[["delinquency_90_days"]]

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_dict = {
    "Performing/paid off":1,
    "90+ delinquent/default":0
}   

full_df = pd.concat([X, y], axis=1)
train_data = pd.concat([x_train, y_train], axis=1)
train_data["delinquency_90_days"] = train_data["delinquency_90_days"].apply(lambda x: label_dict[x])
y_test["delinquency_90_days"] = y_test["delinquency_90_days"].apply(lambda x: label_dict[x])

def up_scale_min_downscale_maj():
    # Separate majority and minority classes
    majority_class = train_data[train_data["delinquency_90_days"] == 1]  # assume 1 is majority
    minority_class = train_data[train_data["delinquency_90_days"] == 0]  # assume 0 is minority

    target_size = 3500  # target samples per class

    # Downsample majority to target size
    majority_resampled = resample(
        majority_class,
        replace=False,
        n_samples=target_size,
        random_state=42
)

# Upsample minority to target size
    minority_resampled = resample(
        minority_class,
        replace=True,
        n_samples=target_size,
        random_state=42
)

# Combine both
    train_balanced = pd.concat([majority_resampled, minority_resampled])

# Shuffle the training set
    train_balanced = train_balanced.sample(frac=1, random_state=42)

# Split back into X and y
    X_train_balanced = train_balanced.drop(columns=["delinquency_90_days"])
    y_train_balanced = train_balanced["delinquency_90_days"]
    return X_train_balanced, y_train_balanced   

# def down_scale_maj_alone()
    

# def upscale_min_alone()


if __name__=="__main__":
    X_train_balanced, y_train_balanced = up_scale_min_downscale_maj()
    X_train_balanced.to_csv("data/X_train.csv", index=False)
    y_train_balanced.to_csv("data/y_train.csv", index=False)
    x_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

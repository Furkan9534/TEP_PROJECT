
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(
    r"C:\Users\Furkan-KALKAN\Desktop\TEP_PROJECT\data\processed"
)

train = pd.read_csv(OUTPUT_DIR / "tep_train.csv")
test = pd.read_csv(OUTPUT_DIR / "tep_test.csv")

print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)

print("\nTraining condition counts:")
print(train["condition"].value_counts())

print("\nTesting condition counts:")
print(test["condition"].value_counts())

print("\nFault samples in test data:")
print(test.groupby("fault_id").size())

print("\nMissing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())

print("\nFirst rows:")
print(test.head())



print("\nTraining samples per fault ID:")
print(train.groupby("fault_id").size().sort_index())

print("\nTraining condition vs fault ID:")
print(pd.crosstab(train["fault_id"], train["condition"]))

print("\nTest condition vs fault ID:")
print(pd.crosstab(test["fault_id"], test["condition"]))


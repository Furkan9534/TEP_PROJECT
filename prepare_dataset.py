
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\Furkan-KALKAN\Desktop\TEP_PROJECT/data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tep_file(path):
    data = np.loadtxt(path)

    # d00.dat is stored as variables x samples
    if data.shape[0] == 52 and data.shape[1] > 52:
        data = data.T

    if data.ndim != 2 or data.shape[1] != 52:
        raise ValueError(f"Unexpected shape in {path.name}: {data.shape}")

    return data

columns = [f"X{i:02d}" for i in range(1, 53)]
train_frames = []
test_frames = []

# Training data
for scenario_id in range(22):
    path = DATA_DIR / f"d{scenario_id:02d}.dat"
    data = load_tep_file(path)

    frame = pd.DataFrame(data, columns=columns)
    frame["fault_id"] = scenario_id
    frame["scenario_id"] = scenario_id
    frame["condition"] = "normal" if scenario_id == 0 else "fault"
    train_frames.append(frame)

# Test data
for scenario_id in range(22):
    path = DATA_DIR / f"d{scenario_id:02d}_te.dat"
    data = load_tep_file(path)

    frame = pd.DataFrame(data, columns=columns)
    frame["scenario_id"] = scenario_id

    if scenario_id == 0:
        frame["fault_id"] = 0
    else:
        # Standard TEP test protocol:
        # first 160 samples are normal, then the fault is introduced
        frame["fault_id"] = scenario_id
        frame.loc[:159, "fault_id"] = 0

    frame["condition"] = np.where(
        frame["fault_id"] == 0, "normal", "fault"
    )
    test_frames.append(frame)

train = pd.concat(train_frames, ignore_index=True)
test = pd.concat(test_frames, ignore_index=True)

train.to_csv(OUTPUT_DIR / "tep_train.csv", index=False)
test.to_csv(OUTPUT_DIR / "tep_test.csv", index=False)

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nTest labels:")
print(test["fault_id"].value_counts().sort_index())

print("\nTest condition counts:")
print(test["condition"].value_counts())

print("\nSaved files to:", OUTPUT_DIR)

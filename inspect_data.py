
from pathlib import Path
import numpy as np

data_dir = Path(r"data")

for file in sorted(data_dir.glob("*.dat")):
    data = np.loadtxt(file)

    if data.shape[0] == 52 and data.shape[1] > 52:
        data = data.T
        print(f"{file.name}: transpoze edildi")

    if data.shape[1] != 52:
        print(f"UYARI: {file.name}, beklenmeyen boyut: {data.shape}")
    else:
        print(f"{file.name}: {data.shape[0]} örnek, {data.shape[1]} değişken")

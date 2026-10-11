
from pathlib import Path
import numpy as np

folder = Path(r"C:\Users\Furkan-KALKAN\Desktop\TEP_PROJECT\data")

for name in ["d00.dat", "d00_te.dat", "d01.dat"]:
    path = folder / name
    data = np.loadtxt(path)

    print(f"\nDosya: {name}")
    print("Boyut:", data.shape)
    print("İlk satırdaki değer sayısı:", len(path.open().readline().split()))
    print("İlk 5 değer:", data[0, :5])

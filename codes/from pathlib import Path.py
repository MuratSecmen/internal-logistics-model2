from pathlib import Path

INPUT_DIR = Path(r"C:\Users\Asus\Documents\GitHub\internal-logistics-model2\internal-logistics-model2\inputs")

print("Klasör var mı?", INPUT_DIR.exists())
print("Klasördeki dosyalar:")

for f in INPUT_DIR.iterdir():
    print(f.name)
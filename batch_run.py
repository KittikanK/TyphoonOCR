import os
import subprocess

DATA_DIR = "./data"
OUT_DIR = "./xls"
DEBUG_DIR = "./debug"
WORDING = "./template/master-data-wording.txt"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.lower().endswith(".pdf"):
        continue
    pdf_path = os.path.join(DATA_DIR, fname)
    out_path = os.path.join(OUT_DIR, fname.replace(".pdf", ".xlsx"))

    cmd = [
        "python", "typhoon_ocr_to_excel_v2.py",
        "--pdf", pdf_path,
        "--out", out_path,
        "--wording", WORDING,
        "--threshold", "70",
        "--debug",
        "--debugdir", DEBUG_DIR,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


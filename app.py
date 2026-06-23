# https://gogstbill.com/wp-content/uploads/2025/07/A4-S1-GST-Invoice-Format_page-0001-scaled.jpg

from rapidocr import RapidOCR
import requests
import tempfile
import zipfile
import os
import shutil
import time
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

engine = RapidOCR()

INPUT_URL = "https://gogstbill.com/wp-content/uploads/2025/07/A4-S1-GST-Invoice-Format_page-0001-scaled.jpg"


WORK_DIR = Path("./temp")
WORK_DIR.mkdir(exist_ok=True)

all_results = []


def ocr_image(image_path):
    start = time.perf_counter()

    result = engine(str(image_path))

    elapsed = time.perf_counter() - start

    texts = []

    if hasattr(result, "txts"):
        texts = list(result.txts)
    else:
        for item in result:
            texts.append(item[1])

    return {
        "file": str(image_path),
        "text": "\n".join(texts),
        "time": elapsed,
    }


print(f"Downloading: {INPUT_URL}")

response = requests.get(INPUT_URL, stream=True)
response.raise_for_status()

filename = INPUT_URL.split("/")[-1]

download_path = WORK_DIR / filename

with open(download_path, "wb") as f:
    for chunk in response.iter_content(8192):
        f.write(chunk)

print("Download complete")


files_to_process = []

suffix = download_path.suffix.lower()

# ---------------------------
# IMAGE
# ---------------------------
if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
    files_to_process.append(download_path)

# ---------------------------
# PDF
# ---------------------------
elif suffix == ".pdf":

    pdf_pages = convert_from_path(download_path)

    for i, page in enumerate(pdf_pages):
        page_path = WORK_DIR / f"page_{i+1}.png"

        page.save(page_path)

        files_to_process.append(page_path)

# ---------------------------
# ZIP
# ---------------------------
elif suffix == ".zip":

    extract_dir = WORK_DIR / "unzipped"

    with zipfile.ZipFile(download_path) as z:
        z.extractall(extract_dir)

    for root, _, files in os.walk(extract_dir):
        for file in files:

            ext = Path(file).suffix.lower()

            if ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:
                files_to_process.append(
                    Path(root) / file
                )

# ---------------------------
# OCR
# ---------------------------
total_start = time.perf_counter()

for idx, file_path in enumerate(files_to_process, start=1):

    result = ocr_image(file_path)

    all_results.append(result)

    print("\n" + "=" * 80)
    print(f"[{idx}/{len(files_to_process)}]")
    print(f"FILE: {file_path.name}")
    print(f"OCR TIME: {result['time']:.3f}s")
    print("=" * 80)

    print(result["text"][:1000])

total_end = time.perf_counter()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Files Processed: {len(all_results)}")
print(
    f"Total Time: "
    f"{total_end - total_start:.3f}s"
)

# Full OCR results
for item in all_results:

    print("\n" + "=" * 80)
    print(item["file"])
    print("=" * 80)

    print(item["text"])

shutil.rmtree(WORK_DIR, ignore_errors=True)
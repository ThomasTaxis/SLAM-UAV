import os
import requests
import zipfile

# 🔧 Replace this with your actual file ID
FILE_ID = "1u1hXlFjXtMvs0J8t9hmbG5Q56Xi3vu74"

ZIP_PATH = "data/dataset.zip"
EXTRACT_TO = "data/data_sequence/"

def download_from_google_drive(file_id, dest_path):
    print("📥 Downloading from Google Drive...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    
    save_response_content(response, dest_path)
    print(f"✅ Downloaded to {dest_path}")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def extract_zip(zip_path, extract_to):
    print(f"📂 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        download_from_google_drive(FILE_ID, ZIP_PATH)
    else:
        print("⚠️ ZIP file already exists. Skipping download.")

    if not os.path.exists(EXTRACT_TO):
        extract_zip(ZIP_PATH, EXTRACT_TO)
    else:
        print("⚠️ Extracted folder already exists. Skipping extraction.")

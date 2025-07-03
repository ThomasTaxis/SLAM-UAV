import requests
import zipfile
import os

def download_and_extract(url, dest_folder, zip_filename='dataset.zip'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    zip_path = os.path.join(dest_folder, zip_filename)

    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    print("Cleaning up...")
    os.remove(zip_path)
    print("Download complete!")

if __name__ == "__main__":
    # Replace with your actual download URL
    gdrive_url = "https://drive.google.com/file/d/1u1hXlFjXtMvs0J8t9hmbG5Q56Xi3vu74"
    download_and_extract(gdrive_url, dest_folder="data")

import gdown
import zipfile
import os

def download_and_extract(file_id, dest_folder, zip_filename='dataset.zip'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    zip_path = os.path.join(dest_folder, zip_filename)

    # Build the direct download URL for gdown
    url = f'https://drive.google.com/uc?id={file_id}'

    print("Downloading dataset with gdown...")
    gdown.download(url, zip_path, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    print("Cleaning up...")
    os.remove(zip_path)
    print("Download complete!")

if __name__ == "__main__":
    file_id = "1u1hXlFjXtMvs0J8t9hmbG5Q56Xi3vu74"  # Your Google Drive file ID
    download_and_extract(file_id, dest_folder="data")

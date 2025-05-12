import os
import requests
from zipfile import ZipFile
from tqdm import tqdm
import argparse

def download_and_extract(url, save_dir):
    filename = os.path.join(save_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                f.write(data)
                bar.update(len(data))
    else:
        print(f"{filename} already downloaded.")

    # Extract zip
    print(f"Extracting {filename}...")
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(save_dir)

    # Delete the zip file after extraction
    print(f"Deleting {filename}...")
    os.remove(filename)

def main():
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to the directory where the dataset will be saved."
    )
    parser.add_argument(
        "--year",
        type=str,
        required=False,
        choices=["2015", "2017"],
        default="2017",
        help="COCO years."
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir + '/images', exist_ok=True)

    # URLs for COCO 2017 dataset
    coco_urls = {
        "train_images": "http://images.cocodataset.org/zips/train" + args.year + ".zip",
        "val_images": "http://images.cocodataset.org/zips/val" + args.year + ".zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval" + args.year + ".zip",
    }
    data_dirs = ['/images', '/images', '']
    for i, (key, url) in enumerate(coco_urls.items()):
        download_and_extract(url, args.target_dir + data_dirs[i])


if __name__ == "__main__":
    main()

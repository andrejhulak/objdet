import os
import shutil
from pathlib import Path

def merge_datasets(dataset1_path, dataset2_path, output_dataset_path):
  output_images_path = os.path.join(output_dataset_path, "images")
  output_labels_path = os.path.join(output_dataset_path, "labels")
  
  os.makedirs(output_images_path, exist_ok=True)
  os.makedirs(output_labels_path, exist_ok=True)
  
  dataset1_images = os.path.join(dataset1_path, "images")
  dataset1_labels = os.path.join(dataset1_path, "labels")
  
  if os.path.exists(dataset1_images):
    for filename in os.listdir(dataset1_images):
      src_path = os.path.join(dataset1_images, filename)
      dst_path = os.path.join(output_images_path, filename)
      shutil.copy2(src_path, dst_path)
  else:
    print(f"Warning: {dataset1_images} not found")
  
  if os.path.exists(dataset1_labels):
    for filename in os.listdir(dataset1_labels):
      src_path = os.path.join(dataset1_labels, filename)
      dst_path = os.path.join(output_labels_path, filename)
      shutil.copy2(src_path, dst_path)
  else:
    print(f"Warning: {dataset1_labels} not found")
  
  dataset2_images = os.path.join(dataset2_path, "images")
  dataset2_labels = os.path.join(dataset2_path, "labels")
  
  if os.path.exists(dataset2_images):
    for filename in os.listdir(dataset2_images):
      src_path = os.path.join(dataset2_images, filename)
      new_filename = f"ds3_{filename}"
      dst_path = os.path.join(output_images_path, new_filename)
      shutil.copy2(src_path, dst_path)
  else:
    print(f"Warning: {dataset2_images} not found")
  
  if os.path.exists(dataset2_labels):
    for filename in os.listdir(dataset2_labels):
      src_path = os.path.join(dataset2_labels, filename)
      new_filename = f"ds3_{filename}"
      dst_path = os.path.join(output_labels_path, new_filename)
      shutil.copy2(src_path, dst_path)
  else:
    print(f"Warning: {dataset2_labels} not found")
  
  print(f"\n{'='*50}")
  print("MERGE COMPLETE!")
  print(f"{'='*50}")
  
  if os.path.exists(output_images_path):
    img_files = os.listdir(output_images_path)
    print(f"Total images: {len(img_files)}")
    ds1_imgs = [f for f in img_files if not f.startswith('ds2_')]
    ds2_imgs = [f for f in img_files if f.startswith('ds2_')]
    print(f"  - From dataset 1: {len(ds1_imgs)}")
    print(f"  - From dataset 2: {len(ds2_imgs)}")
  
  if os.path.exists(output_labels_path):
    lbl_files = os.listdir(output_labels_path)
    print(f"Total labels: {len(lbl_files)}")
    ds1_lbls = [f for f in lbl_files if not f.startswith('ds2_')]
    ds2_lbls = [f for f in lbl_files if f.startswith('ds2_')]
    print(f"  - From dataset 1: {len(ds1_lbls)}")
    print(f"  - From dataset 2: {len(ds2_lbls)}")
  
  print(f"\nMerged dataset saved to: {output_dataset_path}")

DATASET_1_PATH = "data/third ds"
DATASET_2_PATH = "data/big ds"
OUTPUT_PATH = "data/arma"

if __name__ == "__main__":
  print("Merging two datasets...")
  merge_datasets(DATASET_1_PATH, DATASET_2_PATH, OUTPUT_PATH)

import torch
from torch import nn

device= "cuda" if torch.cuda.is_available() else "cpu"

device

torch.__version__

import pathlib
data_dir = pathlib.Path("../data")

import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_data=datasets.Food101(root=data_dir,
                            split="train",
                            download=True)
test_data=datasets.Food101(root=data_dir,
                            split="test",
                            download=True)

train_data

class_names= train_data.classes
class_names

print(class_names[train_data[1][1]])

train_data[1][0]

import random
data_path = data_dir/"food-101"/ "images"
target_classes= ["cup_cakes","chicken_wings","pizza", "waffles", "ramen"]

amount_to_get = 0.2

def get_subset(image_path=data_path,
               data_splits=["train","test"],
               target_classes= ["cup_cakes","chicken_wings","pizza", "waffles", "ramen"],
               amount=0.1,
               seed=33):
  random.seed(33)
  label_splits={}

  for data_split in data_splits:
    print(f"[INFO] Creating image split for: {data_split}...")
    label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"
    with open(label_path, "r") as f:
      labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

    number_to_sample = round(amount * len(labels))
    print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
    sampled_images = random.sample(labels, k=number_to_sample)

    image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
    label_splits[data_split] = image_paths
  return label_splits
        
label_splits = get_subset(amount=amount_to_get)
label_splits["train"][:10]

"""#Move training and testing images to dedicated folders"""

target_dir_name=f"../data/5_Classes_of_Food101_{str(int(amount_to_get*100))}_percent"
print(f"Creating Directory: '{target_dir_name}'")

target_dir=pathlib.Path(target_dir_name) #setup the directories

target_dir.mkdir(parents=True, exist_ok=True)

import shutil

import shutil

for image_split in label_splits.keys():
    for image_path in label_splits[str(image_split)]:
        dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
        if not dest_dir.parent.is_dir():
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying {image_path} to {dest_dir}...")
        shutil.copy2(image_path, dest_dir)

zip_file_name = data_dir / f"5_Classes_of_Food101_{str(int(amount_to_get*100))}_percent"
food=shutil.make_archive(zip_file_name, 
                    format="zip", 
                    root_dir=target_dir)
food

!ls -la ../data/


import os
import shutil
import random

# source folder holding both images and labels
source_dir = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data"

# output path holding the split data
output_base = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\yolov5_split"

# we are expected to split between train and validation data
def split_dataset(source_dir, output_base = output_base, split_ratio=0.8): # 80% train and 20% validation/test
    
    #fix the organization of folders as to match that accepted by yolo
    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')

    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

    #shuffle then split the data
    image_list = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.png')]
    random.shuffle(image_list)
    split_point = int(len(image_list) * split_ratio)

    for i, img_name in enumerate(image_list):
        target_dir = train_dir if i < split_point else val_dir
        shutil.copy(
            os.path.join(source_dir, 'images', img_name),
            os.path.join(target_dir, 'images', img_name)
        )
        label_name = img_name.replace('.png', '.txt')
        shutil.copy(
            os.path.join(source_dir, 'labels', label_name),
            os.path.join(target_dir, 'labels', label_name)
        )
    
    return train_dir, val_dir

train_dir, val_dir = split_dataset(source_dir, split_ratio=0.8)

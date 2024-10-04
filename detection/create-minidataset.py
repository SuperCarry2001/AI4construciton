## create mini-MOCS for debug-model ##

import json
import random
from collections import defaultdict
import os
import shutil

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

train_json_file = r"E:\MOCS_dataset\instances_train.json"
train_img_dir = r"E:\MOCS_dataset\instances_train\instances_train"
val_json_file = r"E:\MOCS_dataset\instances_val.json"
val_img_dir = r"E:\MOCS_dataset\instances_val\instances_val"


train_data = load_json(train_json_file)
val_data = load_json(val_json_file)


def create_mini_dataset(data, max_per_class=50):
    category_annotations = defaultdict(list)
    for ann in data['annotations']:
        category_annotations[ann['category_id']].append(ann)


    mini_annotations = []
    for category_id, anns in category_annotations.items():
        if len(anns) > max_per_class:
            mini_annotations.extend(random.sample(anns, max_per_class))
        else:
            mini_annotations.extend(anns)

    image_ids = set(ann['image_id'] for ann in mini_annotations)

    mini_images = [img for img in data['images'] if img['id'] in image_ids]

    mini_data = {
        'images': mini_images,
        'annotations': mini_annotations,
        'categories': data['categories'],
    }
    return mini_data

mini_train_data = create_mini_dataset(train_data, max_per_class=50)
mini_val_data = create_mini_dataset(val_data, max_per_class=50)

# 保存新的 mini 数据集到 JSON 文件
def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)

save_json(mini_train_data, r"E:\MOCS_dataset\instances_train_mini.json")
save_json(mini_val_data, r"E:\MOCS_dataset\instances_val_mini.json")

def copy_mini_images(mini_data, img_dir, mini_img_dir):
    os.makedirs(mini_img_dir, exist_ok=True)

    for img in mini_data['images']:
        src_path = os.path.join(img_dir, img['file_name'])
        dst_path = os.path.join(mini_img_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)


copy_mini_images(mini_train_data, train_img_dir, r"E:\MOCS_dataset\instances_train_mini")
copy_mini_images(mini_val_data, val_img_dir, r"E:\MOCS_dataset\instances_val_mini")

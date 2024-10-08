"""
{
    "images": [
        {
            "height": 675,
            "width": 1200,
            "id": 1,
            "file_name": "0000001.jpg"
        },
    ],
    "categories": [
        {
            "supercategory": "Construction",
            "id": 1,
            "name": "Worker"
        },
    ],
    "annotations": [
        {
            "segmentation": [
                [
                    298.17353199999997,
                    354.9018,
                    295.3801,
                    343.7262,
                    295.3801,
                    336.5424,
                    298.971792,
                    334.5468,
                    302.16483200000005,
                    336.5424,
                    305.357872,
                    339.735,
                    306.55573999999996,
                    346.12080000000003,
                    307.354,
                    353.3052
                ]
            ],
            "iscrowd": 0,
            "image_id": 19404,
            "bbox": [
                295.0,
                334.0,
                12.0,
                20.0
            ],
            "area": 240.0,
            "category_id": 11,
            "id": 116945
        }
    ]
}
"""

## create mini-MOCS for debug-model ##


import json
import random
from collections import defaultdict
import os
import shutil

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


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

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)

def copy_mini_images(mini_data, img_dir, mini_img_dir):
    os.makedirs(mini_img_dir, exist_ok=True)

    for img in mini_data['images']:
        src_path = os.path.join(img_dir, img['file_name'])
        dst_path = os.path.join(mini_img_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)

'''
train_json_file = r"E:\MOCS_dataset\instances_train.json"
train_img_dir = r"E:\MOCS_dataset\instances_train\instances_train"
val_json_file = r"E:\MOCS_dataset\instances_val.json"
val_img_dir = r"E:\MOCS_dataset\instances_val\instances_val"
train_data = load_json(train_json_file)
val_data = load_json(val_json_file)
mini_train_data = create_mini_dataset(train_data, max_per_class=50)
mini_val_data = create_mini_dataset(val_data, max_per_class=50)


# create mini dataset every class 50 images

save_json(mini_train_data, r"E:\MOCS_dataset\instances_train_mini.json")
save_json(mini_val_data, r"E:\MOCS_dataset\instances_val_mini.json")
copy_mini_images(mini_train_data, train_img_dir, r"E:\MOCS_dataset\instances_train_mini")
copy_mini_images(mini_val_data, val_img_dir, r"E:\MOCS_dataset\instances_val_mini")
'''

# create mini dataset with interesting classes from train and val
interesting_classes_id = [7,8]
def filter_interesting_classes(data, interesting_classes_id):
    mini_annotations = [ann for ann in data['annotations'] if ann['category_id'] in interesting_classes_id]
    image_ids = set(ann['image_id'] for ann in mini_annotations)
    mini_images = [img for img in data['images'] if img['id'] in image_ids]
    mini_data = {
        'images': mini_images,
        'annotations': mini_annotations,
        'categories': data['categories'],
    }
    return mini_data

#mini_train_data = filter_interesting_classes(load_json(r"E:\MOCS_dataset\instances_train.json"), interesting_classes_id)
#save_json(mini_train_data, r"E:\MOCS_dataset\chosen_mini_train.json")
#copy_mini_images(mini_train_data, r"E:\MOCS_dataset\instances_train\instances_train", r"E:\MOCS_dataset\chosen_mini_train")
data = load_json(r"E:\MOCS_dataset\chosen_train.json")
mini_train_data = create_mini_dataset(data, max_per_class=100)
save_json(mini_train_data, r"E:\MOCS_dataset\chosen_train_mini.json")
copy_mini_images(mini_train_data, r"E:\MOCS_dataset\chosen_train", r"E:\MOCS_dataset\chosen_train_mini")
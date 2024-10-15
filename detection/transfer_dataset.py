## create mini-dataset for debug-model ##
import json
import random
from collections import defaultdict
import os
import shutil

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

## 每类数据选50个
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

## mocs2yolo dataloader ## 
## mocs is COCO format, and yolo is yolo format ##

MOCS_categories = {1: "Worker", 2: "Static crane", 3: "Hanging head", 4: "Crane", 5: "Roller",
                    6: "Bulldozer", 7: "Excavator", 8: "Truck",9: "Loader", 10: "Pump truck",
                      11: "Concrete mixer", 12: "Pile driving", 13: "Other vehicle"}


def mocs2yolo(json_file, output_dir):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    categories = MOCS_categories
    interested_category_id = 1

    for annotation in coco_data["annotations"]:
        image_info = next(image for image in coco_data["images"] if image["id"] == annotation["image_id"])
        image_width = image_info["width"]
        image_height = image_info["height"]

        category_id = annotation["category_id"]
        category_name = categories[category_id]


        if category_id == interested_category_id:
            x, y, w, h = annotation["bbox"]
            x_center = x + w / 2
            y_center = y + h / 2

            # Normalize coordinates
            x_center /= image_width
            y_center /= image_height
            w /= image_width
            h /= image_height

            # Format YOLO line: cls, x_center, y_center, width, height
            yolo_line = f"{category_id - 1} {x_center} {y_center} {w} {h}"

            # Generate YOLO file name based on the original image file name
            image_file_name = image_info["file_name"]
            yolo_file_name = os.path.splitext(image_file_name)[0] + ".txt"
            yolo_file_path = os.path.join(output_dir, yolo_file_name)

            # Write YOLO line to the file
            with open(yolo_file_path, "a") as yolo_file:
                yolo_file.write(yolo_line + "\n")

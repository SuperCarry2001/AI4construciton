## for yolov8 ##
from ultralytics import YOLO
import json
import os


## mocs2yolo dataloader ##

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

'''
json_file_path = r"E:\MOCS\instances_val.json"
output_directory = r"E:\COCO\yolov8-test\yolo-master\YOLOv3\PyTorch-YOLOv3\data\MOCS\labels"
mocs2yolo(json_file_path, output_directory)
'''

## train yolov8 ##

## before traing, we need to fix .yaml file, and change the path of train and val data ##


model = YOLO(r'C:\Users\ZJ\Desktop\AI4construciton\detection\yolov8_original.pt')  # load a pretrained model (recommended for training)
model.train(data=r'C:\Users\ZJ\Desktop\AI4construciton\detection\MOCS.yaml', epochs=100, imgsz=640, batchsize = 32)
import os
import json
from PIL import Image

# Input/Output paths
image_dir = 'C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\images'        # path to .png images
json_dir = 'C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\labels\\json'    # path to JSON label files
output_label_dir = 'C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\labels'  # desired path for YOLO format labels
os.makedirs(output_label_dir, exist_ok=True) # creating the folder that will hold the labels as txt files

# Convert each JSON to YOLO format
for json_file in os.listdir(json_dir):
    if not json_file.endswith('.json'):  #in case the path has several types of files, only consider the json files which hold lables
        continue

    json_path = os.path.join(json_dir, json_file) # retrieve the path for each unique json lables file
    image_name = json_file.replace('.json', '.png') # image and label in json file have same name
    image_path = os.path.join(image_dir, image_name) # retrieve the path for each unique image

    if not os.path.exists(image_path):
        print(f"Warning: {image_name} not found.")
        continue
    
     # get the height and width for image
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    # read the contents of json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # convert to YOLO format
    yolo_lines = []
    for obj in data:
        # retrieve class id
        class_id = obj['ObjectClassId'] 
        # retrieve coordinates of bounding box
        x1, y1 = obj['Left'], obj['Top'] 
        x2, y2 = obj['Right'], obj['Bottom']

        # get the coordinates of bounding box in yolo format + normalize
        x_center = (((x1 + x2)/2.0) / img_w)
        y_center = (((y1 + y2)/2.0) / img_h)
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        # put the yolo formatted data retrieved in a line
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}") 

    # save .txt label file
    output_txt = os.path.join(output_label_dir, json_file.replace('.json', '.txt'))

    # write on this saved txt file, on each line, the data corresponding for 1 unique object detected
    with open(output_txt, 'w') as f:
        f.write('\n'.join(yolo_lines))

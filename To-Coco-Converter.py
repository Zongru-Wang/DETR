import os
import json
from PIL import Image

# Define base paths
base_dir = './Single-Images-With-Label'
output_json_path = './coco_annotations.json'
classes_file_path = os.path.join(base_dir, 'classes.txt')

# Read class names
with open(classes_file_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Prepare COCO format data
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories to COCO data
for idx, cls in enumerate(classes):
    coco_data["categories"].append({
        "id": idx + 1,
        "name": cls,
        "supercategory": "none"
    })

# Function to read annotations from a file
def read_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
    return annotations

# Process each image and its annotations
image_id = 1
annotation_id = 1

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and not folder.endswith('_labels'):
        # Corresponding labels folder
        labels_folder = f"{folder}_labels"
        labels_folder_path = os.path.join(base_dir, labels_folder)
        
        if not os.path.exists(labels_folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                image_path = os.path.join(folder_path, file)
                label_file = os.path.splitext(file)[0] + '.txt'
                label_path = os.path.join(labels_folder_path, label_file)

                # Skip if annotation file does not exist
                if not os.path.exists(label_path):
                    continue

                # Read image
                image = Image.open(image_path)
                image_width, image_height = image.size

                # Add image information to COCO data
                image_info = {
                    "file_name": os.path.relpath(image_path, base_dir),
                    "id": image_id,
                    "width": image_width,
                    "height": image_height
                }
                coco_data["images"].append(image_info)

                # Read annotations
                annotations = read_annotations(label_path)
                for annotation in annotations:
                    class_id, x_center, y_center, width, height = map(float, annotation)
                    class_id = int(class_id) + 1  # COCO class id starts from 1

                    # Calculate bounding box coordinates
                    bbox_width = width * image_width
                    bbox_height = height * image_height
                    bbox_x = (x_center * image_width) - (bbox_width / 2)
                    bbox_y = (y_center * image_height) - (bbox_height / 2)

                    # Add annotation to COCO data
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_info)

                    annotation_id += 1

                image_id += 1

# Write COCO data to JSON file
with open(output_json_path, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations have been saved to {output_json_path}")

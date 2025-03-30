import os
import cv2

def save_augmented_data(image, bboxes, class_names, image_base_name, images_path, labels_path, suffix):

    output_image_path = os.path.join(images_path, image_base_name + f"{suffix}.png") # the image are output with name being: (original_name + suffix).png
    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # save the image in the defined output_image_path

   # save yolo .txt labels for each image in output_txt_path
    output_txt_path = os.path.join(labels_path, image_base_name + f"{suffix}.txt")
    with open(output_txt_path, 'w') as f:
        for bbox, cls_name in zip(bboxes, class_names):
            class_id = int(float(cls_name))
            x_center, y_center, box_width, box_height = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"Saved: {output_image_path} & {output_txt_path}") #to make sure everything is write
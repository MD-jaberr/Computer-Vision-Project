import albumentations as A
import cv2

#define 2 different pipelines for image augmentation

transform1 = A.Compose([
    A.Blur(blur_limit=3, p=1.0), #add bluring to image
    A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),  # Rotation within -30 and +30 deg
], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"])) #show the bboxes based on yolo format

transform2 = A.Compose([
    A.GaussNoise(var_limit=(20.0, 100.0), p=1.0) #add gaussian noise to image
], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"])) #show the bboxes based on yolo format

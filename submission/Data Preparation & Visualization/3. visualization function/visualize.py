import cv2
import matplotlib.pyplot as plt

#the function displays the image with the bounding boxes(s)
def visualize(image, bboxes, class_names, box_color=(255, 0, 0), thickness=2):

    img = image.copy()
    img_h, img_w = img.shape[:2] #get height and width

    for bbox, class_name in zip(bboxes, class_names):
        x_center, y_center, w, h = bbox  #retrieve the coordinates of bounding box from bbox variable

        #convert normalized yolo format to pixel coordinates
        x_min = int((x_center - w / 2) * img_w)
        y_min = int((y_center - h / 2) * img_h)
        x_max = int((x_center + w / 2) * img_w)
        y_max = int((y_center + h / 2) * img_h)

        #draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=thickness)

        #draw class label background
        ((text_width, text_height), _) = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x_min, y_min - text_height - 5), (x_min + text_width, y_min), box_color, -1)

        #draw class label text
        cv2.putText(
            img,
            text=str(class_name),
            org=(x_min, y_min - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Show image
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
from torch.utils.data import DataLoader

# Paths
images_path = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\images"
labels_path = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\labels"

# Create dataset and dataloader
dataset = GivenData(images_path, labels_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

for img, bboxes, class_names, image_base_name in dataloader: #dataloader has a getter function that holds: img, bboxes, class_names, image_base_name
    if len(bboxes) == 0:
        print(f"Skipping {image_base_name} (no bboxes)")
        continue

    # get info of yolo bboxes and class ids
    bboxes_only = [box[1:] for box in bboxes]  # strip class_id
    class_ids = [int(name.split(":")[0].strip()) for name in class_names]

    # use the pipelines for augmentation
    try:
        augmented_1 = transform1(image=img, bboxes=bboxes_only, class_labels=class_ids)
        augmented_2 = transform2(image=img, bboxes=bboxes_only, class_labels=class_ids)
    except Exception as e:
        print(f"Skipping {image_base_name} due to error during augmentation: {e}")
        continue

    # extract the augmented data
    augmented_image_1 = augmented_1['image']
    augmented_bboxes_1 = augmented_1['bboxes']
    augmented_class_ids_1 = augmented_1['class_labels']

    augmented_image_2 = augmented_2['image']
    augmented_bboxes_2 = augmented_2['bboxes']
    augmented_class_ids_2 = augmented_2['class_labels']

    # convert class ids to strings
    formatted_class_names_1 = [f"{cls_id}" for cls_id in augmented_class_ids_1]
    formatted_class_names_2 = [f"{cls_id}" for cls_id in augmented_class_ids_2]

    # visualize and save
    visualize(augmented_image_1, augmented_bboxes_1, formatted_class_names_1)
    visualize(augmented_image_2, augmented_bboxes_2, formatted_class_names_2)

    save_augmented_data(augmented_image_1, augmented_bboxes_1, formatted_class_names_1,
                        image_base_name, images_path, labels_path, suffix="_augmented_1") #augmentation of first pipeline ends with "_augmented_1"

    save_augmented_data(augmented_image_2, augmented_bboxes_2, formatted_class_names_2,
                        image_base_name, images_path, labels_path, suffix="_augmented_2") #augmentation of second pipeline ends with "_augmented_2"
 


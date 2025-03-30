from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import cv2


class GivenData(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_files = [f for f in os.listdir(images_path) if f.lower().endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]  #get the image file name by index
        image_base_name = os.path.splitext(image_file)[0]  #get the raw name of the image file
        image_path = os.path.join(self.images_path, image_file)  #get the path of the image file
        label_path = os.path.join(self.labels_path, image_base_name + ".txt")  #the label has same name but is .txt, and get its path also

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert to RGB to ensure consistent channel size

        bboxes = []  #variable that stores the coordinates of bounding boxes (labels)
        class_names = []  #variable that stores class names of labels

        with open(label_path, "r") as f:
            #read each line in the file holding labels, extract the bounding boxes information, and store them in bboxes and class_names
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    bboxes.append([class_id, x_center, y_center, width, height])
                    class_names.append(f"{int(class_id)}")


        return img, bboxes, class_names, image_base_name  


images_path = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\images"
labels_path = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\data\\labels"

dataset = GivenData(images_path, labels_path)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

import os

labels_root = "C:/Users/PC/OneDrive - Lebanese American University/inmind/ML track/final project/data/labels"

for filename in os.listdir(labels_root):
    if filename.endswith('.txt'):
        file_path = os.path.join(labels_root, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = [] #create a list that holds the updated lines
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0]) - 1  # subtract the class ids by 1 as to have class ids starting from 0 till 4, instead of 1 till 5
                new_line = f"{class_id} {' '.join(parts[1:])}\n" # the new_line holds the same content as old one, but class id is subtracted by 1
                new_lines.append(new_line) # append the new line or lines for each label file in new_lines list

        with open(file_path, 'w') as f:
            f.writelines(new_lines)  # erase all the old content from the old label file, and write the new lines on it



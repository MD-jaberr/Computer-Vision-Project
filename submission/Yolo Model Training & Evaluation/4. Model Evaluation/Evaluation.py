import os

for root, dirs, files in os.walk("runs/val/"):
    for file in files:
        img_path = os.path.join(root, file)
        print(img_path)
        display(Image(filename=img_path))
import os
import random

# Creates random train, validation, and test splits from dataset

file_list = os.listdir("data/MetalDAM/cropped_images")

for i in range(5):
    train_list = []
    test_list = []
    val_list = []
    my_list = ["train", "test", "val"]
    for file in file_list:
        t = random.choice(my_list)
        if t == "train":
            train_list.append(file)
        elif t == "test":
            test_list.append(file)
        else:
            val_list.append(file)
        
        if len(train_list) == 24 and "train" in my_list:
            my_list.remove("train")
        if len(test_list) == 8 and "test" in my_list:
            my_list.remove("test")
        if len(val_list) == 10 and "val" in my_list:
            my_list.remove("val")

    with open(f"data/MetalDAM/splits/train{i}.txt", 'w') as file:
        for image_name in train_list:
            file.write(f"{image_name}\n")

    with open(f"data/MetalDAM/splits/test{i}.txt", 'w') as file:
        for image_name in test_list:
            file.write(f"{image_name}\n")

    with open(f"data/MetalDAM/splits/val{i}.txt", 'w') as file:
        for image_name in val_list:
            file.write(f"{image_name}\n")

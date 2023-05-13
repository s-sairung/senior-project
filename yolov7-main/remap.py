import os
from tqdm import tqdm
import datetime

path_to_dataset = "./roboflow"
subdirectories = ["train", "valid"]

current_mapping = {0: "scooter", 1: "tuktuk"}
new_mapping = {80: "scooter", 81: "tuktuk"}

for subdirectory in subdirectories:
    path_to_subdirectory = os.path.join(path_to_dataset, subdirectory)
    path_to_label = os.path.join(path_to_subdirectory, "labels")
    for label_file in tqdm(os.listdir(path_to_label)):
        f = open(os.path.join(path_to_label, label_file), "r")
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            parts = line.split(" ")
            class_label = int(parts[0])
            if class_label in current_mapping:
                new_class_label = [j for j in new_mapping if new_mapping[j] == current_mapping[class_label]][0]
                lines[i] = " ".join([str(new_class_label)] + parts[1:])
        f = open(os.path.join(path_to_label, label_file), "w")
        f.writelines(lines)
        f.close()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = os.path.join(path_to_dataset, "label_mapping_log.txt")
log_file = open(log_path, "a")
log_file.write("[{}]: {} -> {}\n".format(timestamp, current_mapping, new_mapping))
log_file.close()
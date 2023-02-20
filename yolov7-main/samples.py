samples_per_class = 500
#dir_in = "./dataset/coco"
#dir_out = "./dataset/coco_sampling"

dir_in = "./data/coco"
dir_out = "./data/coco_sampling"


## Initialize dictionary variable
import os

train_folder_in = os.path.join(dir_in, "train")
image_folder_in = os.path.join(train_folder_in, "images")
label_folder_in = os.path.join(train_folder_in, "labels")
label_files = os.listdir(label_folder_in)

train_folder_out = os.path.join(dir_out, "train")
image_folder_out = os.path.join(train_folder_out, "images")
label_folder_out = os.path.join(train_folder_out, "labels")


##Generate files dictionary and save dictionary variable into a file
import pickle

try:
    f = open("sampling_image_dict.pckl", "rb")
    class_files = pickle.load(f)
    f.close()
    print("Dictionary variable has been loaded from disk.")
except (FileNotFoundError, EOFError) as e:
    class_files = {}
    print("Dictionary variable has not been loaded from disk. Manual generation is required.")

    from tqdm import tqdm

    for filename in tqdm(label_files):
        filepath = os.path.join(label_folder_in, filename)
        label_file = open(filepath, "r")
        labels = label_file.readlines()
        for line in labels:
            label_class = int(line.split(" ")[0])
            try:
                class_files[label_class].add(filename)
            except KeyError:
                class_files[label_class] = {filename}
        label_file.close()

    f = open("sampling_image_dict.pckl", "wb")
    pickle.dump(class_files, f)
    f.close()

    print("Dictionary variable has been saved to disk.")

## Random Sampling
# Note: The number of output samples might be less than samples_per_class * number_of_classes as one label file could contain more than one class.
import random

sample_labels = set()

for class_index in class_files:
    samples = random.sample([*class_files[class_index]], samples_per_class)
    sample_labels = sample_labels.union(samples)
    
print(f"Successfully picked {len(sample_labels)} random samples.")


##Copy Files to the Destination Directory
import shutil
from tqdm import tqdm

if not os.path.isdir(dir_out):
    os.makedirs(dir_out)
if not os.path.isdir(label_folder_out):
    os.makedirs(label_folder_out)
if not os.path.isdir(image_folder_out):
    os.makedirs(image_folder_out)

for filename in tqdm(sample_labels):
    label_source_path = os.path.join(label_folder_in, filename)
    label_destination_path = os.path.join(label_folder_out, filename)
    shutil.copyfile(label_source_path, label_destination_path)

    filename_image = filename.removesuffix(".txt") + ".jpg"
    image_source_path = os.path.join(image_folder_in, filename_image)
    image_destination_path = os.path.join(image_folder_out, filename_image)
    shutil.copyfile(image_source_path, image_destination_path)

print(f"Done copying random samples from {dir_in} to {dir_out}.")
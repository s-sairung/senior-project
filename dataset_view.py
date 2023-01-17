import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

#
# Images that contain all `classes` will be prioritized first, followed
# by images that contain at least one of the required `classes`. If
# there are not enough images matching `classes` in the split to meet
# `max_samples`, only the available images will be loaded.
#
# Images will only be downloaded if necessary
#

splits = ["train", "validation"]
classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=splits,
    label_types=["detections"],
    classes=classes,
)

# dataset.persistent = True # Delete dataset with 'fo.load_dataset("coco-2017-train-validation").delete()'

view = dataset.filter_labels("ground_truth", F("label").is_in(classes))

session = fo.launch_app(view)
session.wait()
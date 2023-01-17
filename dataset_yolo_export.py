import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

export_dir = "yolov7_dataset"
label_field = "ground_truth"

splits = ["train", "validation"]
classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation"],
    label_types=["detections"],
    classes=classes,
)

# dataset.persistent = True # Delete dataset with 'fo.load_dataset("coco-2017-train-validation").delete()'

view = dataset.filter_labels("ground_truth", F("label").is_in(classes))

for split in splits:
    split_view = view.match_tags(split)
    split_view.export(
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
        export_dir=export_dir,  
    )
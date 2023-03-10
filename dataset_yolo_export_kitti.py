import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

export_dir = "yolov7_dataset_kitti"
label_field = "ground_truth"

classes = ["Pedestrian", "Car"]
# classes = ["Pedestrian", "Truck", "Car", "Cyclist", "DontCare", "Misc", "Van", "Tram", "Person_sitting"]

dataset = foz.load_zoo_dataset("kitti", split="train")

# dataset.persistent = True # Delete dataset with 'fo.load_dataset("coco-2017-train-validation").delete()'

view = dataset.filter_labels("ground_truth", F("label").is_in(classes))

view.export(
    dataset_type=fo.types.YOLOv5Dataset,
    label_field=label_field,
    split="train",
    export_dir=export_dir,
    classes=classes,
)
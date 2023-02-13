
from pylabel import importer

path_to_tarin_coco_annotations = "original_dataset/annotations/instances_train.json"
path_to_tarin_images = "../../dataset/train/images/"
path_to_tarin_yolo_annotations = "dataset/train/labels/"

dataset = importer.ImportCoco(path_to_tarin_coco_annotations, path_to_images=path_to_tarin_images, name="train")
dataset.export.ExportToYoloV5(output_path = path_to_tarin_yolo_annotations, cat_id_index=0)

path_to_test_coco_annotations = "original_dataset/annotations/instances_test.json"
path_to_test_images = "../../dataset/test/images/"
path_to_test_yolo_annotations = "dataset/test/labels/"

dataset = importer.ImportCoco(path_to_test_coco_annotations, path_to_images=path_to_test_images, name="test")
dataset.export.ExportToYoloV5(output_path = path_to_test_yolo_annotations, cat_id_index=0)

path_to_val_coco_annotations = "original_dataset/annotations/instances_val.json"
path_to_val_images = "../../dataset/val/images/"
path_to_val_yolo_annotations = "dataset/val/labels/"

dataset = importer.ImportCoco(path_to_val_coco_annotations, path_to_images=path_to_val_images, name="validate")
dataset.export.ExportToYoloV5(output_path = path_to_val_yolo_annotations, cat_id_index=0)
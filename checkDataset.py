
from pylabel import importer
import numpy as np
import cv2

def ShowBoundingBoxes(indx, dataset):
    res = dataset.visualize.ShowBoundingBoxes(indx)
    open_cv_image = np.array(res) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2.imshow('Detection', open_cv_image) 
    cv2.waitKey() & 0xFF == 27

path_to_tarin_coco_annotations = "original_dataset/annotations/instances_train.json"
path_to_tarin_images = "../../dataset/train/images/"
path_to_tarin_yolo_annotations = "dataset/train/labels/"

path_to_test_coco_annotations = "original_dataset/annotations/instances_test.json"
path_to_test_images = "../../dataset/test/images/"
path_to_test_yolo_annotations = "dataset/test/labels/"

path_to_val_coco_annotations = "original_dataset/annotations/instances_val.json"
path_to_val_images = "../../dataset/val/images/"
path_to_val_yolo_annotations = "dataset/val/labels/"

train_dataset = importer.ImportCoco(path=path_to_tarin_coco_annotations, path_to_images=path_to_tarin_images, name="train")
test_dataset = importer.ImportCoco(path_to_test_coco_annotations, path_to_images=path_to_test_images, name="test")
val_dataset = importer.ImportCoco(path_to_val_coco_annotations, path_to_images=path_to_val_images, name="validate")

img_indx = 4502
ShowBoundingBoxes(img_indx, train_dataset)

img_indx = 600
ShowBoundingBoxes(img_indx, test_dataset)

img_indx = 590
ShowBoundingBoxes(img_indx, val_dataset)

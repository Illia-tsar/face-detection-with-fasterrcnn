import os
import pandas as pd
from sklearn.model_selection import train_test_split
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def load_train_test(data_path, val_size):
    train_df = pd.read_csv(os.path.join(data_path, "train/train.csv"), delimiter=",")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"), delimiter=",")

    test_names = test_df.Name.tolist()
    train_names, val_names = train_test_split(train_df.Name.tolist(), test_size=val_size, random_state=17)
    return train_names, val_names, test_names


def create_dataset_dict(filenames, bboxes, data_path, is_test=False):
    dataset_dicts = []
    for img_id, img_name in enumerate(filenames):
        img_path = os.path.join(data_path, "train/image_data", img_name)

        if not is_test:
            img_bboxes = bboxes[bboxes.Name == img_name]

            record = {
                "file_name": img_path,
                "image_id": img_id,
                "height": img_bboxes.iloc[0].height,
                "width": img_bboxes.iloc[0].width
            }

            objs = []
            for _, row in img_bboxes.iterrows():
                xmin = int(row.xmin)
                ymin = int(row.ymin)
                xmax = int(row.xmax)
                ymax = int(row.ymax)

                objs.append(
                    {
                        "bbox": [xmin, ymin, xmax, ymax],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                        "iscrowd": 0
                    }
                )

            record["annotations"] = objs
            dataset_dicts.append(record)
        else:
            dataset_dicts.append({"file_name": img_path})
    return dataset_dicts


def register_datasets(data_path, val_size=.16):
    bboxes = pd.read_csv(os.path.join(data_path, "train/bbox_train.csv"), delimiter=",")
    train_names, val_names, test_names = load_train_test(data_path, val_size)

    for d in ["train", "val", "test"]:
        if d == "test":
            DatasetCatalog.register(
                "faces_test",
                lambda:
                create_dataset_dict(
                    test_names,
                    bboxes,
                    data_path,
                    is_test=True
                )
            )
        else:
            DatasetCatalog.register(
                "faces_" + d,
                lambda d=d:
                create_dataset_dict(
                    train_names if d == "train" else val_names,
                    bboxes,
                    data_path
                )
            )
        MetadataCatalog.get("faces_" + d).set(thing_classes=["face"])

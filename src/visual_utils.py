import random
import math
import matplotlib.pyplot as plt
from dataset import register_datasets
from model_utils.config import gen_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata
from detectron2.engine import DefaultPredictor


class SimpleVisualizer:
    def __init__(
        self,
        data_path,
        weights_path="",
        val_size=.16
    ):
        self.cfg = gen_cfg()
        self.cfg.MODEL.WEIGHTS = self.cfg.MODEL.WEIGHTS if not weights_path else weights_path

        register_datasets(data_path, val_size=val_size)

    def _apply_visualizer(self, img, instances, metadata, is_gt=False):
        visualizer = Visualizer(
            img,
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE
        )
        if is_gt:
            visualizer = visualizer.draw_dataset_dict(instances)
        else:
            visualizer = visualizer.draw_instance_predictions(instances)
        return visualizer.get_image()

    def plot(self, num_to_show=2, subset="train", threshold=.85, ground_truth=False):
        dataset_dicts = DatasetCatalog.get("faces_" + subset)
        dataset_metadata = MetadataCatalog.get("faces_" + subset)

        if not ground_truth:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            predictor = DefaultPredictor(self.cfg)

        rows = math.ceil(num_to_show / 2)
        _, ax = plt.subplots(rows, 2, figsize=(13, 13))
        for idx, dct in enumerate(random.sample(dataset_dicts, num_to_show)):
            img = plt.imread(dct["file_name"])

            result = self._apply_visualizer(
                img,
                dct if ground_truth else predictor(img[..., ::-1])["instances"].to("cpu"),
                dataset_metadata,
                is_gt=ground_truth
            )
            if rows == 1:
                ax[idx].imshow(result)
            else:
                ax[idx // 2, idx % 2].imshow(result)

        plt.show()

    def save_predict(self, img_path, save_path, threshold=.85):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        predictor = DefaultPredictor(self.cfg)

        img = plt.imread(img_path)
        pred = predictor(img[..., ::-1])
        instances = pred["instances"].to("cpu")

        img_meta = Metadata()
        img_meta.set(thing_classes=["face"])

        result = self._apply_visualizer(img, instances, img_meta, is_gt=False)
        plt.imsave(save_path, result)

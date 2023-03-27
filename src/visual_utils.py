from dataset import register_datasets
from model_utils.config import gen_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode


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

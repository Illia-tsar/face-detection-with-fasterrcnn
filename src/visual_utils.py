from dataset import register_datasets
from model_utils.config import gen_cfg


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

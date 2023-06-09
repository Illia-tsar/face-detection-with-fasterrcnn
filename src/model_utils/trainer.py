import os
import pprint
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
from model_utils.hooks import LossEvalHook
from detectron2.engine.hooks import LRScheduler, PeriodicWriter
from torch.optim.lr_scheduler import CyclicLR


class RCNNTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("eval_coco", exist_ok=True)
            output_folder = "eval_coco"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            )
        )
        hooks.pop()
        hooks.append(
            PeriodicWriter(self.build_writers(), period=1)
        )
        pprint.pprint(hooks)

        return hooks

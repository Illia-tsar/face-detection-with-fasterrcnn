from detectron2 import model_zoo
from detectron2.config import get_cfg


def gen_cfg(config=None):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("faces_train",)
    cfg.DATASETS.TEST = ("faces_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    if config is None:
        # if config is None, set the default values
        cfg.DATALOADER.NUM_WORKERS = 2

        cfg.MODEL.BACKBONE.FREEZE_AT = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.MAX_ITER = 18000
        cfg.SOLVER.BASE_LR = 0.0005
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.SOLVER.STEPS = (3000, 12000)
        cfg.SOLVER.WARMUP_ITERS = 1200
        cfg.SOLVER.GAMMA = 0.1

        cfg.TEST.EVAL_PERIOD = 600

    else:
        cfg.DATALOADER.NUM_WORKERS = config.num_workers

        cfg.MODEL.BACKBONE.FREEZE_AT = config.freeze_at
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.roi_batch_size
        cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS if not config.weights_path else config.weights_path

        cfg.SOLVER.IMS_PER_BATCH = config.batch_size
        cfg.SOLVER.MAX_ITER = config.max_iter
        cfg.SOLVER.BASE_LR = config.learning_rate
        cfg.SOLVER.CHECKPOINT_PERIOD = config.checkpoint
        cfg.SOLVER.STEPS = tuple(config.steps)
        cfg.SOLVER.WARMUP_ITERS = config.warmup_iters
        cfg.SOLVER.GAMMA = config.gamma

        cfg.TEST.EVAL_PERIOD = config.eval_period

    return cfg

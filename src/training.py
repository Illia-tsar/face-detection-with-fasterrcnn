import argparse
import pprint
from detectron2.utils.logger import setup_logger; setup_logger()
from model_utils.trainer import RCNNTrainer
from model_utils.config import gen_cfg
from dataset import register_datasets

parser = argparse.ArgumentParser(
    prog="training",
    description="script for fine-tuning FasterRCNN for Face Detection",
    allow_abbrev=False
)

data_group = parser.add_argument_group("data manipulation")
data_group.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="number of data loading threads(default: %(default)s)"
)
data_group.add_argument(
    "-p",
    "--data_path",
    type=str,
    default="../data",
    help="path to data(default: %(default)s)"
)
data_group.add_argument(
    "--val_size",
    type=float,
    default=0.16,
    help="fraction used for splitting data into train and val(default: %(default)s)"
)

model_group = parser.add_argument_group("model parameters")
model_group.add_argument(
    "--freeze_at",
    type=int,
    default=2,
    choices=range(1, 6),
    help="freeze the first n stages of backbone(default: %(default)s)"
)
model_group.add_argument(
    "--roi_batch_size",
    type=int,
    default=512,
    help="number of regions of interest during training(default: %(default)s)"
)

solver_group = parser.add_argument_group("solver parameters")
solver_group.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="the batch size to use(default: %(default)s)"
)
solver_group.add_argument(
    "--max_iter",
    type=int,
    default=18000,
    help="number of iterations for training(default: %(default)s)"
)
solver_group.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.0005,
    help="controls learning rate(default: %(default)s)"
)
solver_group.add_argument(
    "-s",
    "--checkpoint",
    type=int,
    default=1000,
    help="model's weights are saved after every this number of iterations(default: %(default)s)"
)
solver_group.add_argument(
    "--steps",
    nargs="+",
    type=int,
    default=[3000, 12000],
    help="learning rate is decreased by gamma after this number of iterations(default: %(default)s)"
)
solver_group.add_argument(
    "--warmup_iters",
    type=int,
    default=1200,
    help="number of iterations used for warmup(default: %(default)s)"
)
solver_group.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    help="number used to reduce lr after #steps iterations(default: %(default)s)"
)

test_group = parser.add_argument_group("test parameters")
test_group.add_argument(
    "--eval_period",
    type=int,
    default=600,
    help="evaluate on validation data after this number of iterations(default: %(default)s)"
)


def main(conf):
    register_datasets(conf.data_path, conf.val_size)
    cfg = gen_cfg(config=conf)

    trainer = RCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    config = parser.parse_args()
    pprint.pprint(config)
    main(config)

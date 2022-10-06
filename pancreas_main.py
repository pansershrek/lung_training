import argparse
import logging

from torch.utils.tensorboard import SummaryWriter

from utils.log import Logger
from pancreas_dataset import PancreasDataset
from pancreas_masked_dataset import PancreasMaskedDataset
from pancreas_trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-path-train",
        default="/pancreas/train/imagesTr",
        help="Path to train images"
    )
    parser.add_argument(
        "--bbox-path-train",
        default="/pancreas/train/bbox3d",
        help="Path to train bboxes"
    )
    parser.add_argument(
        "--images-path-val",
        default="/pancreas/val/imagesTr",
        help="Path to val images"
    )
    parser.add_argument(
        "--bbox-path-val",
        default="/pancreas/val/bbox3d",
        help="Path to val bboxes"
    )
    parser.add_argument(
        "--checkpoint-save-dir",
        default="/pancreas/checkpoint_save_dir",
        help="Path to checkpoint's saves dir"
    )
    parser.add_argument(
        "--images-path-inference",
        default="/pancreas/inference/imagesTr",
        help="Path to inference images"
    )
    parser.add_argument(
        "--bbox-path-inference",
        default="/pancreas/inference/bbox3d",
        help="Path to store inference bboxes"
    )
    parser.add_argument("--epochs", default=100, help="Epochs number")
    parser.add_argument("--batch-size", default=8, help="Batch size")
    parser.add_argument("--device", default="cuda:3", help="Device")
    parser.add_argument(
        "--log-path", default="/pancreas/logs", help="Path for logs"
    )
    parser.add_argument(
        "--mode",
        default="inference",
        help="Model mode. There are two options: train and inference"
    )
    parser.add_argument(
        "--inference-model-path",
        default="/pancreas/checkpoint_save_dir/checkpoint_37.pt",
        help="Path to inference model"
    )
    parser.add_argument(
        "--opt-level",
        default="O0",
        help=(
            "Model optimization level. "
            "There are two options: O0, O1, O2 and O3 (first char is big `o`.)"
        )
    )
    args = parser.parse_args()

    train_dataset = PancreasMaskedDataset(
        args.images_path_train, args.bbox_path_train
    )
    val_dataset = PancreasMaskedDataset(
        args.images_path_val, args.bbox_path_val, validate=True
    )
    inference_dataset = PancreasDataset(
        args.images_path_inference, args.bbox_path_inference, validate=True
    )

    writer = SummaryWriter(log_dir=args.log_path)
    logger = Logger(
        log_file_name=f"{args.log_path}/pancreas_logs",
        log_level=logging.DEBUG,
        logger_name='YOLOv4'
    ).get_log()

    trainer = Trainer(
        train_dataset, val_dataset, inference_dataset,
        args.checkpoint_save_dir, writer, logger, args.device, args.epochs,
        args.batch_size, args.opt_level, args.inference_model_path
    )
    if args.mode == "inference":
        trainer.inference()
    else:
        trainer.train()


if __name__ == "__main__":
    main()

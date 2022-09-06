import argparse
import logging

from torch.utils.tensorboard import SummaryWriter

from utils.log import Logger
from pancreas_dataset import PancreasDataset
from pancreas_trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-path-train", default="", help="Path to train images"
    )
    parser.add_argument(
        "--bbox-path-train", default="", help="Path to train bboxes"
    )
    parser.add_argument(
        "--images-path-val", default="", help="Path to val images"
    )
    parser.add_argument(
        "--bbox-path-val", default="", help="Path to val bboxes"
    )
    parser.add_argument(
        "--checkpoint-save-dir",
        default="",
        help="Path to checkpoint's saves dir"
    )
    parser.add_argument("--epochs", default=300, help="Epochs number")
    parser.add_argument("--batch-size", default=4, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--log-path", default="", help="Path for logs")
    args = parser.parse_args()

    train_dataset = PancreasDataset(
        args.images_path_train, args.bbox_path_train
    )
    val_dataset = PancreasDataset(args.images_path_val, args.bbox_path_val)

    writer = SummaryWriter(log_dir=args.log_path)
    logger = Logger(
        log_file_name=args.log_path,
        log_level=logging.DEBUG,
        logger_name='YOLOv4'
    ).get_log()

    trainer = Trainer(
        train_dataset, val_dataset, args.checkpoint_save_dir, writer, logger,
        args.device, args.epochs, args.batch_size
    )


if __name__ == "__main__":
    main()
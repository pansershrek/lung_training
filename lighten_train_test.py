import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import utils.datasets as data

from lighten_model import lightenYOLOv4
import config.yolov4_config as cfg






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/mobilenetv3.pth', help='weight file path')#weight/darknet53_448.weights
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=-1, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    opt = parser.parse_args()



    train_dataset = data.Build_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    train_dataloader = DataLoader(train_dataset,
                                        batch_size=1, #cfg.TRAIN["BATCH_SIZE"],
                                        num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                        shuffle=True, pin_memory=True
                                        )

    trainer = pl.Trainer()
    model = lightenYOLOv4(
        weight_path=opt.weight_path, 
        resume=opt.resume
    )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=train_dataloader)
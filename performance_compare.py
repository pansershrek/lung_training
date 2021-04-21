import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class time_checkpoint():
    anchor_dict = {}
    name_list = []
    def __init__(self, label):
        self.label = label
        self.anchor_dict.update({'head':time.time()})
        self.name_list.append('head')
        self.default_counter = 0

    def add_anchor(self, name=None, hight_light = False):
        if name and not self.anchor_dict.get(name):
            _name = name
        else:
            _name = 'tc'+str(self.default_counter)
            self.default_counter+=1

        current_time = time.time()
        new = {_name: current_time}
        self.anchor_info([self.name_list[-1], self.anchor_dict[self.name_list[-1]]], [_name, current_time], hight_light)
        self.anchor_dict.update(new)
        self.name_list.append(_name)

    def anchor_info(self, last, current, hight_light):
        if hight_light:
            print('=========================================================')
        print('[ {label} ] From {last_name} to {current_name}, takes time {cost}'.format(
            label=self.label,
            last_name=last[0],
            current_name=current[0],
            cost=(current[1] - last[1])
        ))
        if hight_light:
            print('=========================================================')

    def get_statictics(self):
        return "yyy TODO"

class randomDataset(Dataset):
    def __init__(self, data_size):
        self.data_size = data_size
        self.x = torch.randn(data_size, 64)
        self.y = torch.randn(data_size, 64).float()
    def __len__(self):
        return self.data_size
    def __getitem__(self, item):
        x = self.x[item,:]
        y = self.y[item,:]
        return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        output = self.decoder(self.encoder(x))
        return output
        #input = torch.randn(8000, 16, 50, 100).cuda()

class lightNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def simple_train(model, device, train_loader, batch_size):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def main():
    data_size = 1000000
    batch_size = 1024
    tc = time_checkpoint('TEST')

    train_dataloader = DataLoader(randomDataset(data_size),
                                        batch_size=batch_size, #cfg.TRAIN["BATCH_SIZE"],
                                        num_workers=4 #cfg.TRAIN["NUMBER_WORKERS"],
                                )

    # simple
    simple_model = Net().cuda()
    tc.add_anchor('simple_start')
    simple_train(simple_model, 'cuda', train_dataloader, batch_size)
    tc.add_anchor('simple_end', hight_light=True)

    # simple
    simple_model = lightNet().cuda()
    tc.add_anchor('simple_V2_start')
    simple_train(simple_model, 'cuda', train_dataloader, batch_size)
    tc.add_anchor('simple_V2_end', hight_light=True)


    pl_mode = lightNet()
    trainer = pl.Trainer(
        gpus=0, #buggy, opt.gpu_id usage inconsistence with pl document
        max_epochs=1,
    )
    tc.add_anchor('pl_start')
    trainer.fit(pl_mode, train_dataloader=train_dataloader)
    tc.add_anchor('pl_end', hight_light=True)

    print(time_checkpoint.get_statictics())

if __name__ == '__main__':
    main()
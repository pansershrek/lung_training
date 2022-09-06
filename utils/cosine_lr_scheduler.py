import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineDecayLR(_LRScheduler):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0, last_epoch=-1):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup
        self._last_lr = lr_init
        super(CosineDecayLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if self.__warmup and epoch < self.__warmup:
            lr = self.__lr_max / (self.__warmup+1) * (epoch+1)
        else:
            T_max = self.__T_max - self.__warmup
            epoch = epoch - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(epoch/T_max * np.pi))
        self._last_lr = lr
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == '__main__':
    import sys
    sys.path.append("/workspace/pancreas")
    import matplotlib.pyplot as plt
    from model.YOLOv4 import YOLOv4
    import torch.optim as optim

    net = YOLOv4()
    optimizer = optim.SGD(net.parameters(), 1e-4, 0.9, weight_decay=0.0005)
    scheduler = CosineDecayLR(optimizer, 50*2068, 1e-4, 1e-6, 2*2068)

    # Plot lr schedule
    y = []
    for t in range(50):
        for i in range(2068):
            scheduler.step(2068*t+i)
            print(scheduler.get_last_lr())
            y.append(optimizer.param_groups[0]['lr'])

    #print(y)
    plt.figure()
    plt.plot(y, label='LambdaLR')
    plt.xlabel('steps')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.show()
    #plt.savefig("../data/lr.png", dpi=300)
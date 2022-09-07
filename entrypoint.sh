#!/bin/sh
#CUDA_LAUNCH_BLOCKING=1
tensorboard --logdir="/pancreas/logs" --port=6006 &
python3 pancreas_main.py
sleep 10000
#!/usr/bin/env bash

export MXNET_CPU_WORKER_NTHREADS=15
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
python fine-tune.py --train-stage 0 --pretrained-model 'model/resnet-152' --pretrained-epoch 0 --model-prefix 'model/mmd' --num-classes 263 --lr 0.01 --lr-step-epochs '10,16' --num-epochs 18 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 64
sleep 5
python fine-tune.py --train-stage 1 --pretrained-model 'model/mmd' --pretrained-epoch 18 --model-prefix 'model/mmd1' --num-classes 263 --lr 0.0001 --lr-step-epochs '6,8,10,12' --num-epochs 14 --lr-factor 0.5 --gpus 0,1,2,3 --batch-size 64


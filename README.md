# transfer-mxnet
Unsupervised transfer learning for image classification written in mxnet.

This is a library for unsupervised transfer learning using mxnet. We mainly implemented three algorithms:

- `mmd` described in paper "Learning Transferable Features with Deep Adaptation Networks".
- `jmmd` described in paper "Deep Transfer Learning with Joint Adaptation Networks".
- `AdaBN` described in paper "REVISITING BATCH NORMALIZATION FOR PRACTICAL DOMAIN ADAPTATION".

For original caffe implementation of `mmd` and `jmmd`, please refer to [here](https://github.com/thuml/transfer-caffe)

If you have any problem about this code, feel free to concact us with the following email:
- guojia@gmail.com

Note that this repo is only for unsupervised image classfication transfer learning.

Experiments
---------------
We introduce our experiments on cars dataset:
- `Source dataset` is a high quality cars image dataset fetched from web with accurate annotated labels(car models).
- `Target dataset` is a surveillance image dataset which is public available(compcars-sv).

During training, we set all labels in `Target dataset` to null-label(9999 by default) then it becomes unsupervised TL problem.

|    Method    | Accuracy |
| ----------   | -----    |
| CNN(no TL)   |  68.7%   |
| AdaBN        |  71.6%   |
| DAN(mmd)     |  73.7%   |
| JAN(jmmd)    |  78.9%   |

If you want to train your own models with mmd(especically JAN as it is the best approach), please use following steps.

Data Preparation
---------------
 - Download mxnet resnet-152 imagenet-11k pretrained model to `model/` directory, from [here](http://data.mxnet.io/models/imagenet-11k/resnet-152/).
- Prepare your source domain dataset to `data/source.lst` in mxnet `lst` format.
- Prepare your target domain dataset to `data/target.lst`. Generally, label id in `target.lst` should be null-label(9999 by default). But semi-supervised TL is also allowed, you can choose hundreds of items to be their real valid label id.
- Prepare your validation dataset in target domain to `data/val.lst`. It can be the same with `data/target.lst` but all with valid labels.


Training Model
---------------
Env variables setting.
```
export MXNET_CPU_WORKER_NTHREADS=15
```

Training stage 1, do softmax training on source dataset only.

```
python fine-tune.py --train-stage 0 --pretrained-model 'model/resnet-152' --pretrained-epoch 0 --model-prefix 'model/mmd' --num-classes 263 --lr 0.01 --lr-step-epochs '10,16' --num-epochs 18 --lr-factor 0.1 --gpus 0,1,2,3 --batch-size 64
```

Training stage 2, do softmax+mmd joint training to produce final model.

```
python fine-tune.py --train-stage 1 --pretrained-model 'model/mmd' --pretrained-epoch 18 --model-prefix 'model/mmd1' --num-classes 263 --lr 0.0001 --lr-step-epochs '6,8,10,12' --num-epochs 14 --lr-factor 0.5 --gpus 0,1,2,3 --batch-size 64
```

Parameter Tuning
---------------
TODO, check them in source code now.

Use AdaBN
---------------
```
python adabn.py --model <trained-model-prefix> --epoch <load-epoch> --val 'data/val.lst' --gpu 0
```
It will firstly calculate BN statistics using target domain dataset then write back to preloaded model. Second, use this modified model to validate the classification accuracy on target dataset. You can change the corresponding BN layers name in source code.



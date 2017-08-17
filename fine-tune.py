import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
#from common import find_mxnet
from common import data
from common import fit
import mxnet as mx
import mmd

import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists('model/'+filename):
        urllib.urlretrieve(url, 'model/'+ filename)

def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

LABEL_WIDTH = 1

def get_fine_tune_model(symbol, arg_params, args):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    layer_name = args.layer_before_fullc
    all_layers = symbol.get_internals()
    last_before = all_layers[layer_name+'_output']
    lr_mult = 1
    feature = last_before
    fc = mx.symbol.FullyConnected(data=fc7, num_hidden=args.num_classes, name='fc', lr_mult=lr_mult) #, lr_mult=10)
    net = mmd.mmd(feature, fc, args)
    if args.train_stage==0:
      new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    else:
      new_args = arg_params
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str, default='model/resnet-152',
                        help='the pre-trained model')
    parser.add_argument('--pretrained-epoch', type=int, default=0,
                        help='the pre-trained model epoch to load')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--no-checkpoint', action="store_true", default=False,
                        help='do not save checkpoints')
    parser.add_argument('--freeze', action="store_true", default=False,
                        help='freeze lower layers')
    parser.add_argument('--train-stage', type=int, default=0,
                        help='training stage, train softmax only in training stage0 and use mmd loss in training stage1')
    parser.add_argument('--null-label', type=int, default=9999,
                        help='indicate the label id of invalid label')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(data_dir="./data", top_k=0, kv_store='local', data_nthreads=15)
    #parser.set_defaults(model_prefix="", data_nthreads=15, batch_size=64, num_classes=263, gpus='0,1,2,3')
    #parser.set_defaults(image_shape='3,320,320', num_epochs=32,
    #                    lr=.0001, lr_step_epochs='12,20,24,28', wd=0, mom=0.9, lr_factor=0.5)
    parser.set_defaults(image_shape='3,320,320', wd=0, mom=0.9)

    args = parser.parse_args()
    args.label_width = LABEL_WIDTH
    args.gpu_num = len(args.gpus.split(','))
    args.batch_per_gpu = args.batch_size/args.gpu_num
    with open(args.data_dir+'/source.lst') as f:
      args.num_examples = sum(1 for _ in f)
    if args.train_stage==1:
      with open(args.data_dir+'/target.lst') as f:
        target_num_examples = sum(1 for _ in f)
        args.num_examples = min(args.num_examples, target_num_examples)
    print('num_examples', args.num_examples)
    print('gpu_num', args.gpu_num)

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    prefix = args.pretrained_model
    epoch = args.pretrained_epoch
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    fixed_params = None

    if args.freeze:
      fixed_params = []
      active_list = ['bn1', 'fc', 'stage3', 'stage4']
      for k in arg_params:
        is_active = False
        for a in active_list:
          if k.startswith(a):
            is_active = True
            break
        if not is_active:
          fixed_params.append(k)
      print(fixed_params)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args)

    
    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = new_args,
            aux_params  = aux_params,
            fixed_param_names = fixed_params)


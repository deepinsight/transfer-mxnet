import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import random
import json
from common import find_mxnet
import mxnet as mx
from guuker import prt

def get_bn_input_symbol(bn_symbol):
  for sym in bn_symbol.get_children():
    #print(sym.name)
    if sym.name.startswith('_plus') or sym.name.find('conv')>=0:
      return sym
  return None

def get_adabn_params(symbol, arg_params, aux_params, bn_layer_names = ['bn1','stage4_unit3_bn3']):
  ret = {}
  for _bn in bn_layer_names:
    ret[_bn] = None
    #print(aux_params[_bn])
    bn_mean = aux_params[_bn+"_moving_mean"].asnumpy()
    bn_var = aux_params[_bn+"_moving_var"].asnumpy()
    bn_gamma = arg_params[_bn+"_gamma"].asnumpy()
    bn_beta = arg_params[_bn+"_beta"].asnumpy()
    print(bn_mean.shape)
    print(bn_var.shape)
    print(bn_gamma.shape)
    print(bn_beta.shape)
  #bn_layer_name = "bn1"
  #print(arg_params.__class__)
  #print(aux_params.__class__)

  all_layers = sym.get_internals()
  for layer in all_layers:
    #print(layer.name)
    if layer.name in ret:
      bn_input = get_bn_input_symbol(layer)
      assert bn_input is not None
      ret[layer.name] = (bn_input, layer)
      #for _sym in layer.get_children():
      #  if _sym.name in arg_params:
      #    print(_sym.name, arg_params[_sym.name])
      #  else:
      #    print(_sym.name)
  return ret


def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs


img_sz = 360
crop_sz = 320

def image_preprocess(img_full_path, loop):
  img = cv2.cvtColor(cv2.imread(img_full_path), cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  ori_shape = img.shape
  assert img.shape[2]==3

  img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_CUBIC)
  h, w, _ = img.shape
  x0 = int((w - crop_sz) / 2)
  y0 = int((h - crop_sz) / 2)

  img = img[y0:y0+crop_sz, x0:x0+crop_sz]

  if loop%2==1:
    img = np.fliplr(img)

  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)  # change to r,g,b order
  return img

imgs = []
labels = []
val_file = 'data-asv/val.lst'
_max_label = 262
max_label = 0
with open(val_file, 'r') as f:
  for line in f:
    line = line.strip()
    vec = line.split("\t")
    id = int(vec[0])
    label = int(vec[1])
    max_label = max(max_label, label)
    image_path = vec[2]
    imgs.append(image_path)
    labels.append(label)
if _max_label is not None:
  max_label = _max_label
batch_sz = 64

def apply_adabn(ctx, sym, arg_params, aux_params, imgs):
  batch_head = 0
  batch_num = 0
  adabn = get_adabn_params(sym, arg_params, aux_params)
  adabn_list = []
  sym_list = []
  X_list = []
  for k,v in adabn.iteritems():
    adabn_list.append( (k,v) )
    sym_list.append(v[0])
    X_list.append([])
    #_adabn = (k,v)
  num_bn = len(X_list)
  req = mx.symbol.Group(sym_list)
  new_auxs = dict()
  for k, v in aux_params.items():
    #print(v.__class__)
    new_auxs[k] = v.as_in_context(ctx)
  while batch_head<len(imgs):
    prt("processing batch %d" % batch_num)
    current_batch_sz = min(batch_sz, len(imgs)-batch_head)
    input_blob = np.zeros((current_batch_sz,3,crop_sz,crop_sz))
    #print batch_head
    idx = 0
    ids = []
    for idx in xrange(current_batch_sz):
      index = batch_head+idx
      filename = imgs[index]
      img = image_preprocess(filename, 0)
      input_blob[idx,:,:,:] = img

    arg_params["data"] = mx.nd.array(input_blob, ctx)
    arg_params["softmax_label"] = mx.nd.empty((current_batch_sz,), ctx)
    exe = req.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)
    #print(exe.outputs)
    for i in xrange(num_bn):
      net_out = exe.outputs[i].asnumpy()
      net_out = np.mean(net_out, axis=(2,3))
      #print(net_out.shape)
      for d in xrange(net_out.shape[0]):
        X_list[i].append(net_out[d])
    batch_num+=1
    batch_head+=current_batch_sz
    #if batch_num==30:
    #  break

  for i in xrange(num_bn):
    X = X_list[i]
    X = np.array(X, dtype=np.float32)
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    print(mean.shape)
    print(var.shape)
    name = adabn_list[i][0]
    new_auxs[name+"_moving_mean"] = mx.nd.array(mean, ctx)
    new_auxs[name+"_moving_var"] = mx.nd.array(var, ctx)
  return arg_params, new_auxs

def do_eval(ctx, sym, arg_params, aux_params, imgs, labels):
  batch_head = 0
  batch_num = 0
  X = None
  while batch_head<len(imgs):
    prt("processing batch %d" % batch_num)
    current_batch_sz = min(batch_sz, len(imgs)-batch_head)
    input_blob = np.zeros((current_batch_sz,3,crop_sz,crop_sz))
    #print batch_head
    idx = 0
    ids = []
    for idx in xrange(current_batch_sz):
      index = batch_head+idx
      filename = imgs[index]
      img = image_preprocess(filename, 0)
      input_blob[idx,:,:,:] = img


    arg_params["data"] = mx.nd.array(input_blob, ctx)
    arg_params["softmax_label"] = mx.nd.empty((current_batch_sz,), ctx)
    exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)
    #print(exe.outputs)
    net_out = exe.outputs[0].asnumpy()
    for idx in xrange(current_batch_sz):
      index = batch_head+idx
      gt_label = labels[index]
      probs = net_out[idx,:]
      score = np.squeeze(probs)
      if X is None:
        X = np.zeros( (len(imgs), len(score)), dtype=np.float32 )
      X[index,:] += score
    batch_num+=1
    batch_head+=current_batch_sz
  acc = 0.0
  for i in xrange(len(imgs)):
    score = X[i]
    sort_index = np.argsort(score)[::-1]
    prediction = sort_index[0]
    if prediction==labels[i]:
      acc+=1.0
  acc /= len(imgs)
  print("acc %f" % acc)

prefix = 'model/asvauto-resnet-152'
epoch = int(sys.argv[1])
gpu_id = int(sys.argv[2]) #GPU ID for infer
print("input epoch", epoch)
print("input gpu_id", gpu_id)
ctx = mx.gpu(gpu_id)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
print('cal adabn params...')
arg_params, aux_params = apply_adabn(ctx, sym, arg_params, aux_params, imgs)
print('eval..')
do_eval(ctx, sym, arg_params, aux_params, imgs, labels)




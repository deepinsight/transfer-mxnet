import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import random
import json
import mxnet as mx
import math




def calculate_distance2(data1, data2):
  #total_num = batch_size
  _data1 = mx.symbol.expand_dims(data1, axis=1) # B,1,C
  spread_distance2 = mx.symbol.broadcast_sub(_data1, data2) #B,B,C
  spread_distance2 = mx.symbol.reshape(spread_distance2, shape=(-3,-2))  #B*B,C
  spread_distance2 = mx.symbol.square(spread_distance2)
  distance2 = mx.symbol.sum(spread_distance2, axis=1) # B*B,
  return distance2

def calculate_distance1(data1, data2):
  distance = data1 - data2
  #distance = mx.symbol.reshape(distance, shape=(1,-2))
  distance2 = mx.symbol.square(distance) #1,C
  distance2 = mx.symbol.sum(distance2, axis=1) #1,
  return distance2

def multi_kernel_distance(data1, data2, batch_size, kernel_num, base_gamma):
  assert batch_size==1
  #distance2 = calculate_distance2(data1, data2)
  distance2 = calculate_distance1(data1, data2)
  kernel_mul = 2.0
  coef = kernel_num*-0.5
  #times = math.pow(kernel_mul, kernel_num/2.0)
  ks = []
  for i in xrange(kernel_num):
    kernel_gamma = base_gamma * math.pow(kernel_mul, coef)
    ks.append(mx.symbol.exp(distance2*-1.0/kernel_gamma))
    coef += 1.0
  ret = mx.symbol.add_n(*ks) #(b*b,)
  return ret


def multi_kernel_distance2(distance2, kernel_num, base_gamma):
  kernel_mul = 2.0
  coef = kernel_num*-0.5
  ks = []
  for i in xrange(kernel_num):
    kernel_gamma = base_gamma * math.pow(kernel_mul, coef)
    ks.append(mx.symbol.exp(distance2*-1.0/kernel_gamma))
    coef += 1.0
  ret = mx.symbol.add_n(*ks) #(b*b,)
  return ret


def mmd(data, fc, args):
  batch_size = args.batch_per_gpu
  print('mmd batch_size', batch_size)
  assert batch_size%4==0

  #static params
  source_num = batch_size/2
  target_num = source_num
  total_num = batch_size
  data_kernel_num = 4
  label_kernel_num = 1
  group_num = source_num/2
  #data_gamma = 0.0
  label_gamma = 1.3


  softmax = mx.symbol.softmax(data=fc)
  gt_label = mx.symbol.Variable('softmax_label')
  #max_label = mx.symbol.max(gt_label)
  #source_data = mx.symbol.slice_axis(data, axis=0, begin=0, end=source_num)
  #target_data = mx.symbol.slice_axis(data, axis=0, begin=source_num, end=total_num)
  #source_softmax = mx.symbol.slice_axis(softmax, axis=0, begin=0, end=source_num)
  #target_softmax = mx.symbol.slice_axis(softmax, axis=0, begin=source_num, end=total_num)
  distance2 = calculate_distance2(data, data)
  bandwidth = mx.symbol.sum(distance2)
  #data_gamma = (total_num * total_num - total_num)/bandwidth
  data_gamma = bandwidth/(total_num * total_num - total_num)
  k_list = []
  #unbiased mmd
  for i in xrange(group_num):
    xs1 = mx.symbol.slice_axis(data, axis=0, begin=i*2, end=i*2+1)
    xs2 = mx.symbol.slice_axis(data, axis=0, begin=i*2+1, end=i*2+2)
    xt1 = mx.symbol.slice_axis(data, axis=0, begin=source_num+i*2, end=source_num+i*2+1)
    xt2 = mx.symbol.slice_axis(data, axis=0, begin=source_num+i*2+1, end=source_num+i*2+2)
    ys1 = mx.symbol.slice_axis(softmax, axis=0, begin=i*2, end=i*2+1)
    ys2 = mx.symbol.slice_axis(softmax, axis=0, begin=i*2+1, end=i*2+2)
    yt1 = mx.symbol.slice_axis(softmax, axis=0, begin=source_num+i*2, end=source_num+i*2+1)
    yt2 = mx.symbol.slice_axis(softmax, axis=0, begin=source_num+i*2+1, end=source_num+i*2+2)

    k_x = multi_kernel_distance(xs1, xs2, 1, data_kernel_num, data_gamma)
    k_y = multi_kernel_distance(ys1, ys2, 1, label_kernel_num, label_gamma)
    k = k_x*k_y
    if args.use_dan:
      k_list.append(k_x)
    else:
      k_list.append(k)

    k_x = multi_kernel_distance(xt1, xt2, 1, data_kernel_num, data_gamma)
    k_y = multi_kernel_distance(yt1, yt2, 1, label_kernel_num, label_gamma)
    k = k_x*k_y
    if args.use_dan:
      k_list.append(k_x)
    else:
      k_list.append(k)

    k_x = multi_kernel_distance(xs1, xt2, 1, data_kernel_num, data_gamma)*-1.0
    k_y = multi_kernel_distance(ys1, yt2, 1, label_kernel_num, label_gamma)
    k = k_x*k_y
    if args.use_dan:
      k_list.append(k_x)
    else:
      k_list.append(k)

    k_x = multi_kernel_distance(xt1, xs2, 1, data_kernel_num, data_gamma)*-1.0
    k_y = multi_kernel_distance(yt1, ys2, 1, label_kernel_num, label_gamma)
    k = k_x*k_y
    if args.use_dan:
      k_list.append(k_x)
    else:
      k_list.append(k)

  mmd_loss = mx.symbol.add_n(*k_list)/group_num
  net = mx.symbol.SoftmaxOutput(data=fc, label = gt_label, use_ignore=True, ignore_label=args.null_label, name='softmax')
  if args.train_stage>0:
    grad_scale = 1.0 if args.use_dan else 2.0
    mmd = mx.symbol.MakeLoss(mmd_loss, grad_scale=grad_scale)
    net = mx.symbol.Group([net,mmd])
  return net



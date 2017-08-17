import mxnet as mx
import random
import sys
from mxnet.io import DataBatch, DataIter
import numpy as np

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-dir', type=str, default='./data',
                      help='the data dir')
    #data.add_argument('--data-train', type=str, help='the training data')
    #data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    #data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    data.add_argument('--dtype', type=str, default='float32',
                      help='data type: float32 or float16')
    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=1,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max change of aspect ratio, whose range is [0, 1]')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    return aug

def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)



class StIter(mx.io.DataIter):
    def __init__(self, source_iter, target_iter, data_shape, batch_size, gpu_num, data_name='data', label_name='softmax_label'):
      #self.iters = iter_list
      self.source_iter = source_iter
      self.target_iter = target_iter
      self.data_name = data_name
      self.label_name = label_name
      self.data_shape = data_shape
      self.batch_size = batch_size
      self.gpu_num = gpu_num
      #print(source_iter.provide_data)
      self.provide_data = [(self.data_name, (batch_size,) + self.data_shape)]
      self.provide_label = [(self.label_name, (batch_size,) )]
      self.bs_in_each_part = self.batch_size/2/gpu_num


    def part_slice(self, d, i):
      _d = mx.ndarray.slice_axis(d, axis=0, begin=i*self.bs_in_each_part, end=(i+1)*self.bs_in_each_part)
      return _d

    def next(self):
      source_batch = self.source_iter.next()
      target_batch = self.target_iter.next()
      source_data = source_batch.data[0]
      target_data = target_batch.data[0]
      source_label = source_batch.label[0]
      target_label = target_batch.label[0]
      data_parts = []
      label_parts = []
      for i in xrange(self.gpu_num):
        _source_data = self.part_slice(source_data, i)
        _target_data = self.part_slice(target_data, i)
        _source_label = self.part_slice(source_label, i)
        _target_label = self.part_slice(target_label, i)
        #_target_label = mx.ndarray.ones_like(_target_label)*9999
        data_parts.append(_source_data)
        data_parts.append(_target_data)
        label_parts.append(_source_label)
        label_parts.append(_target_label)
      self.data = mx.ndarray.concat(*data_parts, dim=0)
      self.label = mx.ndarray.concat(*label_parts, dim=0)
      #self.data = mx.ndarray.concat( source_data, target_data, dim=0)
      #print(self.data.shape)
      #self.label = mx.ndarray.concat(source_batch.label[0], target_batch.label[0], dim=0)
      #provide_data = [(self.data_name, self.data.shape)]
      #provide_label = [(self.label_name, self.label.shape)]
      #print(self.label.shape)
      #return {self.data_name  :  [self.data],
      #        self.label_name :  [self.label]}
      return DataBatch(data=(self.data,),
                       label=(self.label,))

    def reset(self):
      self.source_iter.reset()
      self.target_iter.reset()
    #@property
    #def provide_data(self):
    #  #return [(k, self.data.shape)]
    #  return [mx.io.DataDesc(self.data_name, self.data.shape, self.data.dtype)]
    #@property
    #def provide_label(self):
    #  #return [(k, self.label.shape)]
    #  return [mx.io.DataDesc(self.label_name, self.label.shape, self.label.dtype)]

def read_lst(file):
  ret = []
  with open(file, 'r') as f:
    for line in f:
      vec = line.strip().split("\t")
      label = int(vec[1])
      img_path = vec[2]
      ret.append( [label, img_path] )
  return ret

def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    dtype = np.float32;
    if 'dtype' in args:
        if args.dtype == 'float16':
            dtype = np.float16
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    #print(rank, nworker, args.batch_size)
    train_resize = int(image_shape[1]*1.6)
    if args.train_stage==0:
      imglist = read_lst(args.data_dir+"/source.lst")
      print(len(imglist))
      assert len(imglist)>0
      imglist2 = read_lst(args.data_dir+"/target.lst")
      assert len(imglist2)>0
      imglist2 = [x for x in imglist2 if x[0]!=args.null_label]
      imglist += imglist2
      print(len(imglist))
      train = mx.img.ImageIter(
          label_width         = args.label_width,
          path_root	          = '', 
          #path_imglist        = args.data_dir+"/train_part.lst",
          imglist             = imglist,
          data_shape          = image_shape,
          batch_size          = args.batch_size,
          resize              = train_resize,
          rand_crop           = True,
          rand_resize         = True,
          rand_mirror         = True,
          shuffle             = True,
          brightness          = 0.4,
          contrast            = 0.4,
          saturation          = 0.4,
          pca_noise           = 0.1,
          num_parts           = nworker,
          part_index          = rank)
    else:
      source = mx.img.ImageIter(
          label_width         = args.label_width,
          path_root	          = '', 
          path_imglist        = args.data_dir+"/source.lst",
          data_shape          = image_shape,
          batch_size          = args.batch_size/2,
          resize              = train_resize,
          rand_crop           = True,
          rand_resize         = True,
          rand_mirror         = True,
          shuffle             = True,
          brightness          = 0.4,
          contrast            = 0.4,
          saturation          = 0.4,
          pca_noise           = 0.1,
          data_name           = 'data_source',
          label_name          = 'label_source',
          num_parts           = nworker,
          part_index          = rank)
      target_file = args.data_dir+"/target.lst"
      target = mx.img.ImageIter(
          label_width         = args.label_width,
          path_root	          = '', 
          path_imglist        = target_file,
          data_shape          = image_shape,
          batch_size          = args.batch_size/2,
          resize              = train_resize,
          rand_crop           = True,
          rand_resize         = True,
          rand_mirror         = True,
          shuffle             = True,
          brightness          = 0.4,
          contrast            = 0.4,
          saturation          = 0.4,
          pca_noise           = 0.1,
          data_name           = 'data_target',
          label_name          = 'label_target',
          num_parts           = nworker,
          part_index          = rank)
      train = StIter(source, target, image_shape, args.batch_size, args.gpu_num)
    val = mx.img.ImageIter(
        label_width         = args.label_width,
        path_root	          = '', 
        path_imglist        = args.data_dir+"/val.lst",
        batch_size          = args.batch_size,
        data_shape          =  image_shape,
        resize		      = int(image_shape[1]*1.125), 
        rand_crop       = False,
        rand_resize     = False,
        rand_mirror     = False,
        num_parts       = nworker,
        part_index      = rank)
    return (train, val)

def test_st():
  image_shape = (3,320,320)
  source = mx.img.ImageIter(
      label_width         = 1,
      #path_root	    = 'data/', 
      #path_imglist         = args.data_train,
      path_imgrec         = './data-asv/train.rec',
      path_imgidx         = './data-asv/train.idx',
      #data_shape          = (3, 320, 320),
      data_shape          = image_shape,
      batch_size          = 16/2,
      rand_crop           = True,
      rand_resize         = True,
      rand_mirror         = True,
      shuffle             = True,
      brightness          = 0.4,
      contrast            = 0.4,
      saturation          = 0.4,
      pca_noise           = 0.1,
      data_name           = 'data_source',
      label_name          = 'label_source',
      num_parts           = 1,
      part_index          = 0)
  target = mx.img.ImageIter(
      label_width         = 1,
      #path_root	    = 'data/', 
      #path_imglist         = args.data_train,
      path_imgrec         = './data-asv/val.rec',
      path_imgidx         = './data-asv/val.idx',
      #data_shape          = (3, 320, 320),
      data_shape          = image_shape,
      batch_size          = 16/2,
      rand_crop           = True,
      rand_resize         = True,
      rand_mirror         = True,
      shuffle             = True,
      brightness          = 0.4,
      contrast            = 0.4,
      saturation          = 0.4,
      pca_noise           = 0.1,
      data_name           = 'data_target',
      label_name          = 'label_target',
      num_parts           = 1,
      part_index          = 0)
  train = StIter(source, target, image_shape, 64, 4)
  for batch in train:
    data = batch.data
    label = batch.label
    print(data)
    print(label)
    #print(label.asnumpy())


if __name__ == '__main__':
  test_st()



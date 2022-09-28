import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.io as sio
import time

def arg_scope(is_training=True):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training, 'decay': 0.94, 'epsilon': 1e-5}):
        
        
        

        #with tf.variable_scope(scope, 'densenet_2', [inputs]) as sc:
        #    end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],padding='SAME') as arg_sc:
            return arg_sc


def googlenet2result(inputs, is_training=True,scope=None, useGradCam=False):
                net1,classresult1, output1 = inception_v3(inputs, is_training=is_training, scope=scope + 'T2')

                net = tf.image.resize_images(output1,(14,14))
                net = slim.conv2d(net,512,[3,3],rate=1,scope='upconv_1')
                net = tf.image.resize_images(net,(28,28))
                net = slim.conv2d(net,128,[3,3],rate=1,scope='upconv_2')
                net = tf.image.resize_images(net,(56,56))
                net = slim.conv2d(net,32,[3,3],rate=1,scope='upconv_4')
                net = tf.image.resize_images(net,(112,112))
                net = slim.conv2d(net,4,[3,3],rate=1,scope='upconv_5')
                net = tf.image.resize_images(net,(224,224))
                net = slim.conv2d(net, 1, [1, 1], stride=1, scope='enrty_conv1')
                imageresult1 = tf.nn.sigmoid(net)


                return classresult1, imageresult1



def deconv2d(input_, output_dim, ks=3, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def upsample(x1, output_channels, in_channels,w,h):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, [-1,w,h,1], strides=[1, pool_size, pool_size, 1])
 
    return deconv


def inception_v3_base(inputs, scope=None):
  '''
  Args:
  inputs：输入的tensor
  scope：包含了函数默认参数的环境
  '''
  end_points = {} # 定义一个字典表保存某些关键节点供之后使用

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 对三个参数设置默认值
                        stride=1, padding='VALID'):
                        # stride = 1, padding = 'SAME'):
     #with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(0.0005)):

      #  因为使用了slim以及slim.arg_scope，我们一行代码就可以定义好一个卷积层
      #  相比AlexNet使用好几行代码定义一个卷积层，或是VGGNet中专门写一个函数定义卷积层，都更加方便
      #
      # 正式定义Inception V3的网络结构。首先是前面的非Inception Module的卷积层
      # slim.conv2d函数第一个参数为输入的tensor，第二个是输出的通道数，卷积核尺寸，步长stride，padding模式

      #一共有5个卷积层，2个池化层，实现了对图片数据的尺寸压缩，并对图片特征进行了抽象
      #   20 x 224 x 224 x 3
      net = slim.conv2d(inputs, 32, [3, 3],
                        stride=2, scope='Conv2d_1a_3x3')
                                                          # 20 x 111 x 111 x 32
      net = slim.conv2d(net, 32, [3, 3],
                        scope='Conv2d_2a_3x3')
                                                 # 20 x 109 x 109 x 32
      net = slim.conv2d(net, 64, [3, 3], padding='SAME',
                        scope='Conv2d_2b_3x3')
                                                # 20 x 109 x 109 x 64
      net = slim.max_pool2d(net, [3, 3], stride=2,
                            scope='MaxPool_3a_3x3')
                                               # 20 x 54 x 54 x 64
      net = slim.conv2d(net, 80, [1, 1],
                        scope='Conv2d_3b_1x1')
                                              # 20 x 54 x 54 x 80
      net = slim.conv2d(net, 192, [3, 3],
                        scope='Conv2d_4a_3x3')
                                                # 20 x 54 x 54 x 192
      net = slim.max_pool2d(net, [3, 3], stride=2,
                            scope='MaxPool_5a_3x3')
                                                     # 20 x 25 x 25 x 192

    '''
    三个连续的Inception模块组，三个Inception模块组中各自分别有多个Inception Module，这部分是Inception Module V3
    的精华所在。每个Inception模块组内部的几个Inception Mdoule结构非常相似，但是存在一些细节的不同
    '''
    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 设置所有模块组的默认参数
                        stride=1, padding='SAME'): # 将所有卷积层、最大池化、平均池化层步长都设置为1
     #with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(0.0005)):
      # 第一个模块组包含了三个结构类似的Inception Module
      '''    
--------------------------------------------------------    
      第一个Inception组   一共三个Inception模块
      '''
      with tf.variable_scope('Mixed_5b'): # 第一个Inception Module名称。Inception Module有四个分支
        # 第一个分支64通道的1*1卷积
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                                                                         # 25x25x64
        # 第二个分支48通道1*1卷积，链接一个64通道的5*5卷积
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')  # 25x25x48
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')   # 25x25x64

        # 第三个分支64通道1*1卷积,96的3*3,再接一个3*3
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3') # 25x25x64

        # 第四个分支64通道3*3平均池化,32的1*1
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')  # 25x25x32
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 将四个分支的输出合并在一起（第三个维度合并，即输出通道上合并）

      '''
      因为这里所有层步长均为1，并且padding模式为SAME，所以图片尺寸不会缩小，但是通道数增加了。四个分支通道数之和
      64+64+96+32=256，最终输出的tensor的图片尺寸为35*35*256
      '''

      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


      with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        # netB = net
      # 64+64+96+64 = 288
      # mixed_1: 35 x 35 x 288

        '''    
        第一个Inception组结束  一共三个Inception模块 输出为:35*35*288
----------------------------------------------------------------------    
        第二个Inception组   共5个Inception模块
        '''

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')   #12*12*384
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')   #12*12*96
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')  #12*12*288
        net = tf.concat([branch_0, branch_1, branch_2], 3)
      # 384+96+288 = 768
      # mixed_3:  12 x 12 x 768

      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7') # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)



      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          '''
          我们的网络每经过一个inception module，即使输出尺寸不变，但是特征都相当于被重新精炼了一遍，
          其中丰富的卷积和非线性化对提升网络性能帮助很大。
          '''
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)



        '''    
        第二个Inception组结束  一共五个Inception模块 输出为:17*17*768
----------------------------------------------------------------------    
        第三个Inception组   共3个Inception模块(带分支)
        '''
      # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
      end_points['Mixed_6e'] = net

      # 第三个inception模块组包含了三个inception module

      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') # 5*5*320
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') #8*8*192
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3) # 输出图片尺寸被缩小，通道数增加，tensor的总size在持续下降中
      # 320+192+768 = 1280
      # mixed_8:  5x 5x 1280.


      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 输出通道数增加到2048
      # 320+(384+384)+(384+384)+192 = 2048
      # mixed_9: 5x 5x 2048.



      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 320+(384+384)+(384+384)+192 = 2048
      # mixed_10: 5x 5x 2048.

      return net, end_points
      #Inception V3网络的核心部分，即卷积层部分就完成了
      '''
      设计inception net的重要原则是图片尺寸不断缩小，inception模块组的目的都是将空间结构简化，同时将空间信息转化为
      高阶抽象的特征信息，即将空间维度转为通道的维度。降低了计算量。Inception Module是通过组合比较简单的特征
      抽象（分支1）、比较比较复杂的特征抽象（分支2和分支3）和一个简化结构的池化层（分支4），一共四种不同程度的
      特征抽象和变换来有选择地保留不同层次的高阶特征，这样最大程度地丰富网络的表达能力。
      '''



# V3最后部分
# 全局平均池化、Softmax和Auxiliary Logits
def inception_v3(inputs,
                 num_classes=2, # 最后需要分类的数量（比赛数据集的种类数）
                 is_training=True, # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
                 dropout_keep_prob=0.8, # 节点保留比率
                 prediction_fn=slim.softmax, # 最后用来分类的函数
                 spatial_squeeze=True, # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
                 reuse=None, # 是否对网络和Variable进行重复使用
                 scope='InceptionV3'): # 包含函数默认参数的环境

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], # 定义参数默认值
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], # 定义标志默认值
                        is_training=is_training):
      # 拿到最后一层的输出net和重要节点的字典表end_points
      net1, end_points = inception_v3_base(inputs, scope=scope)  # 用定义好的函数构筑整个网络的卷积部分

      # Auxiliary logits作为辅助分类的节点，对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'): # 将卷积、最大池化、平均池化步长设置为1
       #with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(0.0005)):

        aux_logits = end_points['Mixed_6e'] # 通过end_points取到Mixed_6e
        # end_points['Mixed_6e']  --> 12x12x768
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
                    aux_logits, [5, 5], stride=3, padding='VALID',
                    scope='AvgPool_1a_5x5')  #3x3x768

          aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                   scope='Conv2d_1b_1x1')  #3x3x128


          aux_logits = slim.conv2d(
              aux_logits, 16, [3, 3],
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              padding='VALID', scope='Conv2d_2a_5x5')    #1x1x16

          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
              scope='Conv2d_2b_1x1')   # 1*1*2

          if spatial_squeeze: # tf.squeeze消除tensor中前两个为1的维度。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits # 最后将辅助分类节点的输出aux_logits储存到字典表end_points中
                    #20*2
      # 处理正常的分类预测逻辑
      # Final pooling and prediction
      with tf.variable_scope('Logits'):

        net = slim.avg_pool2d(net1, [5, 5], padding='VALID',
                              scope='AvgPool_1a_8x8')

        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net

        # 激活函数和规范化函数设为空
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze: # tf.squeeze去除输出tensor中维度为1的节点
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')


  return net,logits,net1 # 最后返回logits和包含辅助节点的end_points

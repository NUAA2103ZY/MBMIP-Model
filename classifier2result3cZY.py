# -*-coding: utf-8 -*-
"""
    @Project: triple_path_networks
    @File   : UNet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-01-24 11:18:15
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.io as sio
import time
import hdf5storage
import os

from utils import inceptionmodel224
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class GBM_classify:
    def __init__(self, sess):
        self.matname, self.savename = [None] * 2
        self.trainimg, self.trainlabel = [None] * 2
        self.validimg, self.validlabel = [None] * 2
        self.testimg, self.testlabel, self.test_flag = [None] * 3
        self.imgwidth, self.imgheight, self.channel, self.trainimgnum, self.validimgnum, self.testimgnum = [None] * 6
        self.batch_size, self.keep_prob = [None] * 2
        self.lamda, self.learning_rate, self.learning_rate_decay = [None] * 3
        self.is_training = True
        self.input, self.target, self.output, self.prediction, self.pre_uint8 = [None] * 5
        self.optimizer = None
        self.global_step , self.train_summar_step, self.evaluate_step, self.save_model_step = [None] * 4
        self.loss, self.accuracy = [None] * 2
        self.loss_mean, self.regularization_loss, self.total_loss = [None] * 3
        self.sess = sess
        self.saver, self.summary, self.train_writer, self.valid_writer = [None] * 4
        self.savefolder, self.saveflag, self.modelsavefolder = [None] * 3
        self.train_logdir, self.valid_logdir = [None] * 2
        self.display_loss, self.display_result, self.testmodelname = [None] * 3
        self.epoch, self.iter, self.global_iter, self.flag = [None] * 4

    def loadtraindata(self, name, img_name, mask_name):
        print("load traindata ...")
        # if self.matname == None:
        self.matname = name
        dataset = hdf5storage.loadmat(self.matname)
        # dataset = sio.loadmat(self.matname)
        self.trainimg = dataset[img_name]
        self.trainlabel = dataset[mask_name]
        self.imgheight = np.size(self.trainimg, 1)
        self.imgwidth = np.size(self.trainimg, 2)

        self.trainimgnum = np.size(self.trainimg, 0)
        print("load traindata finished")

    def loadvaliddata(self, name, img_name, mask_name):
        print("load validdata ...")
        # if self.matname == None:
        self.matname = name
        dataset = hdf5storage.loadmat(self.matname)
        # dataset = sio.loadmat(self.matname)
        self.validimg = dataset[img_name]
        self.validlabel = dataset[mask_name]
        self.validimgnum = np.size(self.validimg, 0)
        print("load validdata finished")

    def loadtestdata(self, name, img_name, mask_name):
        print("load testdata ...")
        # if self.matname == None:
        self.matname = name
        dataset = hdf5storage.loadmat(self.matname)
        # dataset = sio.loadmat(self.matname)
        self.testimg = dataset[img_name]
        self.testlabel = dataset[mask_name]
        self.testimgnum = np.size(self.testimg, 0)
        print("load testdata finished")

    def test_softmax_focal_ce_3(self, n_classes, gamma, alpha, logits, label):
        epsilon = 1.e-8
        # y_true and y_pred
        # y_true = tf.one_hot(label, n_classes)
        y_true = label
        probs = tf.nn.softmax(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        if alpha != 0.0:
            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        else:
            alpha_t = tf.ones_like(y_true)
        xent = tf.multiply(y_true, -tf.log(y_pred))
        focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
        reduced_fl = tf.reduce_max(focal_xent, axis=1)
        return tf.reduce_mean(reduced_fl)

    def L1loss(self, in_, target):
     return tf.reduce_mean(tf.abs(in_ - target))


    def L2loss(self, in_, target):
     return tf.reduce_mean((in_-target)**2)

    def build(self, learning_rate):
        print("build model ...")

        #self.learning_rate = learning_rate

        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.imgheight, self.imgwidth, 3],
                                         name='input')
            self.target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2],
                                         name='output')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        with tf.name_scope('output_prediction'):
            self.prediction, self.imageresult = inceptionmodel224.googlenet2result(inputs=self.input,is_training=self.is_training,scope='inceptionv3')
            self.net_softmax = tf.nn.softmax(self.prediction)
        with tf.name_scope('loss'):
            self.target_reshape = tf.expand_dims(self.target, -1, 'target_reshape')
            self.imgloss1 = self.L1loss(tf.squeeze(self.imageresult,-1),self.input[:,:,:,0])+self.L2loss(tf.squeeze(self.imageresult,-1),self.input[:,:,:,0])
            self.lossfocal= self.test_softmax_focal_ce_3(3, 2, 0.25, self.prediction, self.target)
            self.loss = slim.losses.softmax_cross_entropy(self.prediction, self.target)
            slim.losses.add_loss(self.lossfocal)
            self.total_loss = self.lossfocal + 2*self.imgloss1

        with tf.name_scope('accuracy'):
            
            label_max, pred_max = tf.argmax(self.target, 1), tf.argmax(self.net_softmax, 1)
            correct_prediction = tf.equal(label_max, pred_max)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        with tf.name_scope('learning_step'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            tf.summary.scalar('learning_rate', self.learning_rate)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.total_loss,
                                                          self.optimizer,
                                                          global_step=self.global_step,
                                                          update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=self.max_model_to_keep)
        with tf.name_scope('summary'):
            tf.summary.scalar('Loss_all', self.total_loss)
            tf.summary.scalar('Accuracy', self.accuracy)

            self.summary = tf.summary.merge_all()

            self.train_writer = tf.summary.FileWriter(self.train_logdir, self.sess.graph)
            self.valid_writer = tf.summary.FileWriter(self.valid_logdir)

        print("build model finish")

    def train(self, max_epoch):
        print("start training ...")
        global_iter = 0
        self.epoch = max_epoch
        self.iter = self.trainimgnum // self.batch_size
        residual_num = self.trainimgnum % self.batch_size
        if residual_num > 0:
            self.iter = self.iter + 1
        max_iter = self.iter * self.epoch 

        if self.display_loss:
            fig1 = plt.figure(num=1, figsize=(12, 5))
            axe1 = fig1.add_subplot(1, 2, 1)
            axe1.set_xlabel('iteration number')
            axe1.set_ylabel('loss')
            axe1.set_title('Loss')
            axe2 = fig1.add_subplot(1, 2, 2)
            axe2.set_xlabel('iteration number')
            axe2.set_ylabel('accuracy')
            axe2.set_title('Accuracy')
            plt.draw()
         

        loss_collector = []
        accuracy_collector = []
        iter_collector = []
        loss_collector2 = []
        accuracy_collector2 = []
        iter_collector2 = []
        maxaccuracy=0
        self.flag = 1
        loss_epoch_sum, loss_ave_epoch_sum, accuracy_epoch_sum = 0.0, 0.0, 0.0
        loss_eval_sum, loss_ave_eval_sum, accuracy_eval_sum = 0.0, 0.0, 0.0

        for epoch in range(self.epoch):
            iter = 0
            #if epoch < 3:
            #    lr = 0.00001
            #else:
            lr = 0.00001 * (self.epoch - epoch)/ (self.epoch - 5)
            if lr < 0.0000001:
                lr = 0.0000001
            sample_index = np.random.choice(self.trainimgnum, size=self.trainimgnum, replace=False)
            if residual_num > 0:
                sample_index = np.append(sample_index,
                                         np.random.choice(self.trainimgnum, size=self.batch_size-residual_num, replace=False))
            for iter in range(self.iter):

                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                indices = sample_index[start_index:end_index]
                train_input = self.trainimg[indices, :, :, :]
                train_target = self.trainlabel[indices, :]
                feed_dict={self.input: train_input, self.target: train_target, self.is_training: True, self.learning_rate:lr}
                _, step, total_loss, accuracy = self.sess.run([self.train_op, self.global_step, self.total_loss, self.accuracy],
                                                    feed_dict=feed_dict)
                global_iter = global_iter + 1
                self.global_iter = global_iter

                #_, loss, accuracy = model.sess.run([model.learn_step, model.loss, model.accuracy],
                #                                   feed_dict=feed_dict)
                global_iter = global_iter + 1
                loss_ave = total_loss/self.batch_size
                loss_epoch_sum = loss_epoch_sum + total_loss
                loss_ave_epoch_sum = loss_ave_epoch_sum + loss_ave
                accuracy_epoch_sum = accuracy_epoch_sum + accuracy

                loss_eval_sum = loss_eval_sum + total_loss
                loss_ave_eval_sum = loss_ave_eval_sum + loss_ave
                accuracy_eval_sum = accuracy_eval_sum + accuracy

                ###SUMMARY
                if global_iter%self.train_summar_step==0:
                    train_summaryi = tf.Summary()
                    train_summaryi.ParseFromString(self.sess.run(self.summary, feed_dict=feed_dict))
                    loss_eval_sum /= self.train_summar_step
                    loss_ave_eval_sum /= self.train_summar_step
                    accuracy_eval_sum /= self.train_summar_step
                    train_summaryi.value.add(tag="loss/loss", simple_value=loss_eval_sum)
                    train_summaryi.value.add(tag="loss/loss_average", simple_value=loss_ave_eval_sum)
                    train_summaryi.value.add(tag="loss/accuracy", simple_value=accuracy_eval_sum)
                    self.train_writer.add_summary(train_summaryi, global_iter)
                    print("    train --- loss: %.2f, loss average: %.2f, accuracy: %.2f" % (
                    loss_eval_sum, loss_ave_eval_sum, accuracy_eval_sum))
                    loss_eval_sum, loss_ave_eval_sum, accuracy_eval_sum = 0.0, 0.0, 0.0
                    

                ###PLOT
                if global_iter%10==0:
                    if self.display_loss:
                        loss_collector.append(loss_ave)
                        accuracy_collector.append(accuracy)
                        iter_collector.append(global_iter)

                        axe1.plot(iter_collector, loss_collector, 'b-')
                        axe2.plot(iter_collector, accuracy_collector, 'b-')
                        #plt.draw()
                        #plt.pause(0.001)
                        print(">>> iter: %d, epoch: %d, loss_all: %.3f, loss_ave: %.3f, accuracy: %.3f" % (global_iter, epoch+1, total_loss, loss_ave, accuracy))
                #### evalue every evaluate_step
                if self.evaluate_step>0:
                    if global_iter%self.evaluate_step == 0 and global_iter>0:
                        valid_loss_all, valid_loss, valid_accuracy = self.eval(data=self.validimg, label=self.validlabel,
                                                                        flag=global_iter)
                        valid_summaryi = tf.Summary(value=[tf.Summary.Value(tag="loss/loss", simple_value=valid_loss_all),
                                                            tf.Summary.Value(tag="loss/loss_average", simple_value=valid_loss),
                                                            tf.Summary.Value(tag="loss/accuracy", simple_value=valid_accuracy)])
                        self.valid_writer.add_summary(valid_summaryi, global_iter)
                        loss_collector2.append(valid_loss)
                        accuracy_collector2.append(valid_accuracy)
                        iter_collector2.append(global_iter)

                        axe1.plot(iter_collector2, loss_collector2, 'r-')
                        axe2.plot(iter_collector2, accuracy_collector2, 'r-')
                        #plt.draw()
                        #plt.pause(0.001)
                        print("    valid --- loss: %.2f, loss average: %.2f, accuracy: %.2f"%(valid_loss_all, valid_loss, valid_accuracy))
                        
                #### save model every save_model_step
                if global_iter%self.save_model_step == 0 and global_iter>0 :
                    if valid_accuracy>maxaccuracy:
                        maxaccuracy=valid_accuracy
                    
                        self.saver.save(self.sess, self.savename, global_step=global_iter)

                                   

            print(">>> epoch %d is finished, loss %.2f, loss average: %.2f, accuracy: %.2f ..."%(epoch+1, loss_epoch_sum/self.iter, loss_ave_epoch_sum/self.iter, accuracy_epoch_sum/self.iter))
            loss_epoch_sum, loss_ave_epoch_sum, accuracy_epoch_sum = 0.0, 0.0, 0.0
            print('>>> iter: %d/%d, process rate: %.2f / %f'
                                %(global_iter, max_iter, global_iter*self.batch_size/self.trainimgnum, self.epoch)) 



              
        print("training finished for iteration reach the maximum iteration number")

        
        fig1name = 'loss_acc_' + self.saveflag + '.png'
        lossname = 'train_loss_' + self.saveflag + '.mat'
        modelname = 'Unet_model_' + self.saveflag + '.ckpt'
        if self.display_loss:
            fig1.savefig(fname=self.savefolder + '\\' + fig1name)
            print('loss and accuracy curve saved in ', self.savefolder, '\nwith name : ', fig1name)
        sio.savemat(self.savefolder + '\\' + lossname,
                    {'loss': loss_collector, 'accuracy': accuracy_collector, 'iteration': iter_collector})
        print('loss and accuracy data saved in : ', self.savefolder, '\nwith name : ', lossname)
        self.savename = self.modelsavefolder + '\\' + modelname
        self.saver.save(self.sess, self.savename, global_step=global_iter)
        print("model saved in ", self.savename)

    def eval(self, data, label, flag):
        data_num = data.shape[0]
        batch_size = self.batch_size
        batch_num = data_num // batch_size
        residual_num = data_num % batch_size
        collectbatch = batch_num
        Prediction = np.zeros([data_num, 2])

        loss_collecter = 0
        loss_all_collecter = 0
        accuracy_collecter = 0

    
        for iteration in range(batch_num):
            start_index = batch_size * iteration
            end_index = batch_size * (iteration + 1)
            indices = np.arange(start_index, end_index, 1)
            batch_data, batch_label = data[indices, :, :, :], label[indices, :] #
            feed_dict = {self.input: batch_data, self.target: batch_label, self.is_training: False}
            loss_all, accuracy, prediction= self.sess.run([self.total_loss, self.accuracy,self.net_softmax],
                                               feed_dict=feed_dict)
            loss_mean = loss_all / batch_size
            loss_collecter += loss_mean
            loss_all_collecter += loss_all
            accuracy_collecter += accuracy
            Prediction[indices, :] = prediction
        if residual_num > 0:
            collectbatch = batch_num+1
            batch_data = np.zeros([batch_size, self.imgheight, self.imgwidth, 3])#，3
            batch_label = np.zeros([batch_size, 2])
            batch_data[0:residual_num, :, :,:] = data[data_num - residual_num:data_num, :, :,:]
            batch_label[0:residual_num, :] = label[data_num - residual_num:data_num, :]
            indices = np.random.choice(data_num, batch_size - residual_num, replace=False)
            batch_data[residual_num:batch_size, :, :, :] = data[indices, :, :,:]
            batch_label[residual_num:batch_size, :] = label[indices, :]
            feed_dict = {self.input: batch_data, self.target: batch_label, self.is_training: False}
            loss_all, accuracy,prediction = self.sess.run([self.total_loss, self.accuracy,self.net_softmax],
                                               feed_dict=feed_dict)
            loss_mean = loss_all / batch_size
            loss_collecter += loss_mean
            loss_all_collecter += loss_all
            accuracy_collecter += accuracy
            Prediction[data_num - residual_num:data_num,:] = prediction[0:residual_num, :]

        loss_collecter = loss_collecter  / collectbatch
        loss_all_collecter = loss_all_collecter  / collectbatch
        accuracy_collecter = accuracy_collecter  / collectbatch
        validsavefolder = 'valid'+str(flag)+'_'+str(accuracy_collecter)+'.mat'
        sio.savemat(self.savefolder + '\\' + validsavefolder, {'prediction': Prediction})
        #print("eval finished: loss_all: %.2f, loss_mean: %.2f" % (loss_all_collecter, loss_collecter))
        print("eval finished: loss_all: %.2f, loss_mean: %.2f, accuracy: %.2f"%(loss_all_collecter, loss_collecter, accuracy_collecter))
        return loss_all_collecter, loss_collecter, accuracy_collecter


    def test(self, savefilename):
        print('start test ... ')
        print('load test data ...')
        Img = self.trainimg

        imgnum = np.size(Img, 0)
        self.imgheight = np.size(Img, 1)
        self.imgwidth = np.size(Img, 2)

        print('load test data finished ...')
        print('load unet model ...')
        self.build(learning_rate=1e-4)
        self.saver.restore(self.sess, self.testmodelname)
        print('load unet model finished ...')
        batch_num = imgnum // self.batch_size
        residual_num = imgnum % self.batch_size
        pre_num = imgnum
        Prediction = np.zeros([pre_num, 2])
        print('start test process ...')
        time_start = time.time()
        for i in range(0, batch_num, 1):
            index = np.arange(self.batch_size * i, self.batch_size * (i + 1))
            test_img = Img[index, :, :, :]
            feed_dict = {self.input: test_img, self.is_training: False}
            prediction, rebuildImg = self.sess.run([self.net_softmax, self.imageresult], feed_dict=feed_dict)

            Prediction[index,:] = prediction
            print(">>> processing : %.3f%%" % ((i + 1) * self.batch_size / imgnum * 100))

        if residual_num > 0:
            tmp = np.zeros([self.batch_size, self.imgheight, self.imgwidth,3])

            tmp[0:residual_num, :, :,:] = Img[imgnum - residual_num:imgnum, :, :,:]

            indices = np.random.choice(imgnum, self.batch_size - residual_num, replace=False)
            tmp[residual_num:self.batch_size, :, :,:] = Img[indices, :, :,:]

            feed_dict = {self.input: tmp,self.is_training: False}

            prediction = self.sess.run(self.net_softmax, feed_dict=feed_dict)

            Prediction[imgnum - residual_num:imgnum,:] = prediction[0:residual_num, :]

        time_end = time.time()
        print('test data finished ...')
        time_all = time_end - time_start
        print('test cost time : %.3f s, mean processing speed : %.3f s' % (time_all, time_all / imgnum))

        sio.savemat(savefilename, {'result': Prediction})
        print('prediction saved in : ', savefilename, ' with .mat format')

 

def main():
    current_path = sys.path[0]

    with tf.Session() as sess:
        net = GBM_classify(sess)
        net.train_logdir = current_path + '\\logs\\T2_3c_miltitask\\train'
        net.valid_logdir = current_path + '\\logs\\T2_3c_miltitask\\valid'
        if not os.path.exists(net.train_logdir):
            os.makedirs(net.train_logdir)
        if not os.path.exists(net.valid_logdir):
            os.makedirs(net.valid_logdir)
        net.train_summar_step = 5
        net.evaluate_step = 20
        net.save_model_step = 20
        net.learning_rate_decay = 0.96
        is_training = True
        if is_training:
            net.batch_size = 20
            net.max_model_to_keep = 5
            traindatafilename = current_path + '\\dataset\\channel_123.mat'
            validdatafilename = current_path + '\\dataset\\channel_123.mat'
            net.savefolder = current_path + '\\result\\channel_123'
            net.modelsavefolder = current_path + '\\model\\channel_123'
            if not os.path.exists(net.savefolder):
                os.makedirs(net.savefolder)
            if not os.path.exists(net.modelsavefolder):
                os.makedirs(net.modelsavefolder)
            net.saveflag = 'inceptionv3'
            net.savename = net.modelsavefolder + '\\' + 'inceptionv3_model_' + net.saveflag + '.ckpt'
            net.display_loss = True
            net.loadtraindata(traindatafilename, 'train_images', 'train_label')
            net.loadvaliddata(validdatafilename, 'val_images', 'val_label')
            net.build(learning_rate=1e-4)
            net.train(max_epoch=80)
        else:
            net.batch_size = 1
            net.max_model_to_keep = 5
            # net.test_flag =9960
            # net.test_flag = 7380
            # net.test_flag = 23660
            # net.test_flag = 12240
            net.test_flag = 7980
            testdatafilename = current_path + '\\dataset\\testdata.mat'
            net.loadtraindata(testdatafilename, 'IMAGES')
            save_data = current_path + '\\dataset\\testdata_' + str(net.test_flag) + '.mat'
            net.testmodelname = current_path + '\\model\\channel_123\\inceptionv3_model_inceptionv3.ckpt-' + str(
                net.test_flag)
            net.test(save_data)


if __name__ == "__main__":
    main()




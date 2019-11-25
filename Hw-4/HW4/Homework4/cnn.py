import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
import random


def get_mini_batch(im_train, label_train, batch_size):
    
    im_train = np.transpose(im_train)
    label_train = np.transpose(label_train)

    batches = im_train.shape[0]/batch_size
    offset = im_train.shape[0]%batch_size

    mini_batch_x = []
    mini_batch_y = []

    l = random.sample(range(0,im_train.shape[0]), im_train.shape[0])

    current_x = np.empty([0,196])
    current_y = np.empty([0,10])
    encode = np.zeros([10,10])
    for i in range(10):
        encode[i,i] = 1

    count = 1
    for i in l:
        if(count > batch_size):
            mini_batch_x.append(current_x)
            mini_batch_y.append(current_y)
            current_x = np.empty([0,196])
            current_y = np.empty([0,10])
            count = 1
        current_x = np.vstack((current_x,im_train[i,:]))
        current_y = np.vstack((current_y,encode[label_train[i][0],:]))
        count += 1

    mini_batch_x.append(current_x)
    mini_batch_y.append(current_y)

    return mini_batch_x, mini_batch_y


def fc(x, w, b):

    y = np.matmul(w,x)
    y = np.add(y,b)

    return y


def fc_backward(dl_dy, x, w, b, y):

    dl_dx = np.matmul(np.transpose(w),dl_dy)
    dl_db = dl_dy
    dl_dw = np.matmul(dl_dy,np.transpose(x))

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    
    l = np.linalg.norm(y_tilde-y)
    dl_dy = np.subtract(y_tilde,y)

    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    
    y_tilde = np.exp(x)/np.sum(np.exp(x), axis=0)

    dl_dy = np.subtract(y_tilde,y)
    l = -np.sum(np.log(y_tilde)*y)

    return l, dl_dy

def relu(x):

    y = np.maximum(x,0)

    return y


def relu_backward(dl_dy, x, y):

    dl_dx = np.array(dl_dy, copy = True)
    dl_dx[x <= 0] = 0;

    return dl_dx


def conv(x, w_conv, b_conv):

    y = np.zeros([x.shape[0],x.shape[1],b_conv.shape[0]])

    for x1 in range(b_conv.shape[0]):
        filter = w_conv[:,:,0,x1]
        npad = ((1, 1), (1, 1), (0, 0))
        padded_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

        for x2 in range(1,padded_x.shape[0]-1):
            for x3 in range(1,padded_x.shape[1]-1):
                current = 0
                for x4 in range(0,3):
                    for x5 in range(0,3):
                        current += padded_x[x2+x4-1,x3+x5-1,0]*filter[x4,x5]          
                y[x2-1,x3-1,x1] = current + b_conv[x1]

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    
    dl_dw = np.zeros([w_conv.shape[0],w_conv.shape[1],1,b_conv.shape[0]])
    dl_db = np.zeros((b_conv.shape[0],1))

    for x1 in range(b_conv.shape[0]):
        filter = dl_dy[:,:,x1]
        npad = ((1, 1), (1, 1), (0, 0))
        padded_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

        for x2 in range(x.shape[0]):
            for x3 in range(x.shape[1]):
                for x4 in range(0,3):
                    for x5 in range(0,3):
                        dl_dw[x4,x5,0,x1] += filter[x2,x3]*padded_x[x2+x4,x3+x5,0]
            dl_db[x1,0] += filter[x2,x3]

    return dl_dw, dl_db

def pool2x2(x):
    
    y = np.zeros([int(x.shape[0]/2),int(x.shape[1]/2),x.shape[2]])

    for x1 in range(x.shape[2]):
        for x2 in range(0,x.shape[0],2):
            for x3 in range(0,x.shape[1],2):
                y[int(x2/2),int(x3/2),x1] = max(x[x2,x3,x1],x[x2+1,x3,x1],x[x2,x3+1,x1],x[x2+1,x3+1,x1])

    return y

def pool2x2_backward(dl_dy, x, y):
    
    dl_dx = np.zeros([x.shape[0],x.shape[1],x.shape[2]])
    for x1 in range(x.shape[2]):
        for x2 in range(0,x.shape[0],2):
            for x3 in range(0,x.shape[1],2):
                current = max(x[x2,x3,x1],x[x2+1,x3,x1],x[x2,x3+1,x1],x[x2+1,x3+1,x1])
                for i in range(2):
                    for j in range(2):
                        if(x[x2+i,x3+j,x1] == current):
                            dl_dx[x2+i,x3+j,x1] = dl_dy[int(x2/2),int(x3/2),x1]


    return dl_dx


def flattening(x):
    
    y = x.flatten('F')
    y = np.expand_dims(y, axis=1)

    return y


def flattening_backward(dl_dy, x, y):

    dl_dx = np.reshape(dl_dy,x.shape, order='F')

    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):

    LR = 0.1
    decay = 0.5
    number_of_iterations = 3000
    
    batch_number = 0;
    batches = len(mini_batch_x)

    w = np.random.normal(0, 1, (10, 196))
    b = np.random.normal(0, 1, (10, 1))

    iter = 1

    while(iter <= number_of_iterations):

        if(iter% 1000 == 0):
            LR = LR*decay

        if(batch_number >= batches):
            batch_number = 0

        current_batch_x = mini_batch_x[batch_number]
        current_batch_y = mini_batch_y[batch_number]
        dL_dw = np.zeros((10,196))
        dL_db = np.zeros((10, 1))

        for i in range(current_batch_x.shape[0]):
            y = fc(np.expand_dims(current_batch_x[i], axis=1), w, b)
            l, dl_dy = loss_euclidean(y, np.expand_dims(current_batch_y[i], axis=1))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, np.expand_dims(current_batch_x[i], axis=1), w, b, y)
            dL_dw = np.add(dL_dw,dl_dw)
            dL_db = np.add(dL_db,dl_db)
        w = np.subtract(w,dL_dw * (LR/current_batch_x.shape[0]))
        b = np.subtract(b,dL_db * (LR/current_batch_x.shape[0]))

        batch_number += 1
        iter += 1 

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    
    LR = 0.9
    decay = 0.05
    number_of_iterations = 10000
    
    batch_number = 0;
    batches = len(mini_batch_x)

    w = np.random.normal(0, 1, (10, 196))
    b = np.random.normal(0, 1, (10, 1))

    iter = 1

    while(iter <= number_of_iterations):

        if(iter% 1000 == 0):
            LR = LR*decay

        if(batch_number >= batches):
            batch_number = 0

        current_batch_x = mini_batch_x[batch_number]
        current_batch_y = mini_batch_y[batch_number]
        dL_dw = np.zeros((10,196))
        dL_db = np.zeros((10, 1))

        for i in range(current_batch_x.shape[0]):
            y = fc(np.expand_dims(current_batch_x[i], axis=1), w, b)
            l, dl_dy = loss_cross_entropy_softmax(y, np.expand_dims(current_batch_y[i], axis=1))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, np.expand_dims(current_batch_x[i], axis=1), w, b, y)
            dL_dw = np.add(dL_dw,dl_dw)
            dL_db = np.add(dL_db,dl_db)

        w = np.subtract(w,dL_dw * (LR/current_batch_x.shape[0]))
        b = np.subtract(b,dL_db * (LR/current_batch_x.shape[0]))

        batch_number += 1
        iter += 1 

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    
    LR = 1.5
    decay = 0.4
    number_of_iterations = 20000
    
    batch_number = 0;
    batches = len(mini_batch_x)

    w1 = np.random.normal(0, 1, (30, 196))
    b1 = np.random.normal(0, 1, (30, 1))
    w2 = np.random.normal(0, 1, (10, 30))
    b2 = np.random.normal(0, 1, (10, 1))

    iter = 1

    while(iter <= number_of_iterations):

        if(iter % 1000 == 0):
            LR = LR*decay

        if(batch_number >= batches):
            batch_number = 0

        current_batch_x = mini_batch_x[batch_number]
        current_batch_y = mini_batch_y[batch_number]
        dL_dw1 = np.zeros((30,196))
        dL_db1 = np.zeros((30, 1))
        dL_dw2 = np.zeros((10,30))
        dL_db2 = np.zeros((10, 1))

        for i in range(current_batch_x.shape[0]):
            y1 = fc(np.expand_dims(current_batch_x[i], axis=1), w1, b1)
            y2 = relu(y1)
            y3 = fc(y2, w2, b2)
            l, dl_dy = loss_cross_entropy_softmax(y3, np.expand_dims(current_batch_y[i], axis=1))

            dl_dx, dl_dw2, dl_db2 = fc_backward(dl_dy, y2, w2, b2, y3)
            dl_dx = relu_backward(dl_dx, y1, y2)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dx, np.expand_dims(current_batch_x[i], axis=1), w1, b1, y1)

            dL_dw1 = np.add(dL_dw1,dl_dw1)
            dL_db1 = np.add(dL_db1,dl_db1)
            dL_dw2 = np.add(dL_dw2,dl_dw2)
            dL_db2 = np.add(dL_db2,dl_db2)
        w1 = np.subtract(w1,dL_dw1 *  (LR/current_batch_x.shape[0]))
        b1 = np.subtract(b1,dL_db1 *  (LR/current_batch_x.shape[0]))
        w2 = np.subtract(w2,dL_dw2 *  (LR/current_batch_x.shape[0]))
        b2 = np.subtract(b2,dL_db2 *  (LR/current_batch_x.shape[0]))

        batch_number += 1
        iter += 1 

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    
    LR = 0.2
    decay = 0.7
    number_of_iterations = 9000
    
    batch_number = 0;
    batches = len(mini_batch_x)

    w_conv = np.random.normal(0, 1, (3,3,1,3))
    b_conv = np.random.normal(0, 1, (3, 1))
    w_fc = np.random.normal(0, 1, (10, 147))
    b_fc = np.random.normal(0, 1, (10, 1))

    iter = 1

    while(iter <= number_of_iterations):

        if(iter % 1000 == 0):
            LR = LR*decay

        if(batch_number >= batches):
            batch_number = 0

        current_batch_x = mini_batch_x[batch_number]
        current_batch_y = mini_batch_y[batch_number]
        dL_dw_conv = np.zeros((3,3,1,3))
        dL_db_conv = np.zeros((3, 1))
        dL_dw_fc = np.zeros((10, 147))
        dL_db_fc = np.zeros((10, 1))

        for i in range(current_batch_x.shape[0]):
            y1 = conv(np.reshape(current_batch_x[i], (14,14,1), order='F'), w_conv, b_conv)
            y2 = relu(y1)
            y3 = pool2x2(y2)
            y4 = flattening(y3)
            y5 = fc(y4, w_fc, b_fc)

            l, dl_dy = loss_cross_entropy_softmax(y5, np.expand_dims(current_batch_y[i], axis=1))
            dl_dx, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, y4, w_fc, b_fc, y5)
            dl_dx = flattening_backward(dl_dx, y3, y4)
            dl_dx = pool2x2_backward(dl_dx, y2, y3)
            dl_dx = relu_backward(dl_dx, y1, y2)
            dl_dw_conv,dl_db_conv =  conv_backward(dl_dx, np.reshape(current_batch_x[i], (14,14,1), order='F'), w_conv, b_conv, y1)


            dL_dw_conv = np.add(dL_dw_conv,dl_dw_conv)
            dL_db_conv = np.add(dL_db_conv,dl_db_conv)
            dL_dw_fc = np.add(dL_dw_fc,dl_dw_fc)
            dL_db_fc = np.add(dL_db_fc,dl_db_fc)
        w_conv = np.subtract(w_conv,dL_dw_conv *  (LR/current_batch_x.shape[0]))
        b_conv = np.subtract(b_conv,dL_db_conv *  (LR/current_batch_x.shape[0]))
        w_fc = np.subtract(w_fc,dL_dw_fc *  (LR/current_batch_x.shape[0]))
        b_fc = np.subtract(b_fc,dL_db_fc *  (LR/current_batch_x.shape[0]))

        batch_number += 1
        iter += 1 

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':

    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()




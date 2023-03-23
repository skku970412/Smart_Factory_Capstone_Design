# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D,Conv1D, Lambda,  Dense, Flatten,MaxPooling2D,MaxPooling1D,AveragePooling1D, Dropout, BatchNormalization,SpatialDropout1D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
# from tensorflow.keras.optimizers import SGD,Adam
#from keras.optimizers import Adam
#수정 2023.02.16
from keras.optimizers import Adam

#from keras.layers import LSTM
#from keras import regularizers

from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import tensorflow as tf
import time
import json
import pickle
# from DenseNet_1DCNN_keras import DenseNet, stem, dense_block, transition_block, MLP_temp

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#hallo = tf.constant('why?' )
#print(hallo)
#2023.02.16수정 _depth제거
#def load_siamese_net_depth(input_shape = (2048,2)):
def load_siamese_net(input_shape = (2048,2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)  

    convnet = Sequential()

    # convnet.add(tf.keras.layers.SeparableConv1D(16, 64, depthwise_regularizer=regularizers.l2(0.01), pointwise_regularizer=regularizers.l2(0.01),
    #              activity_regularizer=regularizers.l1(0.01), activation='relu',strides=16, depth_multiplier=3, padding='same',input_shape=input_shape))
    # # convnet.add(tf.keras.layers.SeparableConv1D(16, 64, strides=16, depth_multiplier=5, padding='same',input_shape=input_shape))
    # # # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)
    # # # convnet.add(tf.keras.layers.ReLU(max_value=6))    
    # convnet.add(MaxPooling1D(strides=2))
    # # # convnet.add(Dropout(0.25))
    # convnet.add(tf.keras.layers.SeparableConv1D(32, 3,  strides=1, depth_multiplier=3, padding='same'))
    # # # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # # # convnet.add(tf.keras.layers.ReLU(max_value=6))
    # convnet.add(MaxPooling1D(strides=2))
    # # # convnet.add(Dropout(0.25))
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 2, strides=1, depth_multiplier=3, padding='same'))
    # # # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # # # convnet.add(tf.keras.layers.ReLU(max_value=6))
    # convnet.add(MaxPooling1D(strides=2))
    # # # convnet.add(Dropout(0.25))
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 3, strides=1, depth_multiplier=3, padding='same'))
    # # # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # # # convnet.add(tf.keras.layers.ReLU(max_value=6))
    # convnet.add(MaxPooling1D(strides=2))      
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 3,  strides=1, depth_multiplier=3 ))
    # # # convnet.add(tf.keras.layers.ReLU(max_value=6))
    # convnet.add(MaxPooling1D(strides=2))


    # convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    # convnet.add(tf.keras.layers.SeparableConv1D(16, 64, depthwise_regularizer=regularizers.l2(0.01), pointwise_regularizer=regularizers.l2(0.01),
    #              activity_regularizer=regularizers.l1(0.01), activation='relu',strides=16, depth_multiplier=6, padding='same',input_shape=input_shape))
    convnet.add(tf.keras.layers.SeparableConv1D(14, 64, activation='sigmoid',strides=8, depth_multiplier=29, padding='same',input_shape=input_shape))   # 16 , mul 29
    convnet.add(MaxPooling1D(strides=3))   
    # convnet.add(GlobalAveragePooling1D())
    convnet.add(tf.keras.layers.SeparableConv1D(32, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=3))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 2, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))                 
    convnet.add(MaxPooling1D(strides=3  ))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1,input_shape=input_shape))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
# 2023.02.19 Flatten()을 두번 한 까닭이 있나요? (질문)
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))
   						
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
#     #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#     #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    #optimizer = Adam(0.00006)
    optimizer = Adam()
    #optimizer = Adam(0.01)
#     #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer) 
    #siamese_net.compile(loss="categorical_crossentropy",optimizer=optimizer)
    # siamese_net.compile(loss=tf.keras.losses.categorical_crossentropy,
    #             optimizer=tf.keras.optimizers.Adam())
#     print('\nsiamese_net summary:')
#     siamese_net.summary()
#     print(siamese_net.count_params()) 
    return siamese_net

def load_wdcnn_net_depth(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    convnet = Sequential()
    # WDCNN 
    
    # #convnet.add(tf.keras.layers.SeparableConv1D(16, 64, activation='relu',strides=16, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # #convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(tf.keras.layers.SeparableConv1D(32, 3,  activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # #convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 2,  activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # #convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 3,  activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu', strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    # #convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))
#---------------------------------------------------------------

#call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # optimizer = Adam(0.00006)
    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
#     print('\nsiamese_net summary:')
#     siamese_net.summary()
#     print(siamese_net.count_params())
    
    return siamese_net


def load_wdcnn_net(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()
    # WDCNN 1(Ori)
    # convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))

    convnet.add(Conv1D(filters=16, kernel_size=64, strides=9, activation='sigmoid', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))                 
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))   
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
    
    
  

#### WDCNN 2
    # convnet.add(Conv1D(filters=32, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    # convnet.add(BatchNormalization())
    # convnet.add(AveragePooling1D(strides=2))

    # convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(BatchNormalization())
    # convnet.add(AveragePooling1D(strides=2))
    
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(BatchNormalization())
    # convnet.add(AveragePooling1D(strides=2))
    
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    # convnet.add(BatchNormalization())
    # convnet.add(AveragePooling1D(strides=2))
    
    # convnet.add(Conv1D(filters=64, kernel_size=3, activation='relu', strides=1))
    # convnet.add(BatchNormalization())
    # convnet.add(AveragePooling1D(strides=2))
    
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))

    #convnet.add(Dense(10, activation = 'sigmoid'))
    # print(convnet.summary())
   
    # WDCNN 3
    # convnet.add(Conv1D(filters=128, kernel_size=64, strides=1, padding='same',input_shape=input_shape))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=32, kernel_size=2, strides=1, padding='same'))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same'))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Conv1D(filters=8, kernel_size=3, strides=1))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=2))
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))
   
#### WDCNN 4 #####
    # convnet.add(Conv1D(filters=16, kernel_size=64, strides=9, padding='same',input_shape=input_shape))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=3))
    # convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=3))
    # convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='same'))                 
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=3))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=3))
    # convnet.add(Conv1D(filters=64, kernel_size=3, strides=1))
    # convnet.add(tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999))
    # convnet.add(activations='relu')
    # convnet.add(Dropout(0.25))
    # convnet.add(MaxPooling1D(strides=3))
    # convnet.add(Flatten())
    # convnet.add(Dense(100,activation='sigmoid'))

    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(10,activation='softmax')(Dropout(0.5)(encoded_cnn))
    wdcnn_net = Model(inputs=left_input,outputs=prediction_cnn)


    # optimizer = Adam(0.00006)
    optimizer = Adam() 
    wdcnn_net.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    # print('\nsiamese_net summary:')
    # cnn_net.summary()
    print(wdcnn_net.count_params()) #52806
    return wdcnn_net

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
from keras.layers import Input, LSTM, RNN, GRU, Reshape, LSTMCell
from keras.models import Model

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

def load_siamese_net1(input_shape = (2048,2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)  

    convnet = Sequential()

    convnet.add(tf.keras.layers.SeparableConv1D(14, 64, activation='relu',strides=8, depth_multiplier=29, padding='same',input_shape=input_shape))   # 16 , mul 29
    convnet.add(MaxPooling1D(strides=3))   
    convnet.add(tf.keras.layers.SeparableConv1D(32, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=3))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 2, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))               
    convnet.add(MaxPooling1D(strides=3  ))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1,input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))

    #convnet.add(Flatten())
    #convnet.add(Dense(100,activation='sigmoid'))
   						
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()

    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer) 

    return siamese_net


def load_siamese_net(input_shape = (2048,2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)  

    convnet = Sequential()

    convnet.add(tf.keras.layers.SeparableConv1D(14, 64, activation='relu',strides=8, depth_multiplier=29, padding='same',input_shape=input_shape))   # 16 , mul 29
    convnet.add(MaxPooling1D(strides=3))   
    convnet.add(tf.keras.layers.SeparableConv1D(32, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=3))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 2, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))               
    convnet.add(MaxPooling1D(strides=3  ))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1,input_shape=input_shape))
    #convnet.add(MaxPooling1D(strides=2))
    #convnet.add(Flatten())
    #convnet.add(Dense(100,activation='sigmoid'))
    convnet.add(Flatten())
    convnet.add(Reshape((1, 64)))
    convnet.add(Dense(100, activation='sigmoid'))

    # CNN 모델의 출력을 3D 텐서로 변환
    #conv_output_shape = convnet.output_shape[1:]  # (None, 4096)
    #conv_output = Reshape(conv_output_shape)(convnet.output)
    #conv_output_shape = convnet.output_shape[1:]  # (None, 100)
    conv_output_shape = (1, None, 64) # (batch_size, time_steps, features)

    #conv_output = Reshape((1, conv_output_shape[1]))(convnet.output)
    conv_output = Reshape((1, conv_output_shape[2]))(convnet.output)

    # LSTM 모델 구성
    #lstm = Sequential()
    #lstm.add(LSTM(128, input_shape=conv_output_shape, dropout=0.2, recurrent_dropout=0.2))
    #lstm.add(Dense(10, activation='softmax'))
    #수정 차원 변환
    # LSTM 모델 구성
    lstm = Sequential()
    #lstm.add(LSTM(128, input_shape=(None, conv_output_shape[1]), dropout=0.2, recurrent_dropout=0.2))
    lstm.add(LSTM(128, input_shape=conv_output_shape[1:], dropout=0.2, recurrent_dropout=0.2))

    lstm.add(Dense(10, activation='softmax'))

    # 두 개의 입력에 각각 CNN 모델을 적용
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # CNN 모델의 출력값을 LSTM 모델의 입력으로 사용
    lstm_input_shape = conv_output_shape[:2]  # (None, 64)
    encoded_l = Reshape(lstm_input_shape)(encoded_l)
    encoded_r = Reshape(lstm_input_shape)(encoded_r)
    encoded_l = lstm(encoded_l)
    encoded_r = lstm(encoded_r)


    # LSTM 모델의 출력값을 이용하여 이진 분류 수행
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

    return siamese_net



















def load_wdcnn_net_depth(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    convnet = Sequential()

    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    return siamese_net


def load_wdcnn_net(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()

    convnet.add(Conv1D(filters=16, kernel_size=64, strides=9, activation='relu', padding='same',input_shape=input_shape))
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

    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(10,activation='softmax')(Dropout(0.5)(encoded_cnn))
    wdcnn_net = Model(inputs=left_input,outputs=prediction_cnn)

    optimizer = Adam() 
    wdcnn_net.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    print(wdcnn_net.count_params())
    return wdcnn_net


#lstm 추가
def cnn_lstm_model(input_shape = (2048,2),nclasses=10):
    input_layer = Input(shape=input_shape)
    cnnlstmnet = Sequential()

    # CNN layers
    cnnlstmnet.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    cnnlstmnet.add(MaxPooling1D(pool_size=2))
    cnnlstmnet.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
    cnnlstmnet.add(MaxPooling1D(pool_size=2))
    cnnlstmnet.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=input_shape))
    cnnlstmnet.add(MaxPooling1D(pool_size=2))
    
    # LSTM layers
    cnnlstmnet.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=input_shape))
    cnnlstmnet.add(MaxPooling1D(pool_size=2))
    cnnlstmnet.add(RNN(LSTMCell(units=128), return_sequences=True, input_shape=input_shape))
    cnnlstmnet.add(RNN(LSTMCell(units=64)))
 
    
    # Dense layers
    cnnlstmnet.add(Flatten())
    cnnlstmnet.add(Dense(256,activation='relu'))
    
    
    output_layer = Dense(nclasses, activation='softmax')(cnnlstmnet)
    

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(cnnlstmnet.count_params())

    return model




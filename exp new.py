import tensorflow as tf
import numpy as np
import imp
import pandas as pd
import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import siamese
import utils
#추가: 그래픽카드 잡기 위한 코드
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import models
import cwru 

window_size = 2048
data = cwru.CWRU(['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
data.nclasses,data.classes,len(data.X_train),len(data.X_test)


# imp.reload(models)
siamese_net = models.load_siamese_net((window_size,2))
print('\nsiamese_net summary:')
siamese_net.summary()

print('\nsequential_3 is WDCNN:')
siamese_net.layers[2].summary()

wdcnn_net = models.load_wdcnn_net()
print('\nwdcnn_net summary:')
wdcnn_net.summary()

#from tensorflow.python import keras



imp.reload(siamese)
imp.reload(utils)

snrs = [-4,-2,0,2,4,6,8,10,None]


settings = {
  "N_way": 10,           # how many classes for testing one-shot tasks>
  "batch_size": 32,
  "best": -1,
  "evaluate_every": 200,   # interval for evaluating on one-shot tasks
  "loss_every": 20,      # interval for printing loss (iterations)
  "n_iter": 15000,
  "n_val": 2,          #how many one-shot tasks to validate on?
  "n": 0,
  "save_path":"",
  "save_weights_file": "weights-best-10-oneshot-low-data.hdf5"
}

exp_name = "EXP-AB-wlss"
exps = [60,90,120,200,300,600,900,1500,3000,6000,12000,19800]
# exps = [60]
# exps = [60,90,120]
times = 10

is_training = True   # enable or disable train models. if enable training, save best models will be update.


def train_and_test_siamese_network(settings, siamese_net, siamese_loader):
    best = settings['best']
    for i in range(settings['n_iter']):
        (inputs,targets) = siamese_loader.get_batch(settings['batch_size'])
        loss = siamese_net.train_on_batch(inputs, targets)
        if i % settings['evaluate_every'] == 0:
            print("=== Evaluating ===")
            val_acc, val_loss = siamese_loader.test_oneshot2(siamese_net, 
                                                             settings['N_way'], 
                                                             len(siamese_loader.classes['val']),
                                                             verbose=False)
            print("\n Siamese: Loss: {}, Val Acc: {}".format(loss, val_acc))
            if val_acc >= best:
                print("   Best so far! Saving...")
                siamese_net.save_weights(settings['save_path'] + settings['save_weights_file'])
                best = val_acc
                
        if i % settings['loss_every'] == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))


def EXPAB_train_and_test(exp_name, exps, is_training):
    train_classes = sorted(list(set(data.y_train)))
    train_indices = [np.where(data.y_train == i)[0] for i in train_classes]

    for exp in exps:
        scores_1_shot = []
        scores_5_shot = []
        scores_5_shot_prod = []
        scores_wdcnn = []
        num = int(exp/len(train_classes))
        settings['evaluate_every'] = 300 if exp<1000 else 600
        print(settings['evaluate_every'])

        for time_idx in range(times):
            seed = int(time_idx/4)*10
            np.random.seed(seed)
            print('random seed:', seed)
            print("\n%s-%s"%(exp,time_idx) + '*'*80)
            settings["save_path"] = "tmp/%s/size_%s/time_%s/" % (exp_name,exp,time_idx)
            data._mkdir(settings["save_path"])

            train_idxs = []
            val_idxs = []
            for i, c in enumerate(train_classes):
                select_idx = train_indices[i][np.random.choice(len(train_indices[i]), num, replace=False)]
                split = int(0.6*num)
                train_idxs.extend(select_idx[:split])
                val_idxs.extend(select_idx[split:])
            X_train, y_train = data.X_train[train_idxs], data.y_train[train_idxs]
            X_val, y_val = data.X_train[val_idxs], data.y_train[val_idxs]

            print(train_idxs[0:10])
            print(val_idxs[0:10])

            siamese_net = models.load_siamese_net()
            siamese_loader = siamese.Siamese_Loader(X_train, y_train, X_val, y_val)

            if is_training:
                train_and_test_siamese_network(settings, siamese_net, siamese_loader)

            y_train = keras.utils.to_categorical(y_train, data.nclasses)
            y_val = keras.utils.to_categorical(y_val, data.nclasses)
            y_test = keras.utils.to_categorical(data.y_test, data.nclasses)

            earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
            filepath="%sweights-best-10-cnn-low-data.hdf5" % (settings["save_path"])
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [earlyStopping,checkpoint]

            wdcnn_net = models.WDCNN(input_shape=data.input_shape, 
                                     nb_classes=data.nclasses, 
                                     dropout_rate=0.5)
            
            print('fitting data')
            history = wdcnn_net.fit(X_train, y_train,
                                    validation_data=(X_val, y_val),
                                    callbacks=callbacks_list,
                                    batch_size=settings["batch_size"],
                                    epochs=settings["epochs"],
                                    verbose=1)
            
            # load the best model
            wdcnn_net.load_weights("%sweights-best-10-cnn-low-data.hdf5" % (settings["save_path"]))

            scores_wdcnn.append(wdcnn_net.evaluate(data.X_test, y_test, verbose=1))
            print("%s: %.2f%%" % (wdcnn_net.metrics_names[1], scores_wdcnn[-1][1]*100))



        a =pd.DataFrame(np.array(scores_1_shot).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_1_shot.csv" % (exp_name,exp),index=True)

        a =pd.DataFrame(np.array(scores_5_shot).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot.csv" % (exp_name,exp),index=True)
        
        a =pd.DataFrame(np.array(scores_5_shot_prod).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot_prod.csv" % (exp_name,exp),index=True)

        a =pd.DataFrame(np.array(scores_wdcnn).reshape(-1,len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_wdcnn.csv" % (exp_name,exp),index=True)   

EXPAB_train_and_test(exp_name,exps,is_training)

np.bincount([2,2,3,3,1])
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
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
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
  "n_iter": 1000,
  "n_val": 2,          #how many one-shot tasks to validate on?
  "n": 0,
  "save_path":"",
  "save_weights_file": "weights-best-10-oneshot-low-data.hdf5"
}

exp_name = "EXP-AB-wl0319"
#exps = [900,1500,3000,6000,12000,19800]
exps = [60]
# exps = [60,90,120]
times = 1

is_training = True   # enable or disable train models. if enable training, save best models will be update.

def EXPAB_train_and_test(exp_name,exps,is_training):
    train_classes = sorted(list(set(data.y_train)))
    train_indices = [np.where(data.y_train == i)[0] for i in train_classes]
    for exp in exps:
        scores_1_shot = []
        scores_5_shot = []
        scores_5_shot_prod = []
        scores_wdcnn = []
        #재욱수정_0319
        scores_lstmcnn = []
        
        num = int(exp/len(train_classes))
        settings['evaluate_every'] = 300 if exp<1000 else 600
        print(settings['evaluate_every'])
        for time_idx in range(times):
            seed = int(time_idx/4)*10
            np.random.seed(seed)
            print('random seed:',seed)
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
            X_train, y_train = data.X_train[train_idxs],data.y_train[train_idxs], 
            X_val, y_val = data.X_train[val_idxs],data.y_train[val_idxs], 
            
            print(train_idxs[0:10])
            print(val_idxs[0:10])

            # load one-shot model and training
            # 2023.02.16 수정 load_siamese_net_depth
            #siamese_net = models.load_siamese_net()
            siamese_net = models.load_siamese_net()
            siamese_loader = siamese.Siamese_Loader(X_train,
                                            y_train,
                                            X_val,
                                            y_val)

            if(is_training):
                print(siamese.train_and_test_oneshot(settings,siamese_net,siamese_loader))
            #재욱수정_0319
            #if(is_training):
            #    print(siamese.train_and_test_oneshot(settings,siamese_net,siamese_loader))
                
            # load wdcnn model and training
            
            y_train = keras.utils.to_categorical(y_train, data.nclasses)
            y_val = keras.utils.to_categorical(y_val, data.nclasses)
            y_test = keras.utils.to_categorical(data.y_test, data.nclasses)

            earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
            # checkpoint
            # filepath="tmp/weights-best-cnn-{epoch:02d}-{val_acc:.2f}.hdf5"
            filepath="%sweights-best-10-cnn-low-data.hdf5" % (settings["save_path"])
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [earlyStopping,checkpoint]

            wdcnn_net = models.load_wdcnn_net()
            if(is_training):
                    wdcnn_net.fit(X_train, y_train,
                              batch_size=32,
                              epochs=300,
                              verbose=0,
                              callbacks=callbacks_list,
                              validation_data=(X_val, y_val))
                    
            # loading best weights and testing
            print("load best weights",settings["save_path"] + settings['save_weights_file'])
            siamese_net.load_weights(settings["save_path"] + settings['save_weights_file'])
            print("load best weights",filepath)
            # wdcnn_net.load_weights(filepath)
            for snr in snrs:
                print("\n%s_%s_%s"%(exp,time_idx,snr) + '*'*80)
                X_test_noise = []
                if snr != None:
                    for x in data.X_test:
                        #cnt_x 는 어디서 초기화되나요? 혹은 특정 값이 있나요?
                        X_test_noise.append(utils.noise_rw(x,snr))
                    X_test_noise = np.array(X_test_noise)
                else:
                    X_test_noise = data.X_test
                
                
                # test 1_shot and 5_shot
                siamese_loader.set_val(X_test_noise,data.y_test)
                s = 'val'
                preds_5_shot = []
                prods_5_shot = []
                scores = []
                for k in range(5):
                    val_acc,preds, prods = siamese_loader.test_oneshot2(siamese_net,len(siamese_loader.classes[s]),
                                                                 len(siamese_loader.data[s]),verbose=False)
    #                 utils.confusion_plot(preds[:,1],preds[:,0])
                    print(val_acc,preds.shape, prods.shape)
                    scores.append(val_acc)
                    preds_5_shot.append(preds[:,1])
                    prods_5_shot.append(prods)
                preds = []
                for line in np.array(preds_5_shot).T:
                    pass
                    preds.append(np.argmax(np.bincount(line)))
    #             utils.confusion_plot(np.array(preds),data.y_test) 
                prod_preds = np.argmax(np.sum(prods_5_shot,axis=0),axis=1).reshape(-1)

                score_5_shot = accuracy_score(data.y_test,np.array(preds))*100
                print('5_shot:',score_5_shot)
                
                score_5_shot_prod = accuracy_score(data.y_test,prod_preds)*100
                print('5_shot_prod:',score_5_shot_prod)
                
                scores_1_shot.append(scores[0])
                scores_5_shot.append(score_5_shot)
                scores_5_shot_prod.append(score_5_shot_prod)

                # test wdcnn
                score = wdcnn_net.evaluate(X_test_noise, y_test, verbose=0)[1]*100
                print('wdcnn:', score)
                scores_wdcnn.append(score)


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
        
#재욱수정_0318        
#        a =pd.DataFrame(np.array(scores_lstmcnn).reshape(-1,len(snrs)))
#        a.columns = snrs
#        a.to_csv("tmp/%s/size_%s/scores_lstmcnn.csv" % (exp_name,exp),index=True)   
        
EXPAB_train_and_test(exp_name,exps,is_training)

np.bincount([2,2,3,3,1])

# -*- coding:utf-8 -*-
import c3d_model
import json
from keras.models import Sequential, load_model,model_from_json
from keras.optimizers import SGD,Adam
from keras.utils import np_utils, plot_model
from schedules import onetenth_4_8_12,onetenth_15_30_45
from keras.models import Sequential, load_model,model_from_json
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
import numpy as np
import random
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.pooling import GlobalMaxPooling3D,GlobalAveragePooling3D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.activations import relu
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from learning_rate_schedulers import StepDecay
from learning_rate_schedulers import PolynomialDecay
from keras.callbacks import LearningRateScheduler
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import sys
import keras.backend as K
from confusion_matrix_plot import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from numpy.random import seed

dim_ordering = K.image_data_format()
print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

model_dir = './models'
global backend
if len(sys.argv) > 1:
    if 'tf' in sys.argv[1].lower():
        backend = 'tf'
    else:
         backend = 'th'
print ("[Info] Using backend={}".format(backend))

#model.compile(loss='mean_squared_error', optimizer='sgd')

def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy_c3d_YogaV2_Split3_sportsV3.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss_c3d_YogaV2_Split3_sportsV3.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_c3d_YogaV2_Split3_sportsV3.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

        
def process_batch(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,128,171,3),dtype='float32')
    #mini_batch = np.zeros((16,1))
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        if train:
            #crop_x = random.randint(0, 15)
            #crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            count = 0
            for j in range(64):
                if (j%4==1):
                    #print(j)
                    #print(count)
                    img = imgs[symbol + j]
                    #print(count)
                    #print(img)
                    image = cv2.imread(img_path + path + '/' + img)
                    #image_h, image_w,image_c= np.shape(image)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    #crop_random = random.random()
                    
                    #square_sz = min(image_h, image_w)
                    #crop_h = int((image_h - square_sz)*crop_random)
                    #crop_w = int((image_w - square_sz)*crop_random)
                    #image = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
                    image = cv2.resize(image, (171, 128))
                    #cv2.imshow('image',image)
                    #cv2.waitKey(1)
                    if is_flip == 1:
                        image = cv2.flip(image, 1)
                    batch[i][count][:][:][:] = image#[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                    count = count +1
            labels[i] = label
        else:
            count = 0
            for j in range(64):
                if (j%4==1):
                    #count = count+1
                    #print(count)
                    img = imgs[symbol + j]
                    image = cv2.imread(img_path + path + '/' + img)
                    #image_h,image_w,image_c= np.shape(image)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    #crop_random = random.random()
                    
                    #square_sz = min(image_h, image_w)
                    #crop_h = int((image_h - square_sz)/2)
                    #crop_w = int((image_w - square_sz)/2)
                    #image = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
                    
                    image = cv2.resize(image, (171, 128))
                    #cv2.imshow('image',image)
                    #cv2.waitKey(1)
                    batch[i][count][:][:][:] = image#[8:120, 30:142, :]
                    #batch[i][j][:][:][:] = image
                    count = count+1
            labels[i] = label
    return batch, labels
    '''
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1:
                    image = cv2.flip(image, 1)
                batch[i][j][:][:][:] = image#[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image#[8:120, 30:142, :]
                #batch[i][j][:][:][:] = image
            labels[i] = label
    return batch, labels
'''

def preprocess(inputs):
    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
    #print(mean_cube.shape)
                #diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
    inputs -= mean_cube
                #diagnose(X, verbose=True, label='Mean-subtracted X', plots=show_images)
                # center crop
    #inputs = inputs[:, 8:120, 30:142, :] # (l, h, w, c)
    #inputs[..., 0] -= 99.9
    #inputs[..., 1] -= 92.1
    #inputs[..., 2] -= 82.6
    #inputs[..., 0] /= 65.8
    #inputs[..., 1] /= 62.3
    #inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs


def generator_train_batch(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b],img_path,train=True)
            #print(x_train.shape)
            x = preprocess(x_train)
            #print(x.shape)
            #x=x_train[:,:,:, :]
            x=x[:,:,crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            #print(x.shape)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            #x = np.transpose(x, (0,2,3,1,4))
            #print(x.shape)
            #x = np.transpse(x,(3,0,1,2))
            #3, 0, 1, 2
            yield x, y


def generator_val_batch(val_txt,batch_size,num_classes,img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        #random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = x[:,:,8:120, 30:142, :]
            #x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    img_path = '/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v2/Images/'
    train_file = '/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v2/traintestlist-v2/train_list03.txt'
    test_file = '/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v2/traintestlist-v2/test_list03.txt'
    if backend == 'th':
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
    else:
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    model = model_from_json(open(model_json_filename, 'r').read())
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")
    plot_model(model, to_file="model_c3d_yoga_sportsV3.png", show_shapes=True,show_layer_names=True)
    model.summary()

    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    #model.pop()

    for layer in model.layers[:]:
        layer.trainable = False
        # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)
    #model.pop()
    #model.pop()
    #model.add(Dense(2048,kernel_initializer="he_normal",activation='relu',kernel_regularizer=l2(0.0001)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(GlobalAveragePooling3D())
    model.add(Dense(256,kernel_initializer="he_normal",activation='relu',kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5,name='drop'))
    
    model.add(Dense(10, activation='softmax', name='fc8'))
    #model.summary()
    

# lets assume `model` is main model 


    inspect_layers = [
                    #    'fc6',
                    #    'fc7',
        ]
    for layer in inspect_layers:
        model = c3d_model.get_int_model(model=model, layer=layer, backend=backend)

        #int_output = int_model.predict_on_batch(np.array([X]))
        #int_output = int_output[0, ...]
        print ("[Debug] at layer={}: output.shape={}".format(layer, int_output.shape))

    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    print(train_samples)
    lines = f2.readlines()
    #print(lines)
    num = len(lines)
    val_labels = np.zeros(num,dtype='int')
    for i in range(num):
        val_label = lines[i].split(' ')[-1]
        #symbol = lines[i].split(' ')[1]
        label = val_label.strip('\n')
        label = int(label)
        val_labels[i] = label


    #label = lines.splitlines(' ')[-1]
    print(val_labels.shape)
    f2.close()
    val_samples = len(lines)
    print(val_samples)

    num_classes = 10
    train_batch_size = 8
    val_batch_size= 4
    val_batch_size_test = 1
    epochs = 30

    save_best=ModelCheckpoint('models/weights_c3d_YogaV2_Split3_SportsV3.h5', monitor='val_accuracy', save_best_only=True)
    #lr = 0.0001
    lr=0.0001
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    #adam = Adam(lr=lr)
    #schedule = StepDecay(initAlpha=1e-3, factor=0.1, dropEvery=10)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(generator_train_batch(train_file, train_batch_size, num_classes,img_path),
                                  steps_per_epoch=train_samples // train_batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_4_8_12(lr),save_best],
                                  validation_data=generator_val_batch(test_file,
                                        val_batch_size,num_classes,img_path),
                                  validation_steps=val_samples // val_batch_size,
                                  verbose=1)

    print('re-compiling model')
    #generator_val_batch.reset()
    #generator_train_batch.reset()


    for layer in model.layers[:]:
            layer.trainable = True
        # Check the trainable status of the individual layers
    for layer in model.layers:
            print(layer, layer.trainable)

    epochs = 30

    save_best=ModelCheckpoint('models/weights_c3d_YogaV2_Split3_SportsV3.h5', monitor='val_accuracy', save_best_only=True)
    lr = 0.0001
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    #adam = Adam(lr=lr)
    #schedule = StepDecay(initAlpha=1e-3, factor=0.1, dropEvery=10)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(generator_train_batch(train_file, train_batch_size, num_classes,img_path),
                                 steps_per_epoch=train_samples // train_batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_4_8_12(lr),save_best],
                                  validation_data=generator_val_batch(test_file,
                                        val_batch_size,num_classes,img_path),
                                  validation_steps=val_samples // val_batch_size,
                                  verbose=1)
    history_dict = history.history
    #print(history_dict.keys())
    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history,'results/')
    save_history(history,'results/')
    model_json = model.to_json()
    with open("c3d_YogaV2_Split3_SportsV3_json.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights('results/weights_c3d_YogaV2_Split3_SportsV3.h5')
    steps = val_samples

    loaded_model = load_model('models/weights_c3d_YogaV2_Split3_SportsV3.h5')
    score = loaded_model.evaluate_generator(generator_val_batch(test_file,val_batch_size_test,num_classes,img_path),steps)
    print(score)
    
    Y_pred = loaded_model.predict_generator(generator_val_batch(test_file,val_batch_size_test,num_classes,img_path),steps)
    Y_pred = np.argmax(Y_pred, axis=1)
    print(Y_pred)
    print('Confusion_matrix')
    #generator_val_batch(test_file,val_batch_size,num_classes,img_path)

    #print(data_y.shape)

    cm = confusion_matrix(val_labels,Y_pred)
    print(cm)
    #ts_ac = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5])/val_samples
    ts_ac = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9])/val_samples
    print("Testing: " + str(ts_ac))
    #target_names = ['Bhujangasana','Padmasana','Shavasana','Tadasana','Trikonasana','Vrikshasana']
    target_names = ['Garland_Pose','Happy_Baby_Pose','Head_To_Knee_Pose','Lunge_Pose','Mountain_Pose','Plank_Pose','Raised_Arms_Pose','Seated_Forward_Bend','Staff_Pose','Standing_Forward_Bend']
    print(classification_report(val_labels, Y_pred, target_names=target_names))
    plot_confusion_matrix(cm,normalize=True,target_names=target_names)


if __name__ == '__main__':
    main()

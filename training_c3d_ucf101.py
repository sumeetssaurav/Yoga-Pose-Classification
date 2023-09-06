# -*- coding:utf-8 -*-
from c3d import c3d
import json
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from schedules import onetenth_4_8_12
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
import numpy as np
import random
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import os
import random
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import sys
import keras
import keras.backend as K
l2=keras.regularizers.l2
from keras.utils import plot_model



model_dir = './models'

#model.compile(loss='mean_squared_error', optimizer='sgd')

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
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
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            count = 0
            for j in range(32):
                if (j%2):
                    #print(j)
                    #print(count)
                    img = imgs[symbol + j]
                    #print(count)
                    #print(img)
                    image = cv2.imread(img_path + path + '/' + img)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (171, 128))
                    if is_flip == 1:
                        image = cv2.flip(image, 1)
                    batch[i][count][:][:][:] = image#[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                    count = count +1
            labels[i] = label
        else:
            count = 0
            for j in range(32):
                if (j%2):
                    #count = count+1
                    #print(count)
                    img = imgs[symbol + j]
                    image = cv2.imread(img_path + path + '/' + img)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (171, 128))
                    batch[i][count][:][:][:] = image#[8:120, 30:142, :]
                    #batch[i][j][:][:][:] = image
                    count = count+1
            labels[i] = label
    return batch, labels


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
        random.shuffle(index)
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
    img_path = '/home/sumeet/Documents/C3D-Keras/ucf101/Images/'
    train_file ='/home/sumeet/Documents/C3D-Keras/train_list.txt'
    test_file = '/home/sumeet/Documents/C3D-Keras/test_list.txt'
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)

    num_classes = 101
    train_batch_size = 16
    val_batch_size= 16
    epochs = 100
    weight_decay= 0.001
    lr = 0.001

    inputs = keras.layers.Input(shape=(16,112,112,3))
    feature = c3d(inputs, weight_decay)
    classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(feature)
    outputs = keras.layers.Activation('softmax', name='Output')(classes)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    plot_model(model, to_file="model.png", show_shapes=True,show_layer_names=True)
    save_best=ModelCheckpoint('models/weights_c3d.h5', monitor='val_acc', save_best_only=True)
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
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
    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history,'results/')
    save_history(history,'results/')
    model_json = model.to_json()
    with open("ucf_C3D_in_json.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights('results/weights_c3d.h5')


if __name__ == '__main__':
    main()

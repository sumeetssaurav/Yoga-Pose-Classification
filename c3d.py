import io
import sys
import numpy as np
import tensorflow as tf
import keras
l2=keras.regularizers.l2

def c3d(inputs,weight_decay):

  #network_input = keras.layers.InputLayer(inputs, name='input_layer') 
  # 3DCNN-BN Layer 1
  conv3d_1 = keras.layers.Conv3D(64,(3,3,3),strides=(1,1,1),padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_1')(inputs)
  conv3d_1 = keras.layers.BatchNormalization(name='BatchNorm_1')(conv3d_1)
  conv3d_1 = keras.layers.Activation('relu', name='ReLU_1')(conv3d_1)
  pool3d_1 = keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same',name='Pool_1')(conv3d_1)
 
  # 3DCNN-BN Layer 2
  conv3d_2 = keras.layers.Conv3D(128,(3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2')(pool3d_1)
  conv3d_2 = keras.layers.BatchNormalization(name='BatchNorm_2')(conv3d_2)
  conv3d_2 = keras.layers.Activation('relu', name='ReLU_2')(conv3d_2)
  pool3d_2 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same',name='Pool_2')(conv3d_2)
 # 3DCNN-BN Layer 3
  conv3d_3 = keras.layers.Conv3D(256,(3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a')(pool3d_2)
  conv3d_3 = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3b')(conv3d_3)
  conv3d_3 = keras.layers.BatchNormalization(name='BatchNorm_3')(conv3d_3)
  conv3d_3 = keras.layers.Activation('relu', name='ReLU_3')(conv3d_3)
  conv3d_3 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same',name='Pool_3')(conv3d_3)
  #3DCNN-BN Layer 4
  conv3d_4 = keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a')(conv3d_3)
  conv3d_4= keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4b')(conv3d_4)
  conv3d_4 = keras.layers.BatchNormalization(name='BatchNorm_4')(conv3d_4)
  conv3d_4 = keras.layers.Activation('relu', name='ReLU_4')(conv3d_4)
  conv3d_4 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same',name='Pool_4')(conv3d_4)
  #3DCNN-BN Layer 5
  conv3d_5 = keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5a')(conv3d_4)
  conv3d_5= keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5b')(conv3d_5)
  conv3d_5 = keras.layers.BatchNormalization(name='BatchNorm_5')(conv3d_5)
  conv3d_5 = keras.layers.Activation('relu', name='ReLU_5')(conv3d_5)
  conv3d_5 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same',name='Pool_5')(conv3d_5)
  flat_features = keras.layers.Flatten()(conv3d_5)
  # FC layers group
  fc1 = keras.layers.Dense(4096,name='fc6') (flat_features)
  fc1 = keras.layers.BatchNormalization(name='BatchNorm_6')(fc1)
  fc1= keras.layers.Activation('relu', name='ReLU_6')(fc1)
  fc1 = keras.layers.Dropout(.5) (fc1)

  fc1 = keras.layers.Dense(4096,name='fc7') (fc1)
  fc1 = keras.layers.BatchNormalization(name='BatchNorm_7')(fc1)
  fc1= keras.layers.Activation('relu', name='ReLU_7')(fc1)
  features = keras.layers.Dropout(.5) (fc1)
  
  return features

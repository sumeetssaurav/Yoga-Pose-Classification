#!/usr/bin/env python
from keras.models import load_model
import cv2
import numpy as np
import keras.backend as K
import time
dim_ordering = K.image_data_format()
print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

mean_cube = np.load('models/train01_16_128_171_mean.npy')
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
model = load_model('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/models/test_models/weights_c3d_YogaV1_Split3_SportsV3.h5')

print("[Info] Loading labels...")
'''
with open('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/traintestlist-v1/classInd.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))
'''
labels = ['Bhujangasana','Padmasana','Shavasana','Tadasana','Trikonasana','Vrikshasana']

print("[Info] Loading a sample video...")
cap = cv2.VideoCapture('/media/ccg1/UBUNTU/home/ccg10/Documents/C3D-Yoga/yoga-ceeri-v1/Videos/Vrikshasana/lakshmi_vriksh.mp4')

vid = []
frame_count = 0
frame_time = 0
while True:
    ret, img = cap.read()
    if ret:
        vid.append(cv2.resize(img, (171, 128)))
        frame_count += 1
        t = time.time()
        #print(len(vid))
        if len(vid)==16:
            X = np.array(vid, dtype=np.float32)
            X -= mean_cube
            X = X[:, 8:120, 30:142, :] # (l, h, w, c)
            output = model.predict(np.array([X]))
            label = np.argmax(output[0])
            frame_time += time.time() - t
            fps = frame_count / float(frame_time)
            #print(fps)
            fps_label = "FPS : {:.2f}".format(fps)
            cv2.putText(img, fps_label, (400, 260),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 0, 0), 2)
            cv2.putText(img, labels[label].split(' ')[-1].strip(), (400, 200),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 0), 2)
            cv2.putText(img, "prob: %.4f" % output[0][label], (400, 230),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 0, 255), 2)
            vid.pop(0)
        cv2.imshow('result',img)
        cv2.waitKey(1)
    else:
        break
cap.release()
cv2.destroyAllWindows()

import numpy as np
import pandas as pd
import os

train_labels = pd.read_csv('/hpctmp/e0427773/histopathologic-cancer-detection/train_labels.csv', dtype=str)
train_labels['label'] = train_labels['label'].astype(float)
train_labels['label'].value_counts()

train_labels_pos = train_labels[train_labels['label']==1]
train_labels_neg = train_labels[train_labels['label']==0]
train_labels_neg = train_labels_neg.sample(n = train_labels_pos.shape[0])
train_labels_equal = pd.concat([train_labels_neg,train_labels_pos]).sample(frac=1, random_state=17).reset_index(drop=True)
train_labels_equal.shape
train_labels_equal['label'].value_counts().plot(kind='pie')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from sklearn.model_selection import train_test_split
train_img, valid_img = train_test_split(train_labels_equal, test_size=0.25, random_state=1710, stratify=train_labels_equal.label)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.layers import PReLU
from keras.initializers import Constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_img['id'] = train_img['id']+'.tif'
valid_img['id'] = valid_img['id']+'.tif'

train_img['label'] = train_img['label'].astype(str)
valid_img['label'] = valid_img['label'].astype(str)

train_datagen=ImageDataGenerator(rescale=1/255)

train_generator=train_datagen.flow_from_dataframe(dataframe=train_img,directory="/hpctmp/e0427773/histopathologic-cancer-detection/train/",
                x_col="id",y_col="label",batch_size=64,seed=1710,shuffle=True,
                class_mode="binary",target_size=(96,96))

valid_generator=train_datagen.flow_from_dataframe(dataframe=valid_img,directory="/hpctmp/e0427773/histopathologic-cancer-detection/train/",
                x_col="id",y_col="label",batch_size=64,seed=1710,shuffle=True,
                class_mode="binary",target_size=(96,96))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(96,96,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
          
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu')) 

model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30, verbose=1
)

hist_df = pd.DataFrame(history.history) 
with open('history1.csv', mode='w') as f:
    hist_df.to_csv(f)

test_set = os.listdir('/hpctmp/e0427773/histopathologic-cancer-detection/test/')
test_df = pd.DataFrame(test_set)
test_df.columns = ['id']
test_datagen=ImageDataGenerator(rescale=1/255)

test_generator=test_datagen.flow_from_dataframe(dataframe=test_df, directory="/hpctmp/e0427773/histopathologic-cancer-detection/test/",x_col="id",batch_size=64,seed=1710,shuffle=False,class_mode=None,target_size=(96,96))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1

preds = model.predict_generator(generator=test_generator,steps=STEP_SIZE_TEST, verbose = 1)

predictions = []
for pred in preds:
    if pred >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

submission = test_df.copy()
submission['id'] = list(map(lambda x: x[:-4], submission['id']))
submission['label']=predictions
submission.to_csv('submission.csv',index=False)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
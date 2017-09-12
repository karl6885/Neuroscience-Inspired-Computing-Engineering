"""
code reference:
 1) https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
 2) https://www.bonaccorso.eu/2016/08/06/cifar-10-image-classification-with-keras-convnet/
"""

from __future__ import print_function # for python 2.7

import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

# for reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # load the dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # cifar10 -> image size: (32, 32, 3). so I modified the first conv layer.
    
    # create the model
    model = Sequential()
    
    # block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Flatten
    model.add(Flatten())
    
    # fc6
    model.add(Dense(4096, activation='relu'))
    
    # fc7
    model.add(Dense(4096, activation='relu'))
    
    # fc8
    model.add(Dense(10, activation='softmax'))
    # cifar10 has only 10 classes!
    
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
                  
    # train the model
    model.fit(X_train / 225.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=10,
              validation_data=(X_test / 225.0, to_categorical(Y_test)))
              
    # evaluate the model
    scores = model.evaluate(X_test / 225.0, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

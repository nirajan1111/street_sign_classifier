import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense,Flatten, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.python.keras import activations
from tensorflow.keras import Model
from mnist import functional_model, MyCustomModel
from myutils import display_examples
# model =tf.keras.Sequential(
#     [
#         Input(shape=(28,28,1)),
#         Conv2D(32, (3,3), activation='relu'),
#         Conv2D(64, (3,3), activation='relu'),

#         MaxPool2D(),
#         BatchNormalization(),
#         Conv2D(128, (3,3), activation='relu'),
#         MaxPool2D(),
#         BatchNormalization(),

#         GlobalAvgPool2D(),
#         Dense(64, activation='relu'),
#         Dense(10, activation='softmax')

#     ]
# )


def Streetsign_detector(nbr_classes):
    my_input = Input(shape=(60,60,3))
    x=Conv2D(32, (3,3), activation='relu')(my_input)
    x=x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Conv2D(32, (3,3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Flatten()(x)
    # x=GlobalAvgPool2D()(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(nbr_classes, activation='softmax')(x)

    model=tf.keras.Model(inputs=my_input, outputs=x)
    return model








if __name__=='__main__':
    model=Streetsign_detector(10)
    model.summary()
    
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    # model.fit(x_train, y_train, batch_size= 64, epochs=3, validation_split=0.2)
    # model.evaluate(x_test, y_test, batch_size=64)
    




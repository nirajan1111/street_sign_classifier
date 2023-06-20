import os
import glob
import shutil
from myutils import create_generators
import tensorflow as tf
from myutils import order_test_set
from splitter import split_data
from dl import Streetsign_detector 

from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping
if __name__=="__main__":
    if False:
        path_to_data="/Users/nirajansah/Downloads/archive/Train"
        path_to_save_train="/Users/nirajansah/Downloads/archive/train_data/train"
        path_to_save_val= "/Users/nirajansah/Downloads/archive/train_data/val"
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    
    if False:
        path_to_images= "/Users/nirajansah/Downloads/archive/Test"
        path_to_csv="/Users/nirajansah/Downloads/archive/Test.csv"
        order_test_set(path_to_images=path_to_images, path_to_csv=path_to_csv)

    TEST=True
    TRAIN=False
    batch_size=64
    path_to_train="/Users/nirajansah/Downloads/archive/train_data/train"
    path_to_val= "/Users/nirajansah/Downloads/archive/train_data/val"
    path_to_test ="/Users/nirajansah/Downloads/archive/Test"
    epochs=15
    train_generator, val_generator, test_generator=  create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes= train_generator.num_classes
    if TRAIN:
        path_to_train="/Users/nirajansah/Downloads/archive/train_data/train"
        path_to_val= "/Users/nirajansah/Downloads/archive/train_data/val"
        path_to_test ="/Users/nirajansah/Downloads/archive/Test"
        
        path_to_save_model= "./Models"
        checkpoint_saver= ModelCheckpoint(
        path_to_save_model,
        monitor= "val_accuracy",
        mode="max", 
        save_best_only=True,
        save_freq= "epoch",
        verbose = 1
    )
        early_stop= EarlyStopping(
        monitor="val_accuracy",
        patience=10
    )
        model= Streetsign_detector(nbr_classes)

        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics= ['accuracy'])
        model.fit(train_generator,epochs=epochs, batch_size=batch_size, validation_data=val_generator, callbacks=[checkpoint_saver, early_stop])

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()
        print("Evaluating valuation datasets... ")
        model.evaluate(val_generator)
        print("Evaluating test data...")
        model.evaluate(test_generator)



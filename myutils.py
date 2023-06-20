import matplotlib.pyplot as plt
import numpy as np
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
def display_examples(examples, labels):
    plt.figure(figsize=(10,10))
    for i in range(20):
        idx = np.random.randint(0,examples.shape[0]-1)
        img= examples[idx]
        label=labels[idx]
        plt.subplot(4,5,i+1)
        plt.tight_layout()
        plt.title(str(label))
        plt.imshow(img)
    plt.show()

def order_test_set(path_to_images, path_to_csv):


    try:
        with open(path_to_csv, 'r') as csvfile:
            reader =csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i==0:
                    continue
                img_name=row[-1].replace('Test/', '')
                label=row[-2]

                path_to_folder=os.path.join(path_to_images,label)
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path =os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print("we got error in reading csv file")


    
def create_generators(batch_size, train_data_path, val_data_path,test_data_path):
    preprocessor= ImageDataGenerator(
        rescale = 1/255.
    )
    train_generator = preprocessor.flow_from_directory(
        train_data_path, 
        class_mode="categorical",
        target_size= (60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    val_generator = preprocessor.flow_from_directory(
        val_data_path, 
        class_mode="categorical",
        target_size= (60,60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    test_generator = preprocessor.flow_from_directory(
        test_data_path, 
        class_mode="categorical",
        target_size= (60,60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    return train_generator, val_generator, test_generator

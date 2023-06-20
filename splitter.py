import os
from sklearn.model_selection import train_test_split
import glob
import shutil
def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)
    for folder in folders:
        full_path= os.path.join(path_to_data, folder)
        images_path= glob.glob(os.path.join(full_path,'*.png'))
        if len(images_path) == 0:
            print(f"No images found in {full_path}. Skipping...")
            continue
        x_train, x_val= train_test_split(images_path, test_size=0.1)

        for x in x_train:
            path_to_folder=os.path.join(path_to_save_train, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x,path_to_folder)
        for y in x_val:
            path_to_folder=os.path.join(path_to_save_val, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(y,path_to_folder)

if __name__=="__main__":
    path_to_data="/Users/nirajansah/Downloads/archive/Train"
    path_to_save_train="/Users/nirajansah/Downloads/archive/train_data/train"
    path_to_save_val= "/Users/nirajansah/Downloads/archive/train_data/val"
    split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)





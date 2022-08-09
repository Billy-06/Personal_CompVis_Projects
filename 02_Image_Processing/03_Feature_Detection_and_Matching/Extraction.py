#feature reduction
import numpy as np
import skimage.io, skimage.color, skimage.feature
import os
import matplotlib.pyplot as plt
import pickle


apple = skimage.io.imread(fname="apple.jpg", as_gray=False)
raspberry = skimage.io.imread(fname="raspberry.jpg", as_gray=False)
lemon = skimage.io.imread(fname='lemon.jpg', as_gray=False)
mango = skimage.io.imread(fname='mango.jpg',as_gray=False)

fruits = ['apple', 'raspberry', 'mango', 'lemon']
dataset_features = np.zeros(shape=(1962, 360))
output = np.zeros(shape=1962)

idx = 0
class_label = 0
for fruit_dir in fruits:
    curr_dir = os.path.join(os.path.sep,'train', fruit_dir)
    all_imgs = os.listdir(os.getcwd()+curr_dir)
    for img_file in all_imgs:
        fruit_data = skimage.io.imread(fname=os.getcwd()+curr_dir+img_file, as_gray=False)
        fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
        hist = np.histogram(a=fruit_data_hsv[:,:,0], bins=360)
        dataset_features[idx, :] = hist[0]
        output[idx] = class_label
        idx += 1
        class_label += 1
        
with open("dataset_features.pkl", 'wb') as f:
    pickle.dump('dataset_features.pkl', f)
    
with open('outputs.pkl', 'wb') as f:
    pickle.dump('outputs.pkl', f)

features_STDs = np.std(a=dataset_features, axis=0)

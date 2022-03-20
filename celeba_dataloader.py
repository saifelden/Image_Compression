import cv2
import numpy as np
from glob import glob 
import random
from einops import rearrange


class CelebA256x256:

    def __init__(self,download_path='/home/saifeldein/git_repos/Image_Compression/archive/celeba_hq_256',split_index = -5000,shuffle=False,batch_size=32):
        self.download_path=download_path
        img_paths = glob(self.download_path+'/*.jpg')
        self.train_imgs = img_paths[:split_index]
        self.test_imgs = img_paths[split_index:]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.train_index=0
        self.test_index = 0
        self.num_of_train_batches = int(len(self.train_imgs)/self.batch_size)
        self.num_of_test_batches = int(len(self.test_imgs)/self.batch_size)
    
    def get_num_of_batches(self,train=True):
        if train:
            return self.num_of_train_batches
        else:
            return self.num_of_test_batches


    def next_batch(self,train=True):
        if train:
            curr_batch= self.train_imgs[self.train_index:self.train_index+self.batch_size]
            self.train_index+=self.batch_size
        else:
            curr_batch= self.test_imgs[self.test_index:self.test_index+self.batch_size]
            self.test_index+=self.batch_size

        batch_imgs = []
        for img_path in curr_batch:
            batch_imgs.append(cv2.imread(img_path))
        return rearrange(np.array(batch_imgs),'b h w c -> b c h w').astype(np.float32)/255
           
    def on_epoch_ends(self):
        self.train_index=0
        self.test_index=0
        if self.shuffle:
            random.shuffle(self.train_imgs)

        


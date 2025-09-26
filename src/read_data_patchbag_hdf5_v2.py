import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
#import lmdb
#import lz4framed
import cv2
import h5py
import pdb

from typing import Any
from PIL import Image
from sklearn.preprocessing import LabelEncoder

## adapted from https://github.com/gevaertlab/RNA-CDM/blob/9270760dd0a39d6134d8a469759f6b0298ee4599/src/read_data.py#L416
##class PatchRNADatasetHDF5

class PatchBagDataset(Dataset):
        def __init__(self, patch_data_path, csv_path,img_size, 
                max_patches_total=300,transform=None, bag_size=40, quick=False, label_encoder=None,
                 normalize=True, return_ids = False
               ):
            self.patch_data_path = patch_data_path
            self.csv_path = csv_path
            self.transform = transform
            self.bag_size = bag_size
            self.max_patches_total = max_patches_total
            self.quick = quick
            self.images = []
            self.filenames = []
            self.labels = []
            self.paths =[]
            self.label_vals =[]
            self.le = label_encoder
            self.img_size = img_size
            self.index = []
            self.data = {}
            self.return_ids = return_ids
            self._preprocess()
        def _preprocess(self):
            if type(self.csv_path) == str:
                csv_file = pd.read_csv(self.csv_path)
                #le = LabelEncoder()
                #le.fit(csv_file['Labels'])
                #csv_file['label_encoded'] = le.transform(csv_file['Labels'])
                csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
                #csv_file['labels'] = [0] * csv_file.shape[0]
                csv_file['Labels2'] = csv_file['Labels']
                csv_file['Labels2'] = csv_file['Labels2'].astype('category')
                csv_file['label_encoded'] = csv_file['Labels2'].cat.codes
            else:
                csv_file = self.csv_path
            if self.quick:
                csv_file = csv_file.sample(15)
                print('len of csv file is ..', csv_file.shape)

            for i, row in tqdm(csv_file.iterrows()):
                project = row['tcga_project']
                WSI = row['wsi_file_name']
                project = str(project) 
                row = row.to_dict()
                path = os.path.join('/grand/RNAtoImage/'+project+self.patch_data_path, WSI, WSI+'.h5')
                print('path is ..', path)
                label = np.asarray(row['Labels'])
                #print('label is .. ',label)
                patient_id = row['patient_id']
                label_encoded = row['label_encoded']
                print('label is .. ',label)
                print('label encoded is .. ',label_encoded)
                if not os.path.exists(path):
                    print(f'Not exist {path}')
                    continue
                try:
                    with h5py.File(path, "r") as f:
                        #n_patches = len(f)-2
                        n_patches = f['num_patches'][()]
                        #print('n_patches is ...', n_patches)
                except:
                    continue
                n_selected = min(n_patches, self.max_patches_total)
                print("n_patches selected is ...", n_selected)
                n_patches= list(range(n_patches))
                ## n_patches_index  same as images in read_data_hdf5
                random.seed(41)
                images = random.sample(n_patches, n_selected)
                #print('images is ..', images)
                #print('bag size', self.bag_size)
                self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
                self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images),'tcga_project': project,
                                    'wsi_path': path, 'patient_id': patient_id,'label_encoded': label_encoded})
                for k in range(len(images) // self.bag_size):
                    #print('k is ..',k)
                    self.index.append((WSI, path, self.bag_size * k, label_encoded,project))
        def shuffle(self):
            for k in self.data.keys():
                wsi_row = self.data[k]
                np.random.shuffle(wsi_row['images'])
        def __len__(self):
            return len(self.index)
        def __getitem__(self, idx):
            (WSI, wsi_path, i, label_encoded, project) = self.index[idx]
            imgs = []
            row = self.data[WSI]
            with h5py.File(wsi_path, 'r') as h5_file:
                # if self.normalize:
                #     imgs_aux = [self.normalizer.transform(h5_file[str(patch)][:])for patch in row['images'][i:i + self.bag_size]]
                #     imgs = [self.transforms(torch.from_numpy(img).permute(2,0,1)) for img in imgs_aux]
                # else:
                imgs = [self.transform(Image.fromarray(cv2.cvtColor(h5_file[str(patch)][:],cv2.COLOR_BGR2RGB))) for patch in row['images'][i:i + self.bag_size]]
            image = torch.stack(imgs, dim=0)
            if self.return_ids:
                return image, label_encoded, row['patient_id']+f'_{label_encoded}'
            out = {
            'image': image,
            'label': label_encoded,
            'project' : project
            #'label_val': self.label_vals[idx],
            #'filename': self.filenames[idx]
            }
            return out
            #return img, label

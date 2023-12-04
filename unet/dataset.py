import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset
import os
import torch

# Inspiration from segmentation models camvid (https://github.com/qubvel/segmentation_models.pytorch)

class Dataset(BaseDataset):
    CLASSES = ["Matrix", "Austenite", "Martensite/Austenite", "Precipitate", "Defect"]

    def __init__(self, images_dir, mask_dir, split_list=[], classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        if len(split_list) == 0:
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(mask_dir, image_id) for image_id in self.ids]
        else:
            self.images_fps = [os.path.join(images_dir, image_id) if image_id in split_list else None for image_id in self.ids]
            self.images_fps = [i for i in self.images_fps if i != None]
            self.masks_fps = [os.path.join(mask_dir, image_id) if image_id in split_list else None for image_id in self.ids]
            self.masks_fps = [i for i in self.masks_fps if i != None]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. matrix)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('int')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        assert (len(self.images_fps) == len(self.masks_fps))
        return len(self.images_fps)

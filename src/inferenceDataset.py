import os 
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image

import torchvision.transforms as T

transform = nn.Sequential(
	T.Resize((228, 228))
)

class InferenceDataset(Dataset):
	def __init__(self, all_img_of_obj, crop_img_of_obj, transform=transform, **kwargs):
		self.transform = transform
		self.all_img_of_obj = self.transform(all_img_of_obj)
		self.crop_img_of_obj = self.transform(crop_img_of_obj)
		
	def __getitem__(self, index):
		img0 = self.all_img_of_obj[index]
		img1 = self.crop_img_of_obj
		return  img0, img1
	
	def __len__(self):
		return self.all_img_of_obj.shape[0]
	
"""
input: folder_path - folder of 1 or more images 
output: torch tensor of the images 
"""
def load_images_from_folder(folder_path, transform=transform): 
	images_list = []
	for file_path in os.listdir(folder_path): 
		tensor = transform(read_image(str(Path(folder_path) / file_path)))
		images_list.append(tensor[:3, :, :])
	return torch.stack(images_list)
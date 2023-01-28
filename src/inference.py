import os 
from pathlib import Path
import random
import numpy as np

import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader 
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights 

from model import * 

from torch.utils.data import Dataset

# current iteration read images from 2 separate folders and return true/false for each pair of images  
# tech debt: load tensor directly from the video instead of from folders 
class testDataset(Dataset):
	def __init__(self, object_folder_path, room_folder_path, transform=None, **kwargs):
		self.object_folder_path = object_folder_path
		self.room_folder_path = room_folder_path
		self.object_file_names =  os.listdir(object_folder_path)
		self.room_file_names =  os.listdir(room_folder_path)
		self.transform = transform
		self.data = self.create_data()

	def __getitem__(self, index):
		return self.data[0][index], self.data[1][index] 

	def __len__(self):
		return len(self.object_file_names) * len(self.room_file_names)
	
	def create_data(self): 
		object_list = []
		room_list = [self.transform(read_image(str(Path(self.room_folder_path) / img_name))) for img_name in self.room_file_names]

		for object_image in self.object_file_names: 
			object_tensor = self.transform(read_image(str(Path(self.object_folder_path) / object_image))).repeat(len(self.room_file_names), 1, 1, 1)
			object_list.extend(object_tensor) 

		object_tensor = torch.stack(object_list)
		room_tensor = torch.stack(room_list * len(self.object_file_names))

		return object_tensor, room_tensor



def inference(): 

	torch.manual_seed(0)
	random.seed(0)
	np.random.seed(0)
	torch.use_deterministic_algorithms(True)

	base_model = efficientnet_v2_s 
	base_model_weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
	model = SiameseModel(base_model=base_model, base_model_weights=base_model_weights)

	model_file_path = '/Users/racheltay/Documents/school/cp5105-capstone-proj/src/siamese_model_e25_b8_lr1e-05_num8192_emb20.pth'
	model.load_state_dict(torch.load(model_file_path))
	
	transforms = nn.Sequential(
		T.Resize((228, 228))
	)

	room_folder_path = '/Users/racheltay/Documents/school/cp5105-capstone-proj/data/inference_test/room'
	object_folder_path = '/Users/racheltay/Documents/school/cp5105-capstone-proj/data/inference_test/bottle'

	test_dataset = testDataset(object_folder_path, room_folder_path, transform=transforms)
	testloader = DataLoader(test_dataset, batch_size=10) #, shuffle = True, batch_size = b)

	with torch.no_grad():
		for _, data in enumerate(testloader):
			img0, img1 = data
			output = model(img0, img1).squeeze(1)
			result = torch.sigmoid(output) > 0.5
			print("are images the same:", result)
			print("are images the same:", sum(result)/len(result)*100, "%")
			
			break 

if __name__ == '__main__':
	inference()
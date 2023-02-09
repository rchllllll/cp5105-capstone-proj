import random
import numpy as np
from torch.utils.data import DataLoader 
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights 

from model import * 
from inferenceDataset import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

# tech debt: making it relative path 
model_file_path = '/Users/racheltay/Documents/school/cp5105-capstone-proj/src/siamese_model_e25_b8_lr1e-05_num8192_emb20.pth'

class SiameseNetworkAPI(): 
	def __init__(self, obj_tensor, room_img):
		self.obj_tensor = obj_tensor
		self.room_img = room_img
		self.obj_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
		self.siamese_network_model = SiameseModel(
			base_model=efficientnet_v2_s, 
			base_model_weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
			)
		self.siamese_network_model.load_state_dict(torch.load(model_file_path)) 

	def inference(self): 
		objects_in_room = self.obj_detection_model(self.room_img)
		all_xy_coords = []
		all_conf_scores = []

		for obj in objects_in_room.crop(save=False):
			xy_coords = obj['box']
			crop_array = obj['im']
			crop_img_of_obj = torch.from_numpy(crop_array.transpose((-1, 0, 1)).copy())

			dataset = inferenceDataset(self.obj_tensor, crop_img_of_obj, transform)
			dataloader = DataLoader(dataset, batch_size=self.obj_tensor.shape[0])
			
			with torch.no_grad():
				for _, data in enumerate(dataloader):
					img0, img1 = data 
					output = self.siamese_network_model(img0, img1)
					final = torch.sigmoid(output) > 0.5
					all_xy_coords.append(xy_coords)
					all_conf_scores.append(final.sum() / final.shape[0])
					
		return all_xy_coords, all_conf_scores
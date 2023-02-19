import random
import numpy as np
from torch.utils.data import DataLoader 
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights 

from model import * 
from InferenceDataset import *

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
		self.obj_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)
		# reference: https://github.com/ultralytics/yolov5/issues/5936
		# use half precision FP16 inference to speed up inference 
		self.obj_detection_model.half = True
		self.siamese_network_model = SiameseModel(
			base_model=efficientnet_v2_s, 
			base_model_weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
			)
		self.siamese_network_model.load_state_dict(torch.load(model_file_path)) 

	def inference(self):
		# reference: https://github.com/ultralytics/yolov5/issues/5936
		# reduce image size to speed up inference
		objects_in_room = self.obj_detection_model(self.room_img, size=228)
		all_xy_coords = []
		all_conf_scores = []

		for obj in objects_in_room.crop(save=False):
			xy_coords = obj['box']
			crop_array = obj['im']
			crop_img_of_obj = torch.from_numpy(crop_array.transpose((-1, 0, 1)).copy())

			dataset = InferenceDataset(self.obj_tensor, crop_img_of_obj, transform)
			# reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
			# enable asynchronous data loading and data augmentation in separate worker subprocesses
			dataloader = DataLoader(dataset, batch_size=self.obj_tensor.shape[0], num_workers=2)
			
			# reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
			# disable gradient calculation for inference 
			with torch.no_grad():
				for _, data in enumerate(dataloader):
					img0, img1 = data 
					output = self.siamese_network_model(img0, img1)
					final = torch.sigmoid(output) > 0.5
					all_xy_coords.append(xy_coords)
					all_conf_scores.append(final.sum() / final.shape[0])
					
		return all_xy_coords, all_conf_scores
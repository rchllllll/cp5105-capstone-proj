import os 
from pathlib import Path

from inferenceDataset import *
from SiameseNetworkAPI import *

if __name__ == '__main__':
	model_file_path = '/Users/racheltay/Documents/school/cp5105-capstone-proj/src/siamese_model_e25_b8_lr1e-05_num8192_emb20.pth'
	main_folder = '/Users/racheltay/Documents/school/cp5105-capstone-proj/data/inference_test/image/'
	obj_folder = main_folder+'object/'
	room_folder = main_folder+'room/room1.png'

	all_img_of_obj = load_images_from_folder(obj_folder)

	print(SiameseNetworkAPI(all_img_of_obj, room_folder).inference())
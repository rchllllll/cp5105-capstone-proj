import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights 

from model import * 

def read_image(): 
    pass 

def inference(): 
    base_model = efficientnet_v2_s 
    base_model_weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = SiameseModel(base_model=base_model, base_model_weights=base_model_weights)

    model_file_path = 'siamese_model_e25_b8_lr1e-05_num8192_emb20.pth'
    model.load_state_dict(torch.load(model_file_path))
    
    transforms = nn.Sequential(
        T.Resize((228, 228))
    )

    with torch.no_grad():
        # read image 1 into tensor 
        # read image 2 into tensor 
        # run image 1 and image 2 through model 
        # return true/ false 
        pass 

if __name__ == '__main__':
    inference()
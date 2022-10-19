import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from config import *

class SiameseModel(nn.Module):
  
  def __init__(self, emb_size = EMB_SIZE): 
    super(SiameseModel, self).__init__()

    self.weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    self.siamese = efficientnet_v2_s(weights=self.weights)
    self.siamese.classifier = nn.Sequential(
        nn.Linear(self.siamese.classifier[1].in_features, 500),
        nn.ReLU(inplace = True),
        nn.Linear(500, emb_size)
    )
    self.classifier = nn.Sequential(
        nn.Linear(emb_size, 1),
        nn.Sigmoid()
    )

  def forward(self, img1, img2): 
    preprocess = self.weights.transforms()
    x1 = preprocess(img1)
    x2 = preprocess(img2)
    out1 = self.siamese(x1)
    out2 = self.siamese(x2)
    # multiply to get combined feature vector representing the similarity btwn the two
    combined_features = out1 * out2
    output = self.classifier(combined_features)
    return output
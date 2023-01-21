import torch 
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names

class SiameseModel(nn.Module):
  
  # potential configurable arguments 
  # channels: int, n_classes: int, dim_sizes: List[int], kernel_size: int, stride: int, padding: int, **kwargs
  # pretrained model and weights 
  def __init__(self, emb_size, base_model, base_model_weights): 
    super(SiameseModel, self).__init__()

    self.weights = base_model_weights
    self.siamese = base_model(weights=self.weights)
    train_nodes, eval_nodes = get_graph_node_names(self.siamese)
    # self.classifier = nn.Sequential(
    #     nn.Linear(self.siamese.get_submodule(train_nodes[-1]).out_features, 1),
    #     nn.Sigmoid()
    # )
    # self.classifier = nn.Linear(self.siamese.get_submodule(train_nodes[-1]).out_features * 2, 1)
    # self.classifier = nn.Linear(self.siamese.get_submodule(train_nodes[-1]).out_features, 1)
    self.feature_extractor = nn.Sequential(
        nn.Linear(self.siamese.get_submodule(train_nodes[-1]).out_features, 512),
        nn.ReLU(inplace = True),
        nn.Linear(512, emb_size)
    )
    self.classifier = nn.Linear(emb_size, 1)

  def forward(self, img1, img2): 
    preprocess = self.weights.transforms()
    x1 = preprocess(img1)
    x2 = preprocess(img2)
    out1 = self.siamese(x1)
    out2 = self.siamese(x2)
    # multiply to get combined feature vector representing the similarity btwn the two
    # combined_features = torch.cat((out1, out2), 1)
    # output = self.classifier(combined_features)
    fv1 = self.feature_extractor(out1)
    fv2 = self.feature_extractor(out2)
    diff = torch.abs(torch.sigmoid(fv1) - torch.sigmoid(fv2))
    output = self.classifier(diff)
    return output
import torch
import torch.nn as nn

class resnet50(nn.Module):
  def __init__(self,pretrained_model,num_classes):
    super(resnet50,self).__init__()
    self.pretrained_model = pretrained_model
    self.classifier = nn.Sequential(
        nn.Linear(self.pretrained_model.fc.in_features,512),
        # nn.Dropout(0.3),
        # nn.BatchNorm1d(1024),
        # nn.ReLU(),
        # nn.Linear(1024,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512,num_classes)
    )
    self.pretrained_model.fc = self.classifier

  def forward(self,x):
    output = self.pretrained_model(x)
    return output

class efficientnet(nn.Module):
  def __init__(self,pretrained_model,num_classes):
    super(efficientnet,self).__init__()
    self.pretrained_model = pretrained_model
    self.classifier = nn.Sequential(
        nn.Linear(self.pretrained_model.classifier.in_features,512),
        # nn.Dropout(0.3),
        # nn.BatchNorm1d(1024),
        # nn.ReLU(),
        # nn.Linear(1024,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512,num_classes)
    )
    self.pretrained_model.classifier = self.classifier

  def forward(self,x):
    output = self.pretrained_model(x)
    return output


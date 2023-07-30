import torch.nn as nn
import torch
import torchvision.models as models

#Loadung the model vgg19 that will serve as the base model
model=models.vgg19(pretrained=True).features

 #Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test data to model
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28'] 
        self.model=models.vgg19(pretrained=True).features[:29] 
       
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)    
        return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model=VGG().to(device).eval()
model.load_state_dict(torch.load('CNN_model.pth'))
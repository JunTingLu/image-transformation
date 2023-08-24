import torch.nn as nn
import torch
import torchvision.models as models

#Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=models.vgg19(pretrained=True)

# test data to model
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['1','3','8','13','20','29']
        #Loading the model vgg19 that will serve as the base model
        self.model=models.vgg19(pretrained=True).features[:30] 
       
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)    
        return features
   
model=VGG().to(device).eval()
# with open("/app/output/nst_cnn_model.pth","rb") as f:
    # torch.load(f)

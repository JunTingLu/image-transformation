import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


#Loadung the model vgg19 that will serve as the base model
model=models.vgg19(pretrained=True).features
# the vgg19 model has three components :
    #features: containg all the conv, relu and maxpool
    #avgpool: containing the avgpool layer
    #classifier: contains the Dense layer(FC part of the model) 


#Assigning the GPU to the variable device
device=torch.device("cuda")


#[0,5,10,19,28] are the index of the layers we will be using to calculate the loss as per the paper of NST
#Defining a class that for the model
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        #Here we will use the following layers and make an array of their indices
        # 0: block1_conv1
        # 5: block2_conv1
        # 10: block3_conv1
        # 19: block4_conv1
        # 28: block5_conv1
        self.req_features= ['0','5','10','19','28'] 
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers
    
   
    #x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        #initialize an array that wil hold the activations from the chosen layers
        features=[]
        #Iterate over all the layers of the mode
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)
                
        return features



#defing a function that will load the image and perform the required preprocessing and put it on the GPU
def image_loader(path):
    image=Image.open(path)
    #defining the image transformation steps to be performed before feeding them to the model
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    #The preprocessing steps involves resizing the image and then converting it to a tensor
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)



#Loading the original and the style image
original_image=image_loader("D:/Aaron/origin/america.jpg")
style_image=image_loader('D:/Aaron/style/style.jpg')

#Creating the generated image from the original image
generated_image=original_image.clone().requires_grad_(True)

def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2)#*0.5
    return content_l

def calc_style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    batch_size,channel,height,width=gen.shape

    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-A)**2)#/(4*channel*(height*width)**2)
    return style_l

def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        #extracting the dimensions from the generated image
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    #calculating the total loss of e th epoch
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss


#Load the model to the GPU
model=VGG().to(device).eval() 

#initialize the paramerters required for fitting the model
epoch=200
lr=0.004
# lr=0.0004
alpha=10
beta=90

#using adam optimizer and it will update the generated image not the model parameter 
optimizer=optim.Adam([generated_image],lr=lr)
# log with process
log_dir='logs_result/'
log=log_dir
#iterating for 1000 times
for e in range (epoch):
    
    writer = SummaryWriter(log)
    #extracting the features of generated, content and the original required for calculating the loss
    gen_features=model(generated_image)
    orig_feautes=model(original_image)
    style_featues=model(style_image)
    
    #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
    total_loss=calculate_loss(gen_features, orig_feautes, style_featues)
    #optimize the pixel values of the generated image and backpropagate the loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    writer.add_scalar('Loss', total_loss, e)
    print(134,writer)
    #print the image and save it after each 100 epoch, bc it would happen over-fitting when epoch>200
    # 若e除150的餘數!=0
    # if(not (e%150)):
    # 紀錄最小的loss epoch
    # min_loss=min(total_loss)
    if (e%150)==0:
        print(total_loss)
        save_image(generated_image,"D:/Aaron/gen.png")
    with open("/CNN_model.pth", "wb") as f:
      torch.save(model, f)
#%%
# 啟動tensorboard查看訓練曲線
""" inspect log """
# terminal 輸入
# %load_ext tensorboard
# %tensorboard --logdir=logs_result/
# %reload_ext tensorboard    
        
    
        

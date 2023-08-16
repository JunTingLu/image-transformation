from flask import Flask
from flask import request,send_file
import base64 
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import torch.optim as optim
from torchvision.utils import save_image
import base64
from flask_cors import CORS
import torch
from utils import *
from test import model

app = Flask(__name__)
CORS(app)

#Assigning the GPU to the variable device
device=torch.device("cuda")

@app.route('/', methods=['GET'])
def health_check():
    return 'success!'



# import torch.nn as nn
# import torch
# import torchvision.models as models

# #Assigning the GPU to the variable device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=models.vgg19(pretrained=True)

# # test data to model
# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG,self).__init__()
#         self.req_features= ['1','3','8','13','20','29']
#         #Loading the model vgg19 that will serve as the base model
#         self.model=models.vgg19(pretrained=True).features[:30] 
       
#     def forward(self,x):
#         features=[]
#         for layer_num,layer in enumerate(self.model):
#             x=layer(x)
#             if (str(layer_num) in self.req_features):
#                 features.append(x)    
#         return features
   
# model=VGG().to(device).eval()


# alpha=1
# beta=50

# # Loss function 
# def calc_content_loss(gen_feat,orig_feat):
#     #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
#     content_l=torch.mean((gen_feat-orig_feat)**2) #*0.5
#     return content_l


# def calc_style_loss(gen,style):
#     #Calculating the gram matrix for the style and the generated image
#     batch_size,channel,height,width=gen.shape
#     G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
#     A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t()) 
#     #Calculating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
#     style_l=torch.mean((G-A)**2) #/(4*channel*(height*width)**2)
#     return style_l


# def calculate_loss(gen_features, orig_features, style_features):
#     style_loss=content_loss=0
#     for gen,cont,style in zip(gen_features,orig_features,style_features):
#         content_loss+=calc_content_loss(gen,cont)
#         style_loss+=calc_style_loss(gen,style)
#     #calculating the total loss of e th epoch
#     total_loss=alpha*content_loss + beta*style_loss 
#     return total_loss



  
# 取得圖片之base64編碼並傳至後端，後端將base64轉換回圖片進行預測
def preprocess_img(data,type):
    str_data=str(data)
    # Search the image content and remove the title
    start_index = str_data.find("data:image/{};base64".format(str(type)))
    if start_index != -1:
        start_index += len("data:image/{};base64".format(str(type)))
    else:
        print('unknown type')    
    content= str_data[start_index:]
    # 對二進制進行編碼，生成base64字符串
    image_bytes = base64.b64decode(content)
    # 二進制處理
    img=Image.open(io.BytesIO(image_bytes))
    img2arr=np.array(img)
    print(39,img2arr.shape)
    img_array=check_RGB(img2arr)
    return img_array


# Judge the channel RGB
def check_RGB(input):
    print(50,input.shape)
    if input.shape[2] == 4:
        print('The image is 4-channels')
        return input[:, :, :3]  #choose the 3-channels
    else:
        return input


# Model train
def model_generate(origin_img,style_img,type):
    save_dir="./output/nst_cnn_model.pth"
    # Transform style_img to array
    gen_img=origin_img.clone().requires_grad_(True)
    optimizer=optim.Adam([gen_img],lr=opt.lr)
    epoch=opt.epoch
    #iterating for 200 times
    for e in range (epoch):
        gen_features=model(gen_img) 
        orig_features=model(origin_img)
        style_features=model(style_img) 
        total_loss=calculate_loss(gen_features, orig_features, style_features)
        total_loss.backward()
        optimizer.step() # update gen_img parameters
        if e==epoch-1:
            save_path="./output/nst.{}".format(str(type))
            save_image(gen_img,save_path)
            # Save pth
            with open(save_dir,'wb') as f:
                torch.save(model.state_dict(), f)
            return  save_path


# Resizing input image
def image_loader(input_img):
    # Transfer np.array to pillow (PIL)
    pil_image = Image.fromarray(np.uint8(input_img))
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image=loader(pil_image).unsqueeze(0)
    # Creating the generated image from the original image (copy origin img as gen_img)
    return image.to(device,torch.float)


@app.route('/img_backend', methods=['GET', 'POST'])
def upload_data():
    if request.method=='POST': 
        file=request.form['image']
        style=request.form['style']
        print(95,style)
        # Grab the sting of image type
        type=file.split(',')[0].split('/')[1].split(';')[0]
        # Preprocessing the image before training
        resized_img=preprocess_img(file,type)
        resized_img=image_loader(resized_img)
        # Decided the style image as input
        style_img=Image.open("./output/style/{}.png".format(style))
        style_img=image_loader(style_img)
        # Generate the output image
        save_path=model_generate(resized_img,style_img,type)
        return send_file(save_path, mimetype='image/{}'.format(type))


if __name__ == '__main__':
    host_ip='127.0.0.1'
    host_port='5000'
    app.run(host=host_ip,port=host_port,debug=True)

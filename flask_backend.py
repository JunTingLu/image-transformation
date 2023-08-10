from flask import Flask
from flask import request,send_file
# from flask_uploads import UploadSet, IMAGES
import base64 
import numpy as np
from torchvision import transforms
from PIL import Image
import io
from cv2 import imwrite
import torch.optim as optim
from torchvision.utils import save_image
import base64
from flask_cors import CORS
from utils import *
from test import model

app = Flask(__name__)
CORS(app)

#Assigning the GPU to the variable device
device=torch.device("cuda")

@app.route('/', methods=['GET'])
def health_check():
    return 'success!'
        
  
# 取得圖片之base64編碼並傳至後端，後端將base64轉換回圖片進行預測
def preprocess_img(data):
    str_data=str(data)
    print(34,data)
    # 搜尋圖片內容 (移除標頭)
    start_index = str_data.find("data:image/png;base64")
    if start_index != -1:
        start_index += len("data:image/png;base64")
    # utf-8 生成二進制 
    content= str_data[start_index:]
    # 對二進制進行編碼，生成base64字符串
    image_bytes = base64.b64decode(content)
    # 二進制處理
    img=Image.open(io.BytesIO(image_bytes))
    img2arr=np.array(img)
    print(39,img2arr.shape)
    return img2arr


# Judge the channel RGB
def check_RGB(input):
    pass


# Model train
def model_generate(origin_img,style_img):
    save_dir="/logs_result/nst_cnn_model.pth"
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
            save_path="nst_{}.png".format(e)
            save_image(gen_img,save_path)
            return  save_path
    with open(save_dir) as f:
        torch.load(model.state_dict(), f)


def image_loader(input_img):
    # Transfer np.array to pillow (PIL)
    pil_image = Image.fromarray(np.uint8(input_img))
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image=loader(pil_image).unsqueeze(0)
    #Creating the generated image from the original image (copy origin img as gen_img)
    return image.to(device,torch.float)


@app.route('/img_backend', methods=['GET', 'POST'])
def upload_data():
    if request.method=='POST': 
        file=request.form['image']
        style=request.form['style']
        # Preprocessing the image before training
        resized_img=preprocess_img(file)
        resized_img=image_loader(resized_img)
        # Decided the style image as input
        style_img=Image.open("output/style/{}.png".format(style))
        style_img=image_loader(style_img)
        # Generate the output image
        save_path=model_generate(resized_img,style_img)
        return send_file(save_path, mimetype='image/png')


if __name__ == '__main__':
    host_ip='127.0.0.1'
    host_port='5000'
    app.run(host=host_ip,port=host_port,debug=True)

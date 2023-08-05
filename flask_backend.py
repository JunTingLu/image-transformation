import os
from flask import Flask
from flask import request, redirect, url_for, render_template,send_from_directory,jsonify
# from flask_uploads import UploadSet, IMAGES
import base64 
from werkzeug.utils import secure_filename
import numpy as np
import cv2
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

app.config['UPLOAD_FOLDER'] ='../static/output/'  # 文件储存地址
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 限制大小 24MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#Assigning the GPU to the variable device
device=torch.device("cuda")

@app.route('/', methods=['GET'])
def health_check():
    return 'success!'


#檢查上傳檔案是否合法的函數
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS        

  
# 取得圖片之base64編碼並傳至後端，後端將base64轉換回圖片進行預測
def preprocess_img(data):
    print(42,data)
    str_data=str(data)
    # 搜尋圖片內容 (移除標頭)
    start_index = str_data.find("data:image/png;base64")
    if start_index != -1:
        start_index += len("data:image/png;base64")
    # utf-8 生成二進制 
    content= str_data[start_index:]

    # 對二進制進行編碼，生成base64字符串
    image_bytes = base64.b64decode(content)
    print(44,image_bytes)
    # 二進制處理
    img=Image.open(io.BytesIO(image_bytes))
    img2arr=np.array(img)
    print(39,img2arr.shape)
    return img2arr


def show_img(input_img):
    """ 將轉換照片回傳前端"""
    success,encoded_img=cv2.imencode('.jpg', input_img)
    # 暫存圖片的二進位資料
    buffer=io.BytesIO()
    buffer.write(encoded_img)
    # 重新將資料讀寫指針移動到初始位置
    buffer.seek(0)
    # 將圖轉為base64編碼
    img_b64=base64.b64encode(buffer.getvalue()).decode()
    return  img_b64


""" Model evaluation"""
# model train
def model_generate(origin_img,style_img):
    target_dir="nst_cnn_model.pth"
    if os.path.exists(target_dir):
        with open(target_dir) as f:
            torch.load(model.state_dict(), f)
    else:
        pass
    # transform style_img to array
    gen_img=origin_img.clone().requires_grad_(True)
    optimizer=optim.Adam([gen_img],lr=opt.lr)
    epoch=200
    #iterating for 1000 times
    for e in range (epoch):
        #extracting the features of generated, content and the original required for calculating the loss
        gen_features=model(gen_img) 
        orig_features=model(origin_img)
        style_features=model(style_img) 
        #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
        total_loss=calculate_loss(gen_features, orig_features, style_features)
        #optimize the pixel values of the generated image and back-propagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step() # update gen_img parameters
        if e==epoch-1:
            save_image(gen_img,"nst_{}.png".format(e))
            return  gen_img


def image_loader(input_img):
    # image=Image.open(input_img)
    # normalization 

    # transfer np.array to pillow (PIL)
    pil_image = Image.fromarray(np.uint8(input_img))
    #defining the image transformation steps to be performed before feeding them to the model
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    #The preprocessing steps involves resizing the image and then converting it to a tensor
    image=loader(pil_image).unsqueeze(0)
    #Creating the generated image from the original image (copy origin img as gen_img)
    # generated_image=image.clone().requires_grad_(True)
    return image.to(device,torch.float)


@app.route('/img_backend', methods=['GET', 'POST'])
def upload_data():
    path_list=[]
    # 初始化img_b64為空值
    img_b64=""
    img_process=""
    if request.method=='POST': 
        file=request.form['image']
        style=request.form['style']
        print(187,file)
        # 圖像處理
        resized_img=preprocess_img(file)
        print(140, resized_img)
        resized_img=image_loader(resized_img)
        print(134)
        # 進行圖像轉換
        style_img=Image.open("./output/style/{}.jpg".format(style))
        style_img=image_loader(style_img)

        # gen_img=model_generate(resized_img,style_img)
        # print(142,gen_img.shape)
        # 將圖片返回前端模板
        # img_b64=show_img(gen_img)
        # print(img_b64)         
        # index.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(index.filename)))    
        # return jsonify({'data':{'result':img_b64,'type':'image'}})
        return 'end'



if __name__ == '__main__':
    host_ip='127.0.0.1'
    host_port='5000'
    app.run(host=host_ip,port=host_port,debug=True)
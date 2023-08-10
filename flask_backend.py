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

#Assigning the GPU to the variable device
device=torch.device("cuda")

@app.route('/', methods=['GET'])
def health_check():
    return 'success!'
        
  
# 取得圖片之base64編碼並傳至後端，後端將base64轉換回圖片進行預測
def preprocess_img(data):
    str_data=str(data)
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


""" 將轉換照片回傳前端"""
def show_img(input_img):
    # 將CUDA Tensor複製到主機內存
    input_numpy=torch.tensor(input_img).cpu().numpy()
    print(64,numpy)
    # 將 Tensor 轉換為 NumPy 數組後，確保 C 順序
    input_numpy=np.ascontiguousarray(input_numpy)
     # 將 NumPy 數組轉換為 base64 編碼的字符串，並編成png格式
    _, encoded_img = cv2.imencode('.png', input_numpy)
    # 通常json只能處理utf-8字串形式
    base64_image = base64.b64encode(encoded_img).decode('utf-8')
    
    # 暫存圖片的二進位資料
    #buffer=io.BytesIO()
    #buffer.write(input_bytes)
    # 重新將資料讀寫指針移動到初始位置
    #buffer.seek(0)
    # 將圖轉為base64編碼
    # img_b64=base64.b64encode(buffer.getvalue()).decode('utf-8')
    return  img_b64


""" Model evaluation"""
# model train
def model_generate(origin_img,style_img):
    save_dir="/logs_result/nst_cnn_model.pth"
    # transform style_img to array
    gen_img=origin_img.clone().requires_grad_(True)
    optimizer=optim.Adam([gen_img],lr=opt.lr)
    epoch=opt.epoch
    #iterating for 1000 times
    for e in range (epoch):
        gen_features=model(gen_img) 
        orig_features=model(origin_img)
        style_features=model(style_img) 
        total_loss=calculate_loss(gen_features, orig_features, style_features)
        total_loss.backward()
        optimizer.step() # update gen_img parameters
        if e==epoch-1:
            save_image(gen_img,"nst_{}.png".format(e))
            return  gen_img
    torch.save(model.state_dict(), save_dir)


def image_loader(input_img):
    # Transfer np.array to pillow (PIL)
    pil_image = Image.fromarray(np.uint8(input_img))
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image=loader(pil_image).unsqueeze(0)
    #Creating the generated image from the original image (copy origin img as gen_img)
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
        # 圖像處理
        resized_img=preprocess_img(file)
        resized_img=image_loader(resized_img)
        print(123,resized_img.shape)
        # 進行圖像轉換
        style_img=Image.open("output/style/{}.png".format(style))
        style_img=image_loader(style_img)

        gen_img=model_generate(resized_img,style_img)
        print(139,gen_img.shape)
        # 將圖片返回前端模板
        img_b64=show_img(gen_img)
        # image=Image.open(input_img)
        print(142,img_b64)    
        return jsonify({'data':{'result':img_b64,'type':'image'}})


if __name__ == '__main__':
    host_ip='127.0.0.1'
    host_port='5000'
    app.run(host=host_ip,port=host_port,debug=True)

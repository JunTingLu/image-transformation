import os
from flask import Flask, request, redirect, url_for, render_template,send_from_directory,jsonify
# from flask_uploads import UploadSet, IMAGES
import base64 
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import io
from cv2 import imwrite
import torch
from torch import unsqueeze
import torch.nn as nn
from main import GeneratorResNet, ResidualBlock

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] ='C:/Users/User/Desktop/python_jupyter/for_job/static/files/'  # 文件储存地址
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 限制大小 24MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


#檢查上傳檔案是否合法的函數
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS        

  
# 取得圖片之base64編碼並傳至後端，後端將base64轉換回圖片進行預測
def proccessing_img(data):
    # 讀取圖片內容
    data = request.files['file'].read()
    # 將data(bytes) 字串寫入緩衝區，並轉成PIL物件
    img=Image.open(io.BytesIO(data))
    print('img',img)
    img2arr=np.array(img)
    resized=cv2.resize(img2arr,(256,256))
    print(resized.shape)
    """ 測試將原圖轉成灰階 """
    # gray_img=img.convert('L')
    return resized


def show_img(input_img):
    """ 測試將原圖轉成灰階 """
    # 因為gray_img為一個Image物件，需將gray_img轉為nd.array形式
    # gray_img=np.array(input_img)
    # print('shape',gray_img.shape)
    # 測試將gray_img 編成jpeg格式
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


# 預先載入model
def load_model(input_img):
    print(input_img.shape)
    num_residual_blocks=9
    in_features=256
    # 引入模型
    # model=cycleGAN_model()
    # 模型加載至cpu上
    model_state=torch.load('./cycleGAN_demo/static/pth-filesG_BA.pth',map_location=torch.device('cpu'))
    # model.eval()
    # 導入權重
    # model.load_state_dict(model_state)
    # preprocessing input img
    transform = transforms.Compose([
    # 將PIL Image轉換為Tensor，並除255使數值界於0~1之間
    transforms.ToTensor(),
    # 標準化將圖片RGB縮放到-1到1之間 (0.5,0.5,0.5) 為均值;(0.5,0.5,0.5)為標準差
    # 產生均值為0標準差為1的分布
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    img_transformed=transform(input_img)
    # 在0維上增加新的一維
    img_tensor = img_transformed.unsqueeze(0)
    # 進行轉換 
    with torch.no_grad():
        # output_img = model(img_tensor)
        
        return True



@app.route('/uploaded', methods=['GET', 'POST'])
def upload_data():
    path_list=[]
    # 初始化img_b64為空值
    img_b64=''
    # 接收json 參數類型
    if request.method=='POST': 
        img=request.files.getlist('file')
        path_list.append(img)
        for index in img:
            if allowed_file(index.filename):
                # 圖片處理
                img_process=proccessing_img(index)
                # 進行圖像轉換
                generated_img=load_model(img_process)
                # print(generated_img.shape)
                # 將圖片返回前端模板
                img_b64=show_img(img_process)
                print(img_b64)         
                # index.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(index.filename)))    
    return render_template('img_upload.html',img_b64=img_b64)
    # return jsonify({'data':{'result':img_b64,'type':'image'}})


if __name__ == '__main__':
    host=''
    port=''
    app.run(debug=True)
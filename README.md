# Image-style transformation
**利用前後端串接實作一個能夠在網頁上進行圖片風格轉換，搭配及時拍照或式剪裁功能，並支持四種不同畫風(梵谷、莫內、素描、動漫)，達到轉換圖片風格的體驗感**

# 開發環境
>- python 3.9
>- cuda11.0
>- torch 1.7.1
>- torchvision 0.8.2

# 使用流程及說明
**1. 建議在cuda環境下執行
**2. 點選欲轉換的風格圖 (default 為梵谷星空風格)**<br>
**3. 點選tools中的crop進行圖像剪裁，或是take photo進行及時照相**<br>
**4. 點選submit，進行風格轉換**<br>
**(pyhon建議能調用cuda資源，加速圖像生成)**

<table>
<tr>
  <td>Function</td>
  <td>Status</td>
</tr>
<tr>
  <td>
    Upload image
  </td>
  <td>
    上傳圖片
  </td>
</tr>
  <tr>
  <td>
    Crop image
  </td>
  <td>
    剪裁圖片
  </td>
</tr>
    <tr>
  <td>
    Submit
  </td>
  <td>
    送出圖片，並進行風格轉換
  </td>
</tr>
  <tr>
  <td>
    Save image
  </td>
  <td>
    儲存生成圖片
  </td>
</tr>
  </tr>
  <tr>
  <td>
    Clear
  </td>
  <td>
    清除當前生成圖片
  </td>
</tr>
</table>

# 使用API
>croppie.js (前端使用croppie.js套件進行圖像剪裁)

# Get start
Build the docker image 
```

```
Run the ducker image you've build
```


```

// 剪裁圖片
(function($) {
//     var width_crop = 256, // 圖片裁切寬度 px 值
//      height_crop = 256, // 圖片裁切高度 px 值
//      type_crop = "square", // 裁切形狀: square 為方形, circle 為圓形
//      width_preview = 400, // 預覽區塊寬度 px 值
//      height_preview = 400, // 預覽區塊高度 px 值
//      compress_ratio = 0.85, // 圖片壓縮比例 0~1
//      myCrop, file, oldImgDataUrl;
   
//       // 裁切初始參數設定
//      myCrop = $("#oldImg").croppie({
//        viewport: { // 裁切區塊
//        width: width_crop,
//        height: height_crop,
//        type: type_crop
//        },
//        boundary: { // 預覽區塊
//        width: width_preview,
//        height: height_preview
//        }
//        });
       
       //       預覽圖片
//    $('input').on('change',function(e){
//             const file = this.files[0];
//             const fr = new FileReader();
//             fr.readAsDataURL(file);
//             fr.onload = function (e) {
//             oldImgDataUrl = e.target.result;
//             oldImg.src = oldImgDataUrl; // 載入 oldImg 取得圖片資訊
//         myCrop.croppie("bind", {
//         url: oldImgDataUrl
//         });
       
//      $('img').attr('src', e.target.result);
//      }});    
   
    //    function displayCropImg(src) {
    //    var html = "<img src='" + src + "' />";
    //    $("#newImg").html(html);
    //    }
       
    //    oldImg.onload = function() {
    //    var width = this.width,
    //    height = this.height,
    //    fileSize = Math.round(file.size / 1000),
    //    html = "";
    //    html += "<p>原始圖片尺寸 " + width + "x" + height + "</p>";
    //    html += "<p>檔案大小約 " + fileSize + "k</p>";
    //    $("#oldImg").before(html);
    //    }
   
    //    function displayNewImgInfo(src) {
    //    var html = "",
    //    filesize = src.length * 0.75;

    //    html += "<p>裁切圖片尺寸 " + width_crop + "x" + height_crop + "</p>";
    //    html += "<p>檔案大小約 " + Math.round(filesize / 1000) + "k</p>";
    //    $("#newImgInfo").html(html);
    //    }
    var $uploadCrop,
        tempFilename,
        rawImg,
        imageId;
        function readFile(input) {
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                reader.onload = function (e) {
                    $('.upload-demo').addClass('ready');
                    $('#cropImagePop').modal('show');
                    rawImg = e.target.result;
                }
                reader.readAsDataURL(input.files[0]);
            }
            else {
                swal("Sorry - you're browser doesn't support the FileReader API");
            }
        }

    $uploadCrop = $('#upload-demo').croppie({
        viewport: {
            width: 150,
            height: 200,
        },
        enforceBoundary: false,
        enableExif: true
    });

    $('#cropImagePop').on('shown.bs.modal', function(){
        // alert('Shown pop');
        $uploadCrop.croppie('bind', {
            url: rawImg
        }).then(function(){
            console.log('jQuery bind complete');
        });
    });

    $('#upload-file').on('change', function () { imageId = $(this).data('id'); tempFilename = $(this).val();
    // $('#cancelCropBtn').data('id', imageId); readFile(this); 
    });
   
    $("#crop_img").on("click", function() {
       $uploadCrop.croppie("result", {
       type: "canvas",
       format:'jpeg',
       quality: 1 //0~1
       }).then(function(resp) {
            // displayCropImg(src);
            $('#oldImg').attr('src', resp);
            $('#cropImagePop').modal('hide');
        });
        });

    })(jQuery);;
 

// 預覽圖片
    const myFile = document.querySelector('#upload-file')
        myFile.addEventListener('change', function(e){
            const file = e.target.files[0]
            const img = document.querySelector('#oldImg')
            const img_display=document.querySelector('#newImg')
            img.src = URL.createObjectURL(file)
            img.clientWidth=256;
            img.clientHeight=256;
        fetch('http://127.0.0.1:5000/uploaded',{
            method:'POST',
            body:JSON.stringify({data:img}),
            headers:{
                'Content-Type':'application/json'
            }  
        })
        .then(response=>response.json()) // 將回傳文字轉成json格式
        //解構 data
        .then(({data})=>{ 
            // 判斷回傳是否為圖片url
            if (data.type==='image'){
                img_display.src=data.image;
                return 
            }        
        });
    });


// 下載轉換後的圖片(Blob URL)
    // create 'a' label
    const download_file = document.querySelector('#download')
    // const new_img=document.querySelector('#newImg')
    // 若oldimg為空才能下載圖片，確保下載到上傳的預覽圖片


    download_file.addEventListener('click', function(e){
    const downloadLink = document.createElement('a')
    // 將 a 標籤的連結改為 Blob URL
    downloadLink.href = URL.createObjectURL(file)
    // 將下載的檔名設定為 file
    downloadLink.download = 'download-image'
    // 點擊標籤
    downloadLink.click()
    });

// 清除圖片
    const renew_img=document.querySelector('#newImg')
    const resetbtn=document.querySelector('#reset')
    resetbtn.addEventListener('click',function(e){
        if (newImg.src!=''){
            renew_img.src="";
        }
    });



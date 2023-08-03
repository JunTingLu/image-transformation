var $uploadCrop,
    rawImg,
    data;

    function readFile(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                rawImg = e.target.result;
                $('#oldImg').attr('src',rawImg)
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $uploadCrop = $('#upload-demo').croppie({
        // 指定裁剪區域的寬度和高度
        viewport: {
            width: 256,
            height: 256,
        },
        boundary: {
            width: 400,
            height: 400
        },
        enforceBoundary: false, //可以超出裁剪區域
        enableExif: true//支援圖片的 Exif 元數據
    });
    
    // 綁地欲剪裁圖片
    $('#cropImagePop').on('shown.bs.modal', function(){
        $uploadCrop.croppie('bind', {
            url: rawImg,
        }).then(function(){
            console.log('bind complete');
        });
        console.log(32,$uploadCrop)
    });

    // 監聽上傳檔案
    $('#upload-file').on('change', function () { 
        readFile(this);
    });

    // 跳出剪裁框
    $('#crop_img').on('click', function () { 
        $('#cropImagePop').modal('show');
    });
    
    // 剪裁後貼至upload image 中
    $('#cropImageBtn').on('click', function (ev) {
        $uploadCrop.croppie('result', {
            type: 'base64',
            format: 'jpeg',
            size: {width:  256, height: 256}
        }).then(function (resize) {
            $('#oldImg').attr('src', resize);
            $('#cropImagePop').modal('hide');
        });
    });

    // submit button


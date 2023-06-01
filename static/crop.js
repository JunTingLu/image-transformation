var $uploadCrop,
    tempFilename,
    rawImg,
    imageId;

    function readFile(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                // $('#cropImagePop').modal('show');
                rawImg = e.target.result;
                console.log(12,rawImg)
            }
            reader.readAsDataURL(input.files[0]);
            console.log(13,reader)
        }
    }

    $uploadCrop = $('#upload-demo').croppie({
        // 指定裁剪區域的寬度和高度
        viewport: {
            width: 350,
            height: 300,
        },
        enforceBoundary: false, //可以超出裁剪區域
        enableExif: true//支援圖片的 Exif 元數據
    });
    
    // 綁地欲剪裁圖片
    // $('#cropImagePop').on('shown.bs.modal', function(){
    //     $uploadCrop.croppie('bind', {
    //         url: rawImg
    //     }).then(function(){
    //         console.log('bind complete');
    //     });
    //     console.log(32,$uploadCrop)
    // });


    $('#upload-file').on('change', function () { 
        // imageId = $(this).data('id'); 
        // tempFilename = $(this).val();
        // $('#cancelCropBtn').data('id', imageId);
        readFile(this);

    });

    $('#crop_img').on('click', function () { 
        $('#cropImagePop').modal('show');
    });
    
    // 剪裁後貼至upload image 中
    $('#cropImageBtn').on('click', function (ev) {
        console.log(48)
        $uploadCrop.croppie('result', {
            type: 'base64',
            format: 'jpeg',
            size: {width: 300, height: 300}
        }).then(function (resp) {
            $('#oldImg').attr('src', resp);
            $('#cropImagePop').modal('hide');
        });
        console.log(57,$uploadCrop)
    });

// export {
//     rawImg 
// };
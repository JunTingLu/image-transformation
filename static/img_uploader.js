// 預覽圖片
const myFile = document.querySelector('#upload-file')
const submit=document.querySelector('#submitbtn')
const img = document.querySelector('#oldImg')
const img_display=document.querySelector('#newImg')

// function upload_file(){
//     myFile.addEventListener('change', function(e){
//         selectedFile = e.target.files[0]
//         img.src = URL.createObjectURL(selectedFile)
//         img.clientWidth=256;
//         img.clientHeight=256;  
//         console.log(12,img)
//     });
// }   

// function submited(){
//     submit.addEventListener('click',()=>{
//         const Data=new FormData()
//         Data.append('input',selectedFile)
//     fetch('http://127.0.0.1:5000/uploaded',{
//             method:'POST',
//             body: Data,
//             headers:{
//                 'Content-Type':'multipart/form-data'
//             }  
//     })
//     .then(response=>response.json()) // 將回傳文字轉成json格式
//     //解構 data
//     .then(({data})=>{ 
//         console.log(27,data.result)
//         // 判斷回傳是否為圖片url
//         if (data.type==='image'){
//             img_display.src=data.result;
//             return 
//         }        
//     }).catch(error,()=>{
//         console.log(error)
//     })
// });
// }


// Choose image style 
const styleButtons = document.querySelectorAll(".card-btn");
style = "vango"; // 預設值
styleButtons.forEach(btn => {
    btn.addEventListener("click", (e) => {
        style = e.target.id;
        // 只保留feature當前選擇的字串
        console.log(55, style);
    });
});


$('#submitbtn').on('click', function (ev) {
    var img_url =img.src;
    data=new FormData()
    data.append('image',img_url)
    data.append('style',style)
    fetch('http://127.0.0.1:5000/img_backend',{
        method:'POST',
        body:data
    })
    .then((response) => response.blob()) //transfer the image file from flask with blob
    .then((blob)=>{ 
            const url = window.URL.createObjectURL(blob); // use url to read the blob concept
            const img = document.createElement('img');
            img_display.src = url;
            document.body.appendChild(img)
            
            // Create a download link for the image
            const downloadLink = document.createElement('a');
            downloadLink.href =url;  // Use the back_img URL as the download link
            downloadLink.download = 'downloaded-image.jpg'; // Set the download filename
            // Trigger the click event for the download link
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            window.URL.revokeObjectURL(url);          
    });
})


// 清除圖片
const renew_img=document.querySelector('#newImg')
const resetbtn=document.querySelector('#reset')
resetbtn.addEventListener('click',function(e){
    if (newImg.src!=''){
        renew_img.src="";
    }
});



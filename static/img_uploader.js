
// 預覽圖片
    const myFile = document.querySelector('#upload-file')
    const submit=document.querySelector('#submitbtn')
    const img = document.querySelector('#oldImg')
    const img_display=document.querySelector('#newImg')
    let selectedFile;

function upload_file(){
    myFile.addEventListener('change', function(e){
        selectedFile = e.target.files[0]
        img.src = URL.createObjectURL(selectedFile)
        img.clientWidth=256;
        img.clientHeight=256;  
        console.log(12,img)
    });
}   

function submited(){
    submit.addEventListener('click',()=>{
        const Data=new FormData()
        Data.append('input',selectedFile)
    fetch('http://127.0.0.1:5000/uploaded',{
            method:'POST',
            body: Data,
            headers:{
                'Content-Type':'multipart/form-data'
            }  
    })
    .then(response=>response.json()) // 將回傳文字轉成json格式
    //解構 data
    .then(({data})=>{ 
        console.log(27,data.result)
        // 判斷回傳是否為圖片url
        if (data.type==='image'){
            img_display.src=data.result;
            return 
        }        
    }).catch(error,()=>{
        console.log(error)
    })
});
}

// function  upload_file(){
//     submited();
// }
upload_file();
submited();

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



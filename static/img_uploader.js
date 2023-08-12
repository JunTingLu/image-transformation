const submit=document.querySelector('#submitbtn')
const img = document.querySelector('#oldImg')
const img_display=document.querySelector('#newImg')
const download=document.querySelector('#download')
let url;

// Choose image style 
const styleButtons = document.querySelectorAll(".card-btn");
style = "vango"; // default type
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
    console.log(img_url)
    fetch('http://127.0.0.1:5000/img_backend',{
        method:'POST',
        body:data,
    })

    .then((response) => response.blob()) //transfer the image file from flask with blob
    .then((blob)=>{ 
            url = window.URL.createObjectURL(blob); // use url to read the blob concept
            img_display.src = url;  
    });
})


// Download image
download.addEventListener('click',()=>{
// Create a download link for the image
const downloadLink = document.createElement('a');
downloadLink.href =url;  // Use the back_img URL as the download link
downloadLink.download = 'downloaded-image.jpg'; // Set the download filename
// Trigger the click event for the download link
document.body.appendChild(downloadLink);
downloadLink.click();
document.body.removeChild(downloadLink);
window.URL.revokeObjectURL(url); 
})


// Clear the current image 
const renew_img=document.querySelector('#newImg')
const resetbtn=document.querySelector('#reset')
resetbtn.addEventListener('click',function(e){
    if (newImg.src!=''){
        renew_img.src=''
        refreshUrl="https://dummyimage.com/3s00x300/f7f2f7/a2a3ab/&text=Transferred Image"
        renew_img.src=refreshUrl;
    }
});



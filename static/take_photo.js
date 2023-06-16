 // define a camera
        let cameraStream 
        const camera =document.querySelector('#camera')
        const open =document.querySelector('#open')
        const video =document.querySelector('#video')
        const Stop =document.querySelector('#stop')
        const selected =document.querySelector('#select')
        const to_oldimg=document.querySelector('#oldImg')
        const constraints = { audio: true, video: { width:400, height:400 }}
        // 擷取照片傳至canvas中
        const canvas = document.querySelector('#canvas');
        const context = canvas.getContext('2d');
        const Screenshot = document.querySelector('#screenshot')
        const oldImg=document.querySelector('#oldImg')
        const cameraWindow=document.querySelector('#camera-window')
        const takePhoto=document.querySelector('#takephoto')
        const cancel=document.querySelector('#cancel')
        const screenshotContainer = document.getElementById('screenshot-container');

        takePhoto.addEventListener('click',()=>{
        // show in window
        cameraWindow.style.display='block';
        camera.addEventListener('click',()=>{
                console.log(22,video)
                navigator.mediaDevices.getUserMedia(constraints).then
                (stream => {
                    cameraStream = stream
                    video.srcObject = stream
                    video.play();
                    // const videoContainer = document.getElementById('video-container');
                    cameraWindow.style.position = 'absolute';
                    cameraWindow.style.top='50%';
                    cameraWindow.style.left='50%';
                    video.style.position = 'absolute';
                    video.style.left = '50%';
                    video.style.top = '50%';
                    video.style.transform = 'translate(-50%, -50%)';
                    console.log(36,video)
                }).catch(err=>{
                alert('open camera fail')
            });
         });
        
        // select the photo
        selected.addEventListener('click',()=>{
            const new_oldimg=document.createElement('div')
            new_oldimg.width=width
            new_oldimg.height=height
        })

        // stop screen 
        Stop.addEventListener('click',()=>{
                if (cameraStream){
                //  getTracks取得所有軌道(track)，返回MediaStreamTrack物件陣列，每個物件包含該軌道的相關資訊(（頻或視頻、ID、狀態)
                    cameraStream.getTracks().forEach(track=>{
                    track.stop()
                    })
                    cameraStream=null
                }
                })
        // show screenshot
        Screenshot.addEventListener('click', () => {
                // 獲取canvas的寬度和高度
                const width=canvas.width= video.videoWidth
                const height=canvas.height= video.videoHeight;
                context.drawImage(video,0,0,width,height);
                screenshotContainer.innerHTML = '';
                screenshotContainer.appendChild(canvas);
                screenshotContainer.style.position='absolute';
                screenshotContainer.style.left = '50%';
                screenshotContainer.style.top = '50%';
                screenshotContainer.style.transform = 'translate(-50%, -50%)';
                // 將canvas的內容轉為base64格式
                // const imagedata=canvas.toDataURL('image/jpg');
        });

        cancel.addEventListener('click',()=>{
            video.src=""
        })
    });
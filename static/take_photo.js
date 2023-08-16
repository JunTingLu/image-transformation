 // define a camera
        let cameraStream 
        const camera =document.querySelector('#camera')
        const open =document.querySelector('#open')
        const video =document.querySelector('#video')
        // const Stop =document.querySelector('#stop')
        const selected =document.querySelector('#select')
        const to_oldimg=document.querySelector('#oldImg')
        // video.style.cssText="width:400px;height:400px;";
        const constraints = { audio: true, video:{width:512,height:512}}
        //Transfer the crop image to  canvas
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
                }).catch(err=>{
                alert('open camera fail')
            });
         });
        
        // select the photo
        selected.addEventListener('click',()=>{
            var dataURL = canvas.toDataURL('image');
            to_oldimg.src=dataURL
            console.log(53,to_oldimg)

            if (cameraStream){
                console.log(50)
                //  getTracks : Return an array of MediaStreamTrack objects, with each object containing relevant information about the track (audio or video), including its ID and status.
                    cameraStream.getTracks().forEach(track=>{
                    track.stop()
                    })
                    cameraStream=null
                }
        })

        // show screenshot
        Screenshot.addEventListener('click', () => {
                // Get the width/height of canvas
                const width=canvas.width= video.videoWidth
                const height=canvas.height= video.videoHeight;
                context.drawImage(video,0,0,width,height);
                screenshotContainer.innerHTML = '';
                screenshotContainer.appendChild(canvas);
                screenshotContainer.style.position='absolute';
                screenshotContainer.style.left = '50%';
                screenshotContainer.style.top = '50%';
                screenshotContainer.style.transform = 'translate(-50%, -50%)';
        });

        cancel.addEventListener('click',()=>{
            video.src=""
            cameraStream.getTracks().forEach(track=>{
                track.stop()
                })
                cameraStream=null
        })
    });

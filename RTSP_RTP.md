# RTSP & RTP with GSTREAMER

## Prerequisites

### Windows
* Install gstreamer from https://gstreamer.freedesktop.org/data/pkg/windows/  
* Add `C:\gstreamer\1.0\x86_64\bin` to Path variable in Environment Variables  
* Start python `import cv2` then  `print(cv2.getBuildInformation())` make sure gstreamer is enabled. If not you will need custom built opencv follow [medium.com build instructions](https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c) or [my Windows build instructions](https://github.com/uutzinger/Windows_Install_Scripts/blob/master/BuildingOpenCV.md).


The Windows camera app can be used to view the resolution and frame rate options of your camera. If you have OBS Studio installed, you can open/add a capture device and adjust the video properties.

### Unix
* Obtain gstreamer `sudo apt-get install gstreamer1.0*`
* Obtain plugins `sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio`
* Obtain RTSP tools `sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp`
* Obtain video 4 linux utilities `sudo apt-get install v4l-utils`

## **Gstreamer/Camera Options**

The easiest way to list camera options: 
```
gst-device-monitor-1.0
```

List a gstreamer plugins capabilities
```
gst-inspect-1.0
gst-inspect-1.0 plugin_of_my_interest
```

## **Video for Linux Subsystem**
On Linux you can list all cameras attached to the system with:
```
v4l2-ctl --list-devices
```
You can list the video formats of the cameras with:
```
v4l2-ctl --list-formats
```
You can modify and adjust your camera settings for example for the Sonty IX219 with:
```
v4l2-ctl --set-ctrl exposure= 13..683709
v4l2-ctl --set-ctrl gain= 16..170
v4l2-ctl --set-ctrl frame_rate= 2000000..120000000
v4l2-ctl --set-ctrl low_latency_mode=True
v4l2-ctl --set-ctrl bypass_mode=Ture
```
The v4l2-ctl command can be executed within python using:
```
os.system("v4l2-ctl -c exposure_absolute={} -d {}".format(my_exposure_time,my_camera_number))
```

## **RTSP**
To create an RTSP network stream one needs an rtsp server. It handles connections from different network clients. A single point to point conenction can be established with RTP not needing an RTSP server.
* On Windows: https://github.com/aler9/rtsp-simple-server
* On Unix: gstreamer RTSP server

Gstreamer rtsp server can be started through python un Unix. On Windows its more involved to start RTSP with python, but you can use RTP.

### **RTSP Clients**

### General RTSP Network Camera Client
* `gst-launch-1.0 rtspsrc location=rtsp://localhost:1181/camera ! fakesink`
* `gst-launch-1.0 rtspsrc location=rtsp://user:pass@192.168.81.32:554/live/ch00_0 ! rtph264depay ! h264parse ! decodebin ! autovideosink`

#### Windows h246 RTSP Camera Client
* `gst-launch-1.0 playbin uri=rtsp://localhost:8554/camera`
* `gst-launch-1.0 rtspsrc location=rtsp://192.168.11.26:1181/camera latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink`

#### Raspian h246 RTSP Camera Client
* `gst-launch-1.0 rtspsrc location=rtsp://localhost:1181/camera latency=10 ! rtph264depay ! h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! autovideosink sync=false`

#### Jetson Nano h246 RTSP Camera Client
* `gst-launch-1.0 rtspsrc location=rtsp://192.168.8.50:8554/unicast latency=10 ! rtph264depay ! h264parse ! omxh264dec ! nvoverlaysink overlay-x=800 overlay-y=50 overlay-w=640 overlay-h=480 overlay=2`
* `gst-launch-1.0 rtspsrc location=rtsp://192.168.8.50:8554/unicast latency=10 ! rtph264depay ! h264parse ! omxh264dec ! autovideosink`

### RTSP Server

Jetson  
https://developer.ridgerun.com/wiki/index.php?title=Jetson_Nano/Gstreamer/Example_Pipelines/Streaming
```
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! nvv4l2h264enc bitrate=8000000 insert-sps-pps=true ! video/x-h264, mapping=/stream1 ! rtspsink service=5000
```

https://stackoverflow.com/questions/59858898/how-to-convert-a-video-on-disk-to-a-rtsp-stream
```
import sys
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

loop = GLib.MainLoop()
Gst.init(None)

class TestRtspMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)

    def do_create_element(self, url):
        #set mp4 file path to filesrc's location property
        src_demux = "filesrc location=/path/to/dir/test.mp4 ! qtdemux name=demux"
        h264_transcode = "demux.video_0"
        #uncomment following line if video transcoding is necessary
        #h264_transcode = "demux.video_0 ! decodebin ! queue ! x264enc"
        pipeline = "{0} {1} ! queue ! rtph264pay name=pay0 config-interval=1 pt=96".format(src_demux, h264_transcode)
        print ("Element created: " + pipeline)
        return Gst.parse_launch(pipeline)

class GstreamerRtspServer():
    def __init__(self):
        self.rtspServer = GstRtspServer.RTSPServer()
        factory = TestRtspMediaFactory()
        factory.set_shared(True)
        mountPoints = self.rtspServer.get_mount_points()
        mountPoints.add_factory("/stream1", factory)
        self.rtspServer.attach(None)

if __name__ == '__main__':
    s = GstreamerRtspServer()
    loop.run()
```
watch with
```
gst-launch-1.0 playbin uri=rtsp://127.0.0.1:8554/stream1
```

https://stackoverflow.com/questions/64707903/how-to-build-gst-rtsp-server-on-windows



## **RTP**
RTP creates point to point network stream. It is accessible in python both for Windows and Unix.

### **Windows h264** 

#### **Receive RTP Stream**
```
gst-launch-1.0 -v udpsrc port=554 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false
```
#### **Video Test Stream to RTP**
*create h264 rtp stream:*
```
gst-launch-1.0 -v videotestsrc ! video/x-raw,width=(int)1280, height=(int)720, framerate=20/1 ! videoscale ! videoconvert ! x264enc tune=zerolatency bitrate=2048 speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=554
``` 
- `videotestsrc ! video/x-raw,width=(int)320,height=(int)240,framerate=20/1`: creates test video at desired resolution and frame rate
- `videoscale`: uses minimum resources if no scaling is needed  
- `videoconvert`: enhances compatibility  
- `x264enc`: creates MPEG-4 AVC, bitrate is in kbit/sec
- `rtph264pay`: creates the rtp payload
- `udpsink`: creates the network stream to the host on UDP

You need to specify the target IP. In order to allow any client to connect you need an RTSP server.

#### **Camera Stream**
You can access camera and stream with:  

Uncompressed
```
gst-launch-1.0 mfvideosrc device-index=0 ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoscale ! videoconvert ! autovideosink
```

MJPG compressed
```
gst-launch-1.0 -v mfvideosrc device-index=0 ! image/jpeg, width=1280, height=720 ! jpegdec ! videoconvert ! autovideosink
```

Camera to RTP
```
gst-launch-1.0 mfvideosrc device-index=0 ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoscale ! videoconvert ! x264enc tune=zerolatency bitrate=5000 speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5000
``` 

#### **Playing Files**
```
gst-launch-1.0 filesrc location=/my_dir/my_audio_file ! decodebin ! audioconvert ! audioresample ! autoaudiosink
```
```
gst-launch-1.0 filesrc location=/my_dir/my_video_file ! decodebin ! videoconvert ! autovideosink
```

```
gst-launch-1.0 playbin uri=file:///path/to/file_video.mp4 
```

#### **Playing Video File to RTP**

```
gst-launch-1.0 -v rtpbin name=rtpbin filesrc location=test.mp4 ! qtdemux name=demux 
demux.video_0 ! decodebin ! x264enc ! rtph264pay config-interval=1 pt=96 ! rtpbin.send_rtp_sink_0 rtpbin.send_rtp_src_0 ! udpsink host=127.0.0.1 port=554 sync=true async=false
```
Watching
```
vlc -v rtp://127.0.0.1:5000
```

### **Unix h264**

#### **Receive RTP Stream**

```
gst-launch-1.0 udpsrc port=554 ! application/x-rtp,encoding-name=(string)H264,payload=(int)96,clock-rate=(int)90000,media=(string)video ! rtph264depay ! decodebin ! videoconvert ! autovideosink
```

debug
```
gst-launch-1.0 udpsrc port=554 ! application/x-rtp, payload=(int)96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! fpsdisplaysink sync=false text-overlay=true
```
or  
```
gst-launch-1.0 udpsrc port=554 ! application/x-rtp, payload=(int)96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```
or
```
gst-launch-1.0 udpsrc port=554 ! application/x-rtp, encoding-name=(string)H264,payload=(int)96 ! rtph264depay ! h264parse ! avdec_h264 ! xvimagesink sync=true async=false -e
```

#### **Camera Stream**

```
gst-launch-1.0 v4l2src ! xvimagesink
```

Uncompressed
```
gst-launch-1.0 v4l2src do-timestamp=true device=/dev/video0 ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoscale ! videoconvert ! autovideosink
```

Jetson
```
gst-launch-1.0 v4l2src ! videoconvert ! 'video/x-raw,format=I420,width=1920,height=1080,framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=NV12, framerate=30/1' ! nvoverlaysink sync=false async=false
```
or
```
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! nvoverlaysink sync=false async=false
```

To RTP
```
gst-launch-1.0 -e v4l2src do-timestamp=true device=/dev/video0 ! video/x-h264,width=640,height=480,framerate=30/1 ! h264parse ! rtph264pay config-interval=1 ! gdppay ! udpsink host=127.0.0.1 port=1234
```

Jetson to RTP
```
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! "video/x-raw,format=I420, width=1920, height=1080, framerate=60/1" ! omxh264enc control-rate=2 insert-sps-pps=true bitrate=16000000 ! video/x-h264, stream-format=byte-stream ! rtph264pay ! udpsink host=192.168.2.100 port=1234 sync=false async=false
```
or
```
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! nvv4l2h264enc insert-sps-pps=true insert-vui=1 bitrate=16000000 !  rtph264pay mtu=1400 name=pay0 ! udpsink host=192.168.2.100 port=1234 sync=false async=false
```

## **Python**  
To create an rtp stream with opencv:
```
rtp_send = cv2.VideoWriter(gstreamer_str,cv2.CAP_GSTREAMER,0, 20, (320,240), True)
```
To create an rtp receiving opbject:
```
rtp_receive = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
```
Simplest way to open a network source:
```
cv2.VideoCapture("rtp://192.168.1.12:554)
```

https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/
https://stackoverflow.com/questions/62481010/stream-h264-video-with-opencvgstreamer-artifacts

Sender:  (bitrate in kbit/sec)
```
gstreamer_str='appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=5000 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=554'
```

or https://github.com/prabhakar-sivanesan/OpenCV-rtsp-server    
```
gstreamer_str = 'appsrc is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 
! videoconvert ! video/x-raw,format=I420 
! x264enc speed-preset=ultrafast tune=zerolatency bitrate=5000
! rtph264pay config-interval=1 pt=96 ....'.format(image_width, image_height, fps)
```

Receiver:  
```
gstreamer_str='udpsrc port=554 ! application/x-rtp,encoding-name=(string)H264,payload=(int)96,clock-rate=(int)90000,media=(string)video ! rtph264depay ! decodebin ! videoconvert ! appsink'
```

or (converts to opencv BGR format)
```
gstreamer_str='udpsrc port=554 ! application/x-rtp,encoding-name=(string)H264,payload=(int)96,clock-rate=(int)90000,media=(string)video ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert ! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
```



#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import os
import numpy as np
import copy
import rospy
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import *
from vision_pkg_full_demo.srv import *

print("Initializing multiple cameras")
rospy.init_node('multiple_OAK', anonymous=True)
image_pub1 = rospy.Publisher("/OAK/stream_compressed", CompressedImage) #WH1
image_pub2 = rospy.Publisher("/OAK/stream_compressed_1", CompressedImage) #Global
image_pub3 = rospy.Publisher("/OAK/stream_compressed_2", CompressedImage) #WH2
topic_start_recording = '/OAK/start_video_recording'
topic_stop_recording = '/OAK/stop_video_recording'
record_time_pub = rospy.Publisher("/OAK/record_time", String)
service_name = '/OAK/capture_img'

def createPipeline(camera_id=0):
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    # camRgb.setPreviewSize(300, 300)
    # camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    # camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # camRgb.setInterleaved(False)

    # # Create output
    # xoutRgb = pipeline.create(dai.node.XLinkOut)
    # xoutRgb.setStreamName("rgb")
    # camRgb.preview.link(xoutRgb.input)

    ###

    #Properties
    if camera_id==1:
        camRgb.initialControl.setManualFocus(155) #155
        camRgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
        camRgb.setFps(10)
    elif camera_id==2:
        camRgb.initialControl.setManualFocus(155) #155
        camRgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
        camRgb.setFps(10)
    else:
        camRgb.setFps(60)    
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)

    # Output
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutVideo.setStreamName("video")
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Linking
    camRgb.video.link(xoutVideo.input)

    return pipeline


with contextlib.ExitStack() as stack:
    deviceInfos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

    videoMap = {}
    devices = []

    for deviceInfo in deviceInfos:
        deviceInfo: dai.DeviceInfo
        device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
        devices.append(device)
        print("===Connected to ", deviceInfo.getMxId())
        mxId = device.getMxId()
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()
        print("   >>> MXID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        print("   >>> USB speed:", usbSpeed)
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName)

        if (str(mxId) == "184430107197470E00"): #Global
            camID=0
        elif (str(mxId) == "14442C1061EC47D700"): #Cables
            camID=1
        elif (str(mxId) == "18443010418A4C0E00"): #WH2
            camID=2
        else:
            print(str(mxId))
            print("ERROR")
            exit()
        pipeline = createPipeline(camID)
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        #q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "cam-" + str(camID)
        #videoMap.append((video, stream_name))
        videoMap[stream_name] = video


    def service_callback(req): 
            """
            Service that captures an image
            """
            flip = True
            print("Saving img...")
            #time.sleep(1)
            resp = TriggerResponse()
            #try:
            
            dir_path = os.path.join(os.path.dirname(__file__), '../imgs/')
            dir1 = os.listdir(dir_path)
            file_index = [0]
            for file in dir1:
                    file_index.append(int((file.split(".")[0]).split("_")[-1]))
            new_index = str(max(file_index)+1)

            img_name = "Image_cables_wh"+str(req.cam_id)+"_"+new_index+".jpg"
            img_path = os.path.join(os.path.dirname(__file__), '../imgs/'+img_name)
            video_0 = videoMap["cam-"+str(req.cam_id)]
            videoIn = video_0.get()
            if flip:
                    image = cv2.flip(videoIn.getCvFrame(), -1)
            else:
                    image = videoIn.getCvFrame()
            cv2.imwrite(img_path, image)
            resp.success = True
            resp.message = img_path
            print("Done")
            #except:
            #        resp.success = False
            #        print("Error")
            return resp        

    rospy.Service(service_name, cameraCapture, service_callback)
    print("OAK capture image service available")

        
    # def video_callback(req):
    #         fps=10
    #         duration=15
    #         i=0
    #         writer= cv2.VideoWriter('/home/remodel/remodel_demos_ws/src/vision_pkg_full_demo/videos/basicvideo.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (1920,1080))
    #         while i<(duration*fps):
    #                 print(i)
    #                 video_1 = videoMap["cam-0"]
    #                 videoIn = video_1.get()
    #                 frame = videoIn.getCvFrame()
    #                 writer.write(frame)
    #                 i+=1

    #         writer.release()
    
    # rospy.Service('/OAK/record_video', Trigger, video_callback)
    # print("OAK record video service available")

    def start_callback(data):
        global stop_video
        dir_path = os.path.join(os.path.dirname(__file__), '../videos/')
        dir1 = os.listdir(dir_path)
        file_index = [0]
        for file in dir1:
                file_index.append(int((file.split(".")[0]).split("_")[-1]))
        new_index = str(max(file_index)+1)

        video_name = "Isometric_video_"+new_index+".mp4"
        video_path = os.path.join(os.path.dirname(__file__), '../videos/'+video_name)
        print("Video started")
        stop_video=False
        fps=30
        i=0.0
        writer= cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (1920,1080))
        msg_time = String()
        video_1 = videoMap["cam-0"]
        while not stop_video:
                videoIn = video_1.get()
                frame = videoIn.getCvFrame()
                writer.write(frame)
                i+=(1/fps)
                time = int(copy.deepcopy(i))
                minutes = int(time/60)
                min_str = str(minutes)
                if minutes <= 9:
                    min_str = "0"+min_str
                seconds = time%60
                sec_str = str(seconds)
                if seconds <= 9:
                    sec_str = "0"+sec_str
                msg_time.data = str(min_str) + ":" + str(sec_str)
                #print(msg_time.data)
                record_time_pub.publish(msg_time)

        msg_time.data = "00:00"
        record_time_pub.publish(msg_time)
        writer.release()

    subsStart = rospy.Subscriber(topic_start_recording, Bool, start_callback)


    def stop_callback(data):
        global stop_video
        stop_video=True        
        print("Video stopped")

    subsStop = rospy.Subscriber(topic_stop_recording, Bool, stop_callback)
    

    while not rospy.is_shutdown():
        for stream_name_i in videoMap:
            video_i = videoMap[stream_name_i]
            if video_i.has():                
                try:
                    videoIn = video_i.get()
                    # Get BGR frame from NV12 encoded video frame to show with opencv
                    # Visualizing the frame on slower hosts might have overhead                    
                    #cv2.imshow(stream_name_i, videoIn.getCvFrame())
                    #### Create CompressedImage ####
                    msg = CompressedImage()
                    msg.header.stamp = rospy.Time.now()
                    msg.format = "jpeg"
                    if stream_name_i == "cam-1":
                        msg.data = np.array(cv2.imencode('.jpg', cv2.flip(videoIn.getCvFrame(), -1))[1]).tostring()
                        image_pub1.publish(msg) #WH1
                    elif stream_name_i == "cam-2":
                        msg.data = np.array(cv2.imencode('.jpg', cv2.flip(videoIn.getCvFrame(), -1))[1]).tostring()
                        image_pub3.publish(msg) #WH2
                    else:
                        msg.data = np.array(cv2.imencode('.jpg', videoIn.getCvFrame())[1]).tostring()                       
                        image_pub2.publish(msg)

                    if cv2.waitKey(1) == ord('q'):
                            break
                except:
                        pass

        if cv2.waitKey(1) == ord('q'):
            break
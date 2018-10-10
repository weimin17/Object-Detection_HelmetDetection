import os
#import requests
#import json
#import base64
import threading
import time
import cv2
import queue
import subprocess as sp
import sys

rtmpurl = 'rtmp://192.168.132.20:1935/live/'

class RtmpStreaming(threading.Thread):

    def __init__(self, threadID, name, width, height, fps):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = rtmpurl + 'name'      
        self.width = width
        self.height = height
        self.fps = fps
        self.interval = 1.0 / fps
        self.ffmpegCmd = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', str(width) + 'x' + str(height),
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv', 
            rtmpurl + str(name)]
        self.isRunning = False
        self.exitFlag  = False        
        self.queue = queue.Queue(50)  #Max alarm in queue
        self.queue_lock = threading.Lock()
        self.maxtry = 5
    def terminate(self):        
        self.exitFlag = True

    def GetRunning(self):
        return self.isRunning

    def run(self):  #循环发
        pipe = None
        errorcount = 0
        while self.exitFlag == False:
            x = time.time()
            self.isRunning = True
            if pipe == None:
                try:
                    print(time.strftime('%Y-%m-%d %H:%M:%S ') + '{} Try to connect {}'.format(self.threadID,str(self.ffmpegCmd)))
                    pipe = sp.Popen(self.ffmpegCmd, stdin=sp.PIPE)
                    time.sleep(2)
                except OSError as e:
                    #print('OSError No {}'.format(e.errno))
                    print('OSError > {}'.format(e.strerror))
                    #print('OSError > {}'.format(e.filename))
                    pipe = None
                except:
                    print('Error while open {}'.format(self.ffmpegCmd))
                    pipe = None
            #Get a Alarm and post to remote server.
            ret, frame = self.getFrame()
            if ret:
                if pipe != None:
                    try:
                        pipe.stdin.write(frame.tostring())
                    except OSError as e:
                        print('OSError > {}'.format(e.strerror))
                        errorcount += 1
                        if errorcount > self.maxtry:
                            pipe = None
                            errorcount = 0
                    except:
                        print('Error while write frame')
                        pipe = None
            y = time.time()
            seconds =  y - x
            if self.interval > seconds:
                time.sleep(self.interval - seconds)

        self.isRunning = False
        #if pipe is not None:
        #    pipe.close()

    def getFrame(self):
        ret = False
        frame = None
        with self.queue_lock:
            try:
                frame = self.queue.get(False)
                ret = True
            except queue.Empty:
                #print(time.strftime('%Y-%m-%d %H:%M:%S ') + '{} Frame Queue is Empty!'.format(self.threadID))
                pass
        return ret, frame

    def addFrame(self, frame):

        #print(time.strftime('%Y-%m-%d %H:%M:%S ') + '{} addFrame: {}.'.format(self.threadID, self.name))
        ret = False
        with self.queue_lock:
            try:
                self.queue.put(frame, False)
                ret = True
            except queue.Full:
                #print(time.strftime('%Y-%m-%d %H:%M:%S ') + '{} Frame Queue is full!'.format(self.threadID))
                pass
        return ret

def main():
    interval = 1/25
    rtmp = RtmpStreaming(3,3,640,480,25)
    rtmp.start()
    time.sleep(5)
    print("RTMP Status: {}",rtmp.GetRunning())
    camera = cv2.VideoCapture('rtsp://admin:k2vision@10.1.20.230:554/h264/ch35/main/av_stream')
    while True:
        x = time.time()
        #imagefile = cv2.imread('img0001.jpg',cv2.IMREAD_COLOR)
        grabbed, frame_lwpCV = camera.read()
        if grabbed:
            imagefile = cv2.resize(frame_lwpCV,(640,480),interpolation=cv2.INTER_CUBIC)
            #imagefile = open(r'img0001.jpg', 'rb')
            resp = rtmp.addFrame(imagefile)
        y = time.time()
        seconds = y - x
        x = y
        if seconds < interval:
            time.sleep(interval -seconds)
        #time.sleep(1/25)

if __name__ == '__main__':
    main()
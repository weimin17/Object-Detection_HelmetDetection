import os
import requests
import json
import base64
import threading
import time
#import cv2
import queue

url = 'http://192.169.3.32:8082/vision/alarm'

def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton

@singleton
class Alarm():
    headers = {'content-type': 'application/json'}

    def __init__(self):
        self.url = url      
        self.isRunning = False
        self.exitFlag  = False
        self.queue = queue.Queue(50)  #Max alarm in queue
        self.queue_lock = threading.Lock()
        self.thread = threading.Thread(target = self.processAlarm, name = 'AlarmPost')
        self.thread.setDaemon(True)
        self.thread.start()
        print('Alarm start new Instance')
    def processAlarm(self):  #循环将告警发出去。
        print('processAlarm().' + threading.current_thread().getName())
        while self.exitFlag == False:
            self.isRunning = True
            #Get a Alarm and post to remote server.
            ret, alarm = self.getAlarm()
            if ret:
                if not self._processAlarm(alarm):
                    print(time.strftime('%Y-%m-%d %H:%M:%S ') + 'post alarm return False, put alarm back to queue.')
                    self._addAlarm(alarm)
            else:
                time.sleep(1)

        self.isRunning = False

    def _processAlarm(self, alarmParams):    
        try:
            r = requests.post(self.url, data = json.dumps(alarmParams), headers = self.headers)
            print(time.strftime('%Y-%m-%d %H:%M:%S ') + 'call request.post return {}.'.format( r))
            if r is not None or r.text is not None:
                return r.text
            return None
        except:
            return False


    def getAlarm(self):
        ret = False
        alarm = None
        with self.queue_lock:
            try:
                alarm = self.queue.get(False)
                ret = True
            except queue.Empty:
                #print(time.strftime('%Y-%m-%d %H:%M:%S ') + '{} Alarm Queue is Empty!'.format(self.threadID))
                pass
        return ret, alarm

    def addAlarm(self, cameraNubmer, triggerEvent, startTime, endTime, imagefiledata):

        alarmParams = {
            'cameraNumber': cameraNubmer,
            'triggerEvent': triggerEvent,
            'startTime': startTime,
            'endTime': endTime,
			'imageBytes': 'data:image/jpg;base64,' + base64.b64encode(imagefiledata).decode('utf-8')
        }

        print(time.strftime('%Y-%m-%d %H:%M:%S ') + 'addAlarm: {}, {}, {}'.format(cameraNubmer, triggerEvent, startTime))
        return self._addAlarm(alarmParams)

    def _addAlarm(self, alarmParam):
        ret = False
        with self.queue_lock:
            try:
                self.queue.put(alarmParam, False)
                ret = True
            except queue.Full:
                print(time.strftime('%Y-%m-%d %H:%M:%S ') + ' Alarm Queue is full!')

        return ret


#def main():
#    #alarm = Alarm()
#    print('Main().' + threading.current_thread().getName())
#    while True:
#        start_time = time.localtime(time.time())
#        end_time = time.localtime(time.time())
#        #imagefile = cv2.imread('img0001.jpg')
#        imagefile = open(r'img0001.jpg', 'rb')
#        resp = Alarm().addAlarm('5',  'helmet_onhead', time.strftime('%Y-%m-%d %H:%M:%S',start_time), time.strftime('%Y-%m-%d %H:%M:%S',end_time), imagefile.read())
#        print(resp)
#        imagefile.close()
#        time.sleep(0.5)
#
#if __name__ == '__main__':
#    main()
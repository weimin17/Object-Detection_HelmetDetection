import sys
import cv2
import time
import argparse
from helmet_detect import HelmetDetectThread
import rtmp_streaming

ckpt_path = "D:/Python/obj_detect_service/object_detection/savedModel/frozen_inference_graph20180702.pb"
videosource = "rtsp://admin:k2vision@10.1.20.230:554/h264/ch34/main/av_stream"
RTMP_WIDTH = 640
RTMP_HEIGHT = 360

def main(argv):
    if argv == None:
        print('Do not have any command line param, run as default param....')
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", help="name for this case")
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-r", "--rstp", help="path to the video stream rstp url")
    ap.add_argument("-m", "--min_area", type=int, default=100, help="minimum area size")
    ap.add_argument("-k", "--ckpt", help="model ckpt path")
    ap.add_argument("-f", "--fps", type=int, default=25, help="video fps")
    ap.add_argument("-a", "--alarm_level",  type=int, default=0, help="alarm level")
    ap.add_argument("-t", "--timer",  type=int, default=1, help="detect period")
    ap.add_argument("-d", "--delay",  type=int, default=0, help="seconds to delay when find event")
    ap.add_argument("-p", "--period", type=int, default=10, help="seconds to realarm when find same event")
    ap.add_argument("-g", "--gpu_mem", type=int, default=40, help="percent of GPU memory")

    args = vars(ap.parse_args())

    # 如果video参数为None，那么我们从摄像头读取数据
    isvideofile = False
    if args.get("video", None) is None:
        videosource = 0
    # 否则我们读取一个视频文件
    else:
        videosource = args["video"]
        isvideofile = True

    if args.get("rstp", None) is not None:
        videosource = args["rstp"]

    if args.get("name", None) is None:
        name = "Camera"
    else:
        name = args["name"]

    print('Name:' + name + ',Video Source ' + str(videosource))

    ckpt = ckpt_path
    if args.get("ckpt", None) is not None:
        ckpt = args["ckpt"]

    fps = 25
    if args.get("fps", None) is not None:
        fps = args["fps"]
    interval = 1/fps

    alarm_level = 0
    if args.get("alarm_level", None) is not None:
        alarm_level = args["alarm_level"]

    timer = 0
    if args.get("timer", None) is not None:
        timer = args["timer"]

    alarm_delay = 0
    if args.get("delay", None) is not None:
        alarm_delay = args["delay"]

    alarm_period = 10
    if args.get("period", None) is not None:
        alarm_period = args["period"]

    gpu_mem = 40
    if args.get("gpu_mem", None) is not None:
        gpu_mem = args["gpu_mem"]

    error_count = 0
    max_try = 5
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',正在打开视频源...')
    camera = cv2.VideoCapture(videosource)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',正在创建RTMP视频流发送线程...')
    rtmp = rtmp_streaming.RtmpStreaming(name, name, RTMP_WIDTH, RTMP_HEIGHT,fps) 
    rtmp.start()
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',正在创建检测线程...')
    detector = HelmetDetectThread(1, name, ckpt, fps, timer, alarm_delay, alarm_level, alarm_period, gpu_mem)
    detector.start()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)

    while True:
        start = time.time()
        # 判断视频是否打开
        if camera is None:
            print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',正在打开视频源...')
            camera = cv2.VideoCapture(videosource)
            time.sleep(1)
            if camera.isOpened():
                print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',打开视频源成功！')
                size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                print('size:'+repr(size))            
            else:
                print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',打开视频源失败！')
                camera = None
                time.sleep(2)
                continue

        #视频源打开成功，开始读取视频
        grabbed, frame_lwpCV = camera.read() # 读取视频流
        if not grabbed: #读取视频出错，进行处理
            error_count += 1
            if error_count > max_try: #超过最大值，关闭视频
                camera.release()
                camera = None
                error_count = 0
                print(time.strftime('%Y-%m-%d %H:%M:%S ') + name + ',读取视频失败，关闭视频！')
                time.sleep(2)
            continue # 无论是否超过最大值
        else: #读一帧成功
            error_count = 0  # 清楚异常标志

        #RTMP connected, start to push streaming.
        detector.putFrame(frame_lwpCV, size)
        rtmp.addFrame(cv2.resize(frame_lwpCV, (RTMP_WIDTH, RTMP_HEIGHT),interpolation=cv2.INTER_CUBIC))

        end = time.time()
        seconds = end - start
        if isvideofile: # video  控制播放速度
            if seconds < interval:
                time.sleep(interval - seconds)

        detected = detector.getDetectecImg()
        if detected is not None:
            cv2.imshow(name, detected)

        # 按'q'健退出循环
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When everything done, release the capture
    rtmp.terminate()
    detector.terminate()
    if camera is not None:
        camera.release()

if __name__ == '__main__':
    main(sys.argv)

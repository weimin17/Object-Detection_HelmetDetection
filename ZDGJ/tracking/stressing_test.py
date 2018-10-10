#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:49:58 2018

@author: drtian
"""
from multiprocessing import Pool
from mainfuction import test_main 
import pandas as pd
import glob
import time
import os

if __name__=='__main__':
    
    DETECT_FREQUECY_set = [25,40,50,75]
    GPU_MEMORY_set = [0.05]
    PATH_TO_CKPT = '/data/saved_models/fast_rcnn_inception_v2/0802_data_fasterRCNN_inception_k2office_finetune/frozen_inference_graph.pb'
    concurrent = 20
    df_row_ind = 0 
    df_empty = pd.DataFrame(columns=['t','DETECT_FREQUECY','GPU_MEMORY','processNums', 'videoName', 'time', 'cpu','memory'])
    for t in range(20):
        print(t)
        for DETECT_FREQUECY in DETECT_FREQUECY_set:
    
                # videos = glob.glob('/home/drtian/ZDGJ/test_video/*.mp4')
            videos = glob.glob('/data/ZDGJ/监控录像/1#机组脱硫CESM小室1529636867_2CE8C22B/fix_16869FE8_1529636867_1.mp4')
            
            for videoName in videos:
                CAMERA_IP = videoName
            #    videoName = '/home/drtian/ZDGJ/test_video/noPeople_20s_1920_1080.mp4'
            #    df_row = [5,"videoName",3.1,"50%","50%"]
            #    df_empty.loc[df_row_ind] = df_row
            for k in range(concurrent):
                    for GPU_MEMORY in GPU_MEMORY_set:
                        k += 1
                        df_row = []
                        df_row.append(t)
                        df_row.append(DETECT_FREQUECY)
                        df_row.append(GPU_MEMORY)
                        df_row.append(k)
                        df_row.append(videoName)
                        startTime = time.time()
                        print('Parent process %s.' % os.getpid())
                        p = Pool(k)
                        for i in range(k):
                            p.apply_async(test_main,args=(CAMERA_IP,DETECT_FREQUECY,GPU_MEMORY,PATH_TO_CKPT))
                        print('Waiting for all subprocesses done...')
                        p.close()
                        p.join()
                        print('All subprocesses done.')
                        endTime = time.time()
                        timedur= endTime - startTime
                        df_row.append(timedur)
                        df_row.append(0)
                        df_row.append(0)
                        df_empty.loc[df_row_ind] = df_row
      
                        with open('test_result_0810.csv', 'a') as f:
                            df_empty.to_csv(f,header=False)
    
#    test_main(CAMERA_IP,DETECT_FREQUECY,GPU_MEMORY,PATH_TO_CKPT)
    
    
    
    

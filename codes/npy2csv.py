'''
convert bndbox (.npy) to submitted format (.csv)
'''
# xmin ymin xmax ymax
import numpy as np
import csv, os, cv2

alldata = np.load('/data/contest/codes/bndbox_0.95_trainall_faster_RCNN_Inception_ResNet_v2_batch1_0809.npy')
# save to csv
csvFile = open('/data/contest/codes/bndbox_0.95_trainall_faster_RCNN_Inception_ResNet_v2_batch1_0809.csv','w')
PATH_TO_TEST_IMAGES_DIR='/data/contest/data/test/'
writer = csv.writer(csvFile)
for key, value in alldata.item().items():
    if len(value)==0:
        row = [key,""]
        print(row)
        writer.writerow(row)
    else:
        image_path = (os.path.join(PATH_TO_TEST_IMAGES_DIR,key))
        img = cv2.imread(image_path)
        size = img.shape
        for i in range(len(value)):
            row = [key,"%s %s %s %s"%(int(size[1]*value[i][0]),int(size[0]*value[i][1]),int(size[1]*value[i][2]),int(size[0]*value[i][3]))]
            print(row)
            writer.writerow(row)

csvFile.close()

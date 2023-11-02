import os
import cv2
import numpy as np
import glob

cwd = os.getcwd()
img_array = []
# for filename in glob.glob(cwd+'/frames/*.jpg'):
frameCt = len(glob.glob(cwd+'/frames/eps010/*.jpg'))
for idx in range (1,frameCt):
    filename = cwd+f"/frames/eps010/{idx}.jpg"
    img = cv2.imread(filename)
    # print(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

auxName = "/Users/doruk/data/woozy-ruby-ostrich-003c9f12ce13-20220712-162931.mp4"
auxVideo = cv2.VideoCapture(auxName)

fps = auxVideo.get(cv2.CAP_PROP_FPS)
print(fps)
# quit()

out = cv2.VideoWriter('eps010.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
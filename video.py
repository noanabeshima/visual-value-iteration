import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import pickle

# pixelwidth, pixelheight, pixels per square, frames per second

# PH, PW = 108, 192
# PPS = 10
# FPS = 40

PH, PW = 27, 43
PPS = 40
FPS = 30

width = PW*PPS
height = PH*PPS

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./maze2.avi', fourcc, float(FPS), (width, height))

pickle_in = open("frames.pickle", "rb")
frames = pickle.load(pickle_in)

overlay2 = frames.pop()
overlay1 = frames.pop()

for f in frames:
    im = cv2.normalize(f, None, alpha=.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im = np.kron(im, np.ones((PPS, PPS)))
    im = overlay2+overlay1*im
    im = np.dstack((im,im,im))
    im = np.uint8(im*255)
    video.write(im)

video.release()

import pandas as pd
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt
import scipy

# Observing depth data withing these dimensions
box_dim = (200, 150, 300, 250)

skip_init_frames = 200
last_frame = 1000
display_period = 5000
kernel_size = 9

def display_depth_frame(frame):
    """ In-the-loop display; used for debug only """
    min, max = np.min(frame), np.max(frame)
    print('min, max', min, max)
    frame = frame * 255 /(max - min)
    rgb = PIL.Image.fromarray(frame)
    draw = ImageDraw.Draw(rgb)
    draw.rectangle(box_dim, fill=None, outline="red")
    rgb.show()


def crop(img, box_dim):
    return img[box_dim[0]:box_dim[2], box_dim[1]:box_dim[3]]    

def scaled_np_frame(frame):
    min, max = np.min(frame), np.max(frame)
    print('min, max', min, max)
    frame = frame * 255 /(max - min)
    return crop(frame, box_dim)

xdata = []
mean_data = []

file_list = glob.glob('./out/*.csv')
print('File count:', len(file_list))

axes = plt.gca()
axes.set_xlim(0, len(file_list))
axes.set_ylim(0, 100)
max_line, = axes.plot(xdata, mean_data, 'r-')

mean_seq = []
for i, f_name in enumerate(file_list):
    if i < skip_init_frames:
        continue
    if i > last_frame:
        continue
    print(f_name) 
    df = pd.read_csv(f_name, header=None) 
    np_frame = df.to_numpy()

    print('original image size:', df.shape)
    if i % display_period == 0:
        display_depth_frame(np_frame)
    cropped_img = scaled_np_frame(np_frame)
    print('cropped image shape', cropped_img.shape)

    mean = np.mean(cropped_img)
    print('numpy mean', mean)
    mean_seq.append(mean)

    xdata.append(i)
    mean_data.append(mean)
    max_line.set_xdata(xdata)
    max_line.set_ydata(mean_data)

    plt.draw()
    plt.pause(0.05)

plt.show()

import scipy
from scipy import signal
mean_data = scipy.signal.medfilt(mean_data, kernel_size=kernel_size)

import matplotlib.pyplot as plt
plt.plot(mean_data)
plt.ylabel('some numbers')
plt.show()

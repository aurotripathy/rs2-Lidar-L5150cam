import pandas as pd
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt

# Observng depth data withing these dimensions
box_dim = (200, 200, 250, 250)

def scaled_np_frame(frame):
    min, max = np.min(frame), np.max(frame)
    print('min, max', min, max)
    frame = frame * 255 /(max - min)
    rgb = PIL.Image.fromarray(frame)
    # draw = ImageDraw.Draw(rgb)
    # draw.rectangle(box_dim, fill=None, outline="red")
    # rgb.show()
    return rgb.crop(box_dim)

def get_min_max_avg(depth_frame):
    min, max = depth_frame.getextrema()
    return min, max, (min + max)/2


xdata = []
max_data = []

file_list = glob.glob('./out/*.csv')
print('File count:', len(file_list))

axes = plt.gca()
axes.set_xlim(0, len(file_list))
axes.set_ylim(0, 100)
max_line, = axes.plot(xdata, max_data, 'r-')

min_max_avg = []
for i, f_name in enumerate(file_list): 
    print(f_name) 
    df = pd.read_csv(f_name, header=None) 
    np_frame = df.to_numpy()

    print('original image size:', df.shape)
    cropped_img = scaled_np_frame(np_frame)

    get_min_max_avg(cropped_img)

    print('cropped image size:', cropped_img.size)
    min, max, avg = get_min_max_avg(cropped_img)
    print('min, max, avg', min, max, avg)
    min_max_avg.append((min, max, avg))

    xdata.append(i)
    max_data.append(max)
    max_line.set_xdata(xdata)
    max_line.set_ydata(max_data)

    plt.draw()
    plt.pause(0.05)

plt.show()

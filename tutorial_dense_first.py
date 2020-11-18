import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

frame1 = cv.imread('basketball1.png')
frame2 = cv.imread('basketball2.png')
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# plotting the optical flow
def plot_quiver_2d(flow, title='optical flow'):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    num_rows = np.shape(flow)[0]
    num_cols = np.shape(flow)[1]
    x = np.arange(0, num_rows, 1)
    y = np.arange(0, num_cols, 1)
    y_pos, x_pos = np.meshgrid(y, x)
    fig, ax = plt.subplots()
    ax.quiver(y_pos, x_pos, v, u)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()

flow = np.array(flow)
plot_quiver_2d(flow)

# saving an HSV version of optical flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imwrite('hsv_opticalflow.png',bgr)

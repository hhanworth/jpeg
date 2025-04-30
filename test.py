import numpy as np
import cv2 

def RGB2YUV420(image):
    # 使得到的图像可以正好分成 8*8 的子块
    def adjust_size(image):
        h, w  = image.shape[0], image.shape[0]
        new_h = h // 8 * 8
        new_w = w // 8 * 8                          # 三次样条插值
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)    
        return image
    image = adjust_size(image)
    h, w, c = image.shape
    image_y = np.zeros((h, w), dtype=np.uint8)
    image_u = np.zeros((h // 2, w // 2), dtype=np.uint8)
    image_v = np.zeros((h // 2, w // 2), dtype=np.uint8)
    for line in range(h):
        for row in range(w):
            B = image[line, row, 0]
            G = image[line, row, 1]
            R = image[line, row, 2]
            Y = np.round(0.299 * R + 0.587 * G + 0.114 * B)
            image_y[line, row] = Y
            if line % 2 == 0 and row % 2 == 0:
                U = np.round(0.5 * R - 0.4187 * G - 0.0813 * B + 128)
                image_u[line // 2, row // 2] = U
            if line % 2 == 1 and row % 2 == 1:
                V = np.round(-0.1687 * R - 0.3313 * G + 0.5 * B + 128)
                image_v[line // 2, row // 2] = V
    image_y = adjust_size(image_y)
    image_u = adjust_size(image_u)
    image_v = adjust_size(image_v)
    return image_y, image_u, image_v


image = cv2.imread('./pic/p.bmp')
h, w, c = image.shape

img1,img2,img3 = RGB2YUV420(image)

print(img1.shape)
print(img2.shape)
print(img3.shape)
import numpy as np
from PIL import Image

img = Image.open("Phase0\\temple.jpg")

img_array = np.array(img)

# print(img_array.shape, img_array.dtype)
# print(img_array)

RGB_weights = np.array([0.299, 0.587, 0.114])

grey_shape = img_array[:, :, 0] #red_channel
# print(grey_shape)

img_grey = np.zeros(grey_shape.shape, dtype=np.uint8)
print(img_grey)

img_grey = np.round(img_array @ RGB_weights).astype(np.uint8)


rgb = np.stack((img_grey,img_grey, img_grey), axis=-1)

img_back_to_original = Image.fromarray(rgb).save("gray_rgb.png")
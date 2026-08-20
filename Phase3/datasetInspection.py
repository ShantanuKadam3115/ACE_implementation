import numpy as np
from PIL import Image

color_image = Image.open("Phase3\\dataset\\seq-01\\seq-01\\frame-000000.color.png")
depth_image = Image.open("Phase3\\dataset\\seq-01\\seq-01\\frame-000000.depth.png")


color_image_array = np.array(color_image)
print(color_image_array.shape, color_image.mode, color_image_array.dtype)
# image.show(image)

depth_image_array = np.array(depth_image)
print(depth_image_array.shape, depth_image.mode, depth_image_array.dtype)
# depth_image.show()
print(np.min(depth_image_array), np.max(depth_image_array))

with open("Phase3\\dataset\\seq-01\\seq-01\\frame-000000.pose.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)

content_array = np.loadtxt("Phase3\\dataset\\seq-01\\seq-01\\frame-000000.pose.txt")
print(content_array)


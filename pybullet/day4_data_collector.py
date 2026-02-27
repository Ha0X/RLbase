import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os

os.makedirs("images", exist_ok=True)

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

for i in range(10):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 0.5])
    for _ in range(100):
        p.stepSimulation()

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        320, 240, renderer=p.ER_TINY_RENDERER)[2:5]
    img = np.reshape(rgbImg, (240, 320, 4))
    cv2.imwrite(f"images/view_{i}.png", img)

p.disconnect()
print("Saved 10 images to 'images/' folder.")

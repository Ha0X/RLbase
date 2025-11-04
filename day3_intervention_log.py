import pybullet as p
import pybullet_data
import time
import csv
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

log_file = "data_log.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "box_start_x", "box_final_x"])

    for trial in range(5):
        box_start_x = random.uniform(0.3, 0.7)
        box_id = p.loadURDF("cube_small.urdf", [box_start_x, 0, 0.05])

        for _ in range(100):
            p.applyExternalForce(box_id, -1, [10, 0, 0], [0, 0, 0], p.WORLD_FRAME)
            p.stepSimulation()
            time.sleep(1/240)

        final_pos = p.getBasePositionAndOrientation(box_id)[0]
        writer.writerow([trial, box_start_x, final_pos[0]])
        p.removeBody(box_id)

p.disconnect()
print("Data saved to", log_file)

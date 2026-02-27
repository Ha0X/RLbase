import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

num_joints = p.getNumJoints(robot_id)
print(f"Number of joints: {num_joints}")

for joint in range(num_joints):
    info = p.getJointInfo(robot_id, joint)
    print(f"Joint {joint}: {info[1].decode('utf-8')}")

target_position = 1.0
for _ in range(1000):
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, targetPosition=target_position)
    p.stepSimulation()
    time.sleep(1/240)

pos, ori = p.getBasePositionAndOrientation(robot_id)
print("Base position:", pos)
print("Base orientation:", ori)

p.disconnect()

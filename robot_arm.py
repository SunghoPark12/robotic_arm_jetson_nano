import math
from pca9685 import PCA9685
import time

#펄스 범위
SERVOMIN_YAW = 400
SERVOMAX_YAW = 2000
SERVOMIN_SHOULDER = 400
SERVOMAX_SHOULDER = 2000
SERVOMIN_ELBOW = 300
SERVOMAX_ELBOW = 2100
SERVOMIN_WRIST = 1250
SERVOMAX_WRIST = 2150
SERVOMIN_WRIST_ROTATION = 346
SERVOMAX_WRIST_ROTATION = 2146
SERVOMIN_GRIPPER = 355
SERVOMAX_GRIPPER = 500

# 각도 범위
YAW_MIN_DEG, YAW_MAX_DEG = -80, 80
SHOULDER_MIN_DEG, SHOULDER_MAX_DEG = 10, 170
ELBOW_MIN_DEG, ELBOW_MAX_DEG = -80, 80
WRIST_MIN_DEG, WRIST_MAX_DEG = -90, 0    #90이 아래로 숙이는 방향향
WRIST_ROT_MIN_DEG, WRIST_ROT_MAX_DEG = -133, 47

LINK1 = 100.0
LINK2 = 80.0
LINK3 = 140.0
EXTENSION_LIMIT = 170

GRIPPER_GAP_MIN = 0
GRIPPER_GAP_MAX = 42
GRIPPER_GAP_OFFSET = 1
GRIPPER_BAR_LENGTH = 25  # 기구 설계에 맞게

X_ORIGIN_OFFSET = 0
Y_ORIGIN_OFFSET = 0
Z_ORIGIN_OFFSET = 106

HOME_POSE = (220, 0, 150, 0, 0)  # x, y, z, wrist, wrist_rotation

# PCA9685
pwm = PCA9685(address=0x40, busnum=1, freq=50)  # 주소 반드시 확인!

def map_number(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def deg_to_rads(deg):
    return deg * math.pi / 180

def rads_to_deg(rad):
    return rad * 180 / math.pi

def gripper(gap):
    if gap < GRIPPER_GAP_MIN or gap > GRIPPER_GAP_MAX:
        return
    opp = (gap / 2) + GRIPPER_GAP_OFFSET
    theta = rads_to_deg(math.asin(opp / GRIPPER_BAR_LENGTH))
    pulse = int(map_number(theta, 3, 90, SERVOMAX_GRIPPER, SERVOMIN_GRIPPER))
    print("pulse: %.2f" % pulse)
    pwm.set_pwm(5, 0, pulse)
    time.sleep(1)

def inverse_kinematics(x, y, z, wrist, rotation):
    x -= X_ORIGIN_OFFSET
    y -= Y_ORIGIN_OFFSET
    z -= Z_ORIGIN_OFFSET

    yaw = rads_to_deg(math.atan2(y, x))
    if not YAW_MIN_DEG <= yaw <= YAW_MAX_DEG:
        raise ValueError("yaw out of range: %.2f" % yaw)

    xyOffset = LINK3 * math.cos(deg_to_rads(wrist))
    xOffset = xyOffset * math.cos(deg_to_rads(yaw))
    yOffset = xyOffset * math.sin(deg_to_rads(yaw))
    zOffset = LINK3 * math.sin(deg_to_rads(wrist))

    x -= xOffset
    y -= yOffset
    z -= zOffset

    ext = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    if ext > EXTENSION_LIMIT or ext < LINK1 - LINK2:
        raise ValueError("ext out of range: %.2f" % ext)
    pitch1 = math.asin(z / ext) if x >= 0 else math.asin(z / -ext)
    elbowAngle = math.acos((LINK1**2 + LINK2**2 - ext**2) / (2 * LINK1 * LINK2))
    pitch2 = math.acos((LINK1**2 + ext**2 - LINK2**2) / (2 * LINK1 * ext))
    shoulder = rads_to_deg(pitch1 + pitch2)
    if not SHOULDER_MIN_DEG <= shoulder <= SHOULDER_MAX_DEG:
        raise ValueError("shoulder out of range: %.2f" % shoulder)
    elbowServoAngle = 180 - rads_to_deg(elbowAngle) - shoulder
    if (elbowServoAngle < ELBOW_MIN_DEG or elbowServoAngle > ELBOW_MAX_DEG or elbowServoAngle > 180 - shoulder - 40):
        raise ValueError("elbowServoAngle out of range: %.2f" % elbowServoAngle)
    print("yaw, shoulder, elbow, wrist, wristrot")
    wristServoAngle = wrist + elbowServoAngle
    if not WRIST_MIN_DEG <= wristServoAngle <= WRIST_MAX_DEG:
        raise ValueError("wristAngle out of range: %.2f" % wristServoAngle)
    if not WRIST_ROT_MIN_DEG <= rotation <= WRIST_ROT_MAX_DEG:
        raise ValueError("rotation out of range: %.2f" % rotation)

    yawPulse = int(round(map_number(yaw, YAW_MIN_DEG, YAW_MAX_DEG, SERVOMAX_YAW, SERVOMIN_YAW)))
    shoulderPulse = int(round(map_number(shoulder, SHOULDER_MIN_DEG, SHOULDER_MAX_DEG, SERVOMAX_SHOULDER, SERVOMIN_SHOULDER)))
    elbowPulse = int(round(map_number(elbowServoAngle, ELBOW_MIN_DEG, ELBOW_MAX_DEG, SERVOMAX_ELBOW, SERVOMIN_ELBOW)))
    wristPulse = int(round(map_number(wristServoAngle, WRIST_MIN_DEG, WRIST_MAX_DEG, SERVOMAX_WRIST, SERVOMIN_WRIST)))
    wristRotationPulse = int(round(map_number(rotation, WRIST_ROT_MIN_DEG, WRIST_ROT_MAX_DEG, SERVOMIN_WRIST_ROTATION, SERVOMAX_WRIST_ROTATION)))
    return [yawPulse, shoulderPulse, elbowPulse, wristPulse, wristRotationPulse]

def move_servos(pulses):
    for i, pulse in enumerate(pulses):
        print("%d" % i)
        print("%d" % pulse)
        response(i, pulse-30, pulse, 1, 0.5)
        #pwm.set_pwm(i, 0, pulse)
        time.sleep(1)

def response(channel, start, end, step, wait):
	print("channel:{channel}test start! ({start}~{end})")
	for pulse in range(start, end+1, step):
		print(f"puilse={pulse}")
		pwm.set_pwm(channel, 0, pulse)
		time.sleep(wait)
	print("\n end!")

def go_pose(x, y, z, wrist=0, rotation=0):
    pulses = inverse_kinematics(x, y, z, wrist, rotation)
    move_servos(pulses)

def go_test(x, y, z, wrist=0, rotation=0):
    pulses = inverse_kinematics(x, y, z, wrist, rotation)
    print(pulses)

if __name__ == "__main__":
    time.sleep(1)
    go_test(130,30,150,-40,0)
    go_pose(130,30,150,-40,0)

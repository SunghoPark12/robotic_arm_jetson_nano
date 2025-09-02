import math
import time
import smbus
from detector import Detector
from calibration_v2 import CameraCalibrator
import cv2


bus = smbus.SMBus(1)
address = 0x40

bus.write_byte_data(address, 0x00, 0x00)
time.sleep(0.01)
bus.write_byte_data(address, 0xFE, 0x79)

def set_pwm(channel, on, off):
	bus.write_byte_data(address, 0x06 + 4*channel, on & 0xFF)
	bus.write_byte_data(address, 0x07 + 4*channel, on >> 8)
	bus.write_byte_data(address, 0x08 + 4*channel, off& 0xFF)
	bus.write_byte_data(address, 0x09 + 4*channel, off >>8)

def angle_to_pwm(channel, angle):
	pulse = int(300 + (angle /180.0)*410)
	set_pwm(channel, 0, pulse)

def response(channel, start, end, step, wait):
        print(f'channel:{channel}test start! ({start}~{end})')
        if start < end:
            for pulse in range(start, end+1, step):
                    #print(f"pulse={pulse}")
                    set_pwm(channel, 0, pulse)
                    time.sleep(wait)
        elif end < start:
            for pulse in range(start, end-1, -step):
                    #print(f"pulse={pulse}")
                    set_pwm(channel, 0, pulse)
                    time.sleep(wait)
        elif end == start:
            set_pwm(channel,0,start)
            time.sleep(wait)
        print("\n end!")


#펄스 범위
SERVOMIN_YAW = 400
SERVOMAX_YAW = 1940
SERVOMIN_SHOULDER = 500   #shoulder middle 970
SERVOMAX_SHOULDER = 1642
SERVOMIN_ELBOW = 572      #elbow middle 1150->1300
SERVOMAX_ELBOW = 1600
SERVOMIN_WRIST = 450
SERVOMAX_WRIST = 2100     #wrist middle 1275
SERVOMIN_WRIST_ROTATION = 346
SERVOMAX_WRIST_ROTATION = 2146		#1680(horizontal), 800(vertical)
SERVOMIN_GRIPPER = 1550
SERVOMAX_GRIPPER = 2000

#HOME_POSE(Initial Pose), 현재 위치 및 펄스 정보 저장하기
HOME_POSE = (180, 0, 210, 0, 0)  # x, y, z, wrist, wrist_rotation
prev_pulse = [1162, 796, 1353, 1331, 1676, SERVOMIN_GRIPPER]
prev_pose = HOME_POSE

# 각도 범위
YAW_MIN_DEG, YAW_MAX_DEG = -87, 87
SHOULDER_MIN_DEG, SHOULDER_MAX_DEG = 10, 146
ELBOW_MIN_DEG, ELBOW_MAX_DEG = -35, 85
WRIST_MIN_DEG, WRIST_MAX_DEG = -90, 90    #-90이 아래로 숙이는 방향
WRIST_ROT_MIN_DEG, WRIST_ROT_MAX_DEG = -133, 47 #0(horizontal), -90(vertical)

LINK1 = 100.0
LINK2 = 80.0
LINK3 = 139.0
EXTENSION_LIMIT = 173 #original limit: 173

GRIPPER_GAP_MIN = 0
GRIPPER_GAP_MAX = 42
GRIPPER_GAP_OFFSET = 1
GRIPPER_BAR_LENGTH = 25

X_ORIGIN_OFFSET = -3
Y_ORIGIN_OFFSET = -3
Z_ORIGIN_OFFSET = 108



def main():
    

    go_grip(249,19,3,-40,0)
    
    
    

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
    set_pwm(5, 0, pulse)
    time.sleep(1)

def gripper_trick(pulse):
    global prev_pulse
    
    response(5, prev_pulse[5], pulse, 1, 0.005)
    time.sleep(0.1)    
    prev_pulse[5]=pulse
    time.sleep(0.1)

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
    #print("yaw, shoulder, elbow, wrist, wristrot")
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
    global prev_pulse
    for i, pulse in enumerate(pulses):
        #print("%d" % i)
        #print("%d" % pulse)
        set_pwm(i,0,pulse)
        #pwm.set_pwm(i, 0, pulse)
        time.sleep(0.01)
        prev_pulse[i]=pulse


def move_servos_soft(pulses):
    global prev_pulse
    for i, pulse in enumerate(pulses):
        #print("%d" % i)
        #print("%d" % pulse)
        response(i, prev_pulse[i], pulse, 1, 0.01)
        #pwm.set_pwm(i, 0, pulse)
        time.sleep(0.01)
        prev_pulse[i]=pulse


def go_pose(x, y, z, wrist=0, rotation=0):
    pulses = inverse_kinematics(x, y, z, wrist, rotation)
    move_servos_soft(pulses)
    global prev_pose
    prev_pose = (x, y, z, wrist, rotation)

def go_test(x, y, z, wrist=0, rotation=0):
    pulses = inverse_kinematics(x, y, z, wrist, rotation)
    print(pulses)


def joint_move(target_pulses, step_pulses=5, step_delay=0.03):
    global prev_pulse
    # 현재 펄스값
    current_pulses = prev_pulse[:]
    # 각 조인트별 변화량
    diffs = [target - curr for target, curr in zip(target_pulses, current_pulses)]
    max_diff = max(abs(d) for d in diffs)
    if max_diff == 0:
        print("No movement needed")
        return
    steps = max_diff // step_pulses if step_pulses > 0 else 1

    for s in range(1, steps+1):
        interp_pulses = [
            int(round(curr + diff * s / steps))
            for curr, diff in zip(current_pulses, diffs)
        ]
        move_servos(interp_pulses)
        time.sleep(step_delay)
    # 마지막 위치 정확히 이동
    move_servos(target_pulses)

def joint_move_servo(x, y, z, wrist=0, rotation=0, step_pulses=5, step_delay=0.02):
    global prev_pose
    pulses = inverse_kinematics(x,y,z,wrist,rotation)
    joint_move(pulses, step_pulses, step_delay)
    prev_pose=(x,y,z,wrist,rotation)

def linear_move(x1, y1, z1, wrist=0, rotation=0, step_dist=5, step_delay=0.02):
    global prev_pose
    (x0, y0, z0, wrist0, rotation0) = prev_pose

    steps = int(
        max(
            abs(x1 - x0),
            abs(y1 - y0),
            abs(z1 - z0)
        ) // step_dist
    )
    if steps == 0:
        steps = 1

    for i in range(1, steps+1):
        xi = x0 + (x1 - x0) * i / steps
        yi = y0 + (y1 - y0) * i / steps
        zi = z0 + (z1 - z0) * i / steps
        wi = wrist0 + (wrist - wrist0) * i / steps
        ri = rotation0 + (rotation - rotation0) * i / steps
        try:
            pulses = inverse_kinematics(xi, yi, zi, wi, ri)
            move_servos(pulses)
            prev_pose = (xi, yi, zi, wi, ri)
        except ValueError as e:
            print("IK error at step %d: %s" % (i, e))
        time.sleep(step_delay)
    # 마지막 목표 위치 이동
    pulses = inverse_kinematics(x1, y1, z1, wrist, rotation)
    move_servos(pulses)
    prev_pose = (x1, y1, z1, wrist, rotation)

def move_move(x,y,z,wrist=0,rotation=0, step_pulses=5, step_delay=0.02):
    #find proper wrist angle without value error
    candidate_wrists = []
    for i_wrist in range(-90,1,5):
        valid = True
        try:
            i=inverse_kinematics(x,y,z,i_wrist,rotation)
        except ValueError:
            valid = False
        if valid:
            candidate_wrists.append(i_wrist)
    if not candidate_wrists:
        raise ValueError("No valid wrist angle found")
    chosen_wrist = min(candidate_wrists, key=lambda w: abs(w-wrist))
    print(f"chosen wrist angle: {chosen_wrist}")
    try:
        linear_move(x,y,z,chosen_wrist,rotation,step_pulses,step_delay)
    except ValueError as e:
        print("linear move value error:",e)
        try:
            joint_move_servo(x,y,z,chosen_wrist,rotation,step_pulses,step_delay)
        except ValueError as e:
            print("joint move value error:",e)
            go_pose(x,y,z,chosen_wrist,rotation)


def go_grip(x,y,z,wrist=0,rotation=0, step_pulses=5, step_delay=0.012):
    gripper_trick(SERVOMIN_GRIPPER)
    time.sleep(0.05)
    move_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    gripper_trick(SERVOMAX_GRIPPER)
    #move_move(x,y,65,wrist,rotation,step_pulses,step_delay)
    #move_move(*HOME_POSE,step_pulses,step_delay)

def go_lay_down(cls, step_pulses=5, step_delay=0.012):
    z_lay_down = 55
    if (cls == 'BJT'):
        move_move(179,-161,z_lay_down,-30,0,step_pulses,step_delay)
    elif (cls == 'Diode'):
        move_move(129,-101,z_lay_down,-65,0,step_pulses,step_delay)
    elif (cls == 'MOSFET'):
        move_move(179,-101,z_lay_down,-50,0,step_pulses,step_delay)
    elif (cls == 'OP_AMP'):
        move_move(229,-101,z_lay_down,-30,0,step_pulses,step_delay)
    elif (cls == 'Resistor'):
        move_move(79,-101,z_lay_down,-80,0,step_pulses,step_delay)
    elif (cls == 'cable'):
        move_move(129,-161,z_lay_down,-50,0,step_pulses,step_delay)
    elif (cls == 'capacitor'):
        move_move(79,-161,z_lay_down,-65,0,step_pulses,step_delay)
    elif (cls == 'variable_resistor'):
        move_move(229,-161,z_lay_down,-10,0,step_pulses,step_delay)
    gripper_trick(SERVOMIN_GRIPPER)
    move_move(*HOME_POSE,step_pulses,step_delay)

def z_cls(cls):
    if (cls == 'BJT'):
        z = 3
    elif (cls == 'Diode'):
        z = 3
    elif (cls == 'MOSFET'):
        z = 3
    elif (cls == 'OP_AMP'):
        z = 3
    elif (cls == 'Resistor'):
        z = 3
    elif (cls == 'cable'):
        z = 2
    elif (cls == 'capacitor'):
        z = 3
    elif (cls == 'variable_resistor'):
        z = 4

    return z



if __name__ == "__main__":
    main()


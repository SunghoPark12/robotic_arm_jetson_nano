import math
import time
import smbus
from detector import Detector
from calibration import CameraCalibrator
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
        print("\n end!")

#HOME_POSE(Initial Pose), 현재 위치 및 펄스 정보 저장하기
HOME_POSE = (180, 0, 210, 0, 0)  # x, y, z, wrist, wrist_rotation
prev_pulse = [1170, 874, 1187, 1314, 1676, 630]
prev_pose = HOME_POSE


#펄스 범위
SERVOMIN_YAW = 400
SERVOMAX_YAW = 1940
SERVOMIN_SHOULDER = 500   #shoulder middle 970
SERVOMAX_SHOULDER = 1642
SERVOMIN_ELBOW = 421      #elbow middle 1150
SERVOMAX_ELBOW = 1450
SERVOMIN_WRIST = 450
SERVOMAX_WRIST = 2100     #wrist middle 1275
SERVOMIN_WRIST_ROTATION = 346
SERVOMAX_WRIST_ROTATION = 2146		#1680(horizontal), 800(vertical)
SERVOMIN_GRIPPER = 355
SERVOMAX_GRIPPER = 630

# 각도 범위
YAW_MIN_DEG, YAW_MAX_DEG = -87, 87
SHOULDER_MIN_DEG, SHOULDER_MAX_DEG = 10, 146
ELBOW_MIN_DEG, ELBOW_MAX_DEG = -35, 85
WRIST_MIN_DEG, WRIST_MAX_DEG = -90, 90    #90이 아래로 숙이는 방향
WRIST_ROT_MIN_DEG, WRIST_ROT_MAX_DEG = -133, 47

LINK1 = 100.0
LINK2 = 80.0
LINK3 = 130.0
EXTENSION_LIMIT = 173

GRIPPER_GAP_MIN = 0
GRIPPER_GAP_MAX = 42
GRIPPER_GAP_OFFSET = 1
GRIPPER_BAR_LENGTH = 25

X_ORIGIN_OFFSET = -10
Y_ORIGIN_OFFSET = 0
Z_ORIGIN_OFFSET = 106



def main():
    
    #detector class
    detector = Detector(engine_path="yolov8n_640_NMS.engine", device="cuda:0")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    window_name = 'webcam'
    ret, frame = cap.read()
    if not ret:
        print("no signal webcam")

    #calbration class
    print("camera calibration")
    calibrator = CameraCalibrator(1280, 720, swap_xy=False, flip_x=False, flip_y=False)
    pixel_points = calibrator.get_pixel_points_by_click(frame)
    print(f"선택된 픽셀 기준점: {pixel_points}")
    if len(pixel_points) < 3:
        print("최소 3개 이상의 기준점이 필요합니다.")
 
    print("실측 좌표(mm) 입력")
    real_points = []
    
    for i in range(len(pixel_points)):
        while True:
            try:
                xy = input(f"기준점 {i+1}의 실측 좌표 (x y, mm 단위, 예: 120 80): ").strip()
                x, y = map(float, xy.split())
                real_points.append((x, y))
                break
            except Exception:
                print("잘못된 입력입니다. 예시처럼 입력하세요: 120 80")
    
    calibrator.set_calibration(pixel_points, real_points)

    while (True):
        cls, pixel_x, pixel_y = detector.obj_pixel(cap, window_name, 20)
        z_chosen = z_cls(cls) #proper z value for the class
        ret, frame = cap.read()
        img_disp = frame.copy()
        if not ret:
            print("no signal webcam")
        cv2.circle(img_disp, (int(round(pixel_x)), int(round(pixel_y))), 8, (0, 0, 255), -1)
        cv2.imshow("detect result", img_disp)
        print("press any key to move next")
        cv2.waitKey(0)
        real_x, real_y = calibrator.pixel_to_real(pixel_x, pixel_y)
        print(f"pixel_x: {pixel_x}, pixel_y: {pixel_y}")
        print(f"robot_x: {real_x}, robot_y: {real_y}")
        time.sleep(1)
        go_grip(real_x,real_y,z_chosen,-50,0)
        go_lay_down(cls)
    
    cap.release()
    cv2.destroyAllWindows()
        
    

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
    response(5, prev_pulse[5], pulse, 1, 0.01)
    time.sleep(0.5)
    prev_pulse[5]=pulse
    time.sleep(0.5)

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
    global prev_pulse
    for i, pulse in enumerate(pulses):
        print("%d" % i)
        print("%d" % pulse)
        set_pwm(i,0,pulse)
        #pwm.set_pwm(i, 0, pulse)
        time.sleep(0.01)
        prev_pulse[i]=pulse


def move_servos_soft(pulses):
    global prev_pulse
    for i, pulse in enumerate(pulses):
        print("%d" % i)
        print("%d" % pulse)
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

def joint_move_servo(x, y, z, wrist=0, rotation=0, step_pulses=5, step_delay=0.03):
    global prev_pose
    pulses = inverse_kinematics(x,y,z,wrist,rotation)
    joint_move(pulses, step_pulses, step_delay)
    prev_pose=(x,y,z,wrist,rotation)

def linear_move(x1, y1, z1, wrist=0, rotation=0, step_dist=5, step_delay=0.03):
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

def move_move(x,y,z,wrist=0,rotation=0, step_pulses=5, step_delay=0.03):
    try:
        linear_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    except ValueError as e:
        print("linear move value error:",e)
        try:
            joint_move_servo(x,y,z,wrist,rotation,step_pulses,step_delay)
        except ValueError as e:
            print("joint move value error:",e)
            go_pose(x,y,z,wrist,rotation)


def go_grip(x,y,z,wrist=0,rotation=0, step_pulses=5, step_delay=0.03):
    gripper_trick(355)
    time.sleep(0.05)
    move_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    gripper_trick(630)
    move_move(*HOME_POSE,step_pulses,step_delay)

def go_lay_down(cls, step_pulses=5, step_delay=0.03):
    if (cls == 'BJT'):
        move_move(190,-62,45,-50,0,step_pulses,step_delay)
    #elif (cls == 'Diode'):
    #    move_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    elif (cls == 'MOSFET'):
        move_move(223,-62,45,-50,0,step_pulses,step_delay)
    #elif (cls == 'OP_AMP'):
    #    move_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    #elif (cls == 'Resistor'):
    #    move_move(x,y,z,wrist,0,step_pulses,step_delay)
    elif (cls == 'cable'):
        move_move(223,-95,45,-50,0,step_pulses,step_delay)
    #elif (cls == 'capacitor'):
    #    move_move(x,y,z,wrist,rotation,step_pulses,step_delay)
    elif (cls == 'variable_resistor'):
        move_move(190,-95,45,-50,0,step_pulses,step_delay)
    gripper_trick(355)
    move_move(*HOME_POSE,step_pulses,step_delay)

def z_cls(cls):
    if (cls == 'BJT'):
        z = 8
    elif (cls == 'Diode'):
        z = 8
    elif (cls == 'MOSFET'):
        z = 8
    elif (cls == 'OP_AMP'):
        z = 8
    elif (cls == 'Resistor'):
        z = 8
    elif (cls == 'cable'):
        z = 8
    elif (cls == 'capacitor'):
        z = 8
    elif (cls == 'variable_resistor'):
        z = 12

    return z



if __name__ == "__main__":
    main()


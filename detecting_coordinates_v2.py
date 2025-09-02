from detector import Detector
from calibration_v2 import CameraCalibrator
import cv2
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
real_points = [] #(0,0),(10,0), (20,0), (30,0), (40,0), (0,20), (20,10)
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

pixel_x, pixel_y = detector.obj_pixel(cap, window_name, 20)
img_disp = frame.copy()
cv2.circle(img_disp, (int(round(pixel_x)), int(round(pixel_y))), 8, (0, 0, 255), -1)
cv2.imshow("detect result", img_disp)
print("press any key to move next")
cv2.waitKey(0)
real_x, real_y = calibrator.pixel_to_real(pixel_x, pixel_y)


print(f"pixel_x: {pixel_x}, pixel_y: {pixel_y}")
print(f"robot_x: {real_x}, robot_y: {real_y}")


cap.release()
cv2.destroyAllWindows()



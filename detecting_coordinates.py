from detector import Detector
from calibration import CameraCalibrator
import cv2
import numpy as np
real_X_OFFSET = -98
real_Y_OFFSET = 268
detector = Detector(engine_path="yolov8n_640_NMS.engine", device="cuda:0")
#pixel_points = [(1279,0),(0,719),(1279,719)]
#real_points = [(5,6),(159,94),(7,94)]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
window_name = 'webcam'
ret, frame = cap.read()
if not ret:
    print("no signal webcam")

#calbration class
print("camera calibration")
calibrator = CameraCalibrator(flip_x=True)
pixel_points = calibrator.get_pixel_points_by_click(frame, num_points=4)

 
real_points = np.array([
    [150, 20],
    [30, 30],
    [120, 70],
    [40, 60]
], dtype=np.float32)

calibrator.set_calibration(pixel_points, real_points)

pixel_x, pixel_y = detector.obj_pixel(cap, window_name, 20)
real_x, real_y = calibrator.pixel_to_real(pixel_x, pixel_y)
robot_x = real_y + real_Y_OFFSET
robot_y = real_x + real_X_OFFSET

print(f"pixel_x: {pixel_x}, pixel_y: {pixel_y}")
print(f"real_x: {real_x}, real_y: {real_y}")
print(f"robot_x: {robot_x}, robot_y: {robot_y}")

#results_batch = detector.detect_n_frames_with_overlay(cap, window_name, 15)
#smoothed = detector.smooth_all_objects(results_batch)
#print(f"smoothing result:", smoothed)
#first_obj=max(smoothed, key=lambda x: x['score'])
#x_center = first_obj['bbox'][0] + (first_obj['bbox'][2] / 2)
#y_center = first_obj['bbox'][1] + (first_obj['bbox'][3] / 2)

cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np

class CameraCalibrator:
    def __init__(self, pixel_points=None, real_points=None, swap_xy=False, flip_x=False, flip_y=False):
        """
        pixel_points: [(u1,v1), (u2,v2), ...]  # 웹캠 기준점 픽셀좌표
        real_points:  [(x1,y1), (x2,y2), ...]  # 기준점의 실측(mm) 좌표
        swap_xy: x와 y축을 바꿀지 여부 (좌표계가 90도 회전된 경우)
        flip_x: x축을 뒤집을지 여부 (좌우 반전)
        flip_y: y축을 뒤집을지 여부 (상하 반전)
        """
        self.swap_xy = swap_xy
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.M = None		
        self.mode = None

        if pixel_points is not None and real_points is not None:
            self.set_calibration(pixel_points, real_points)

    def set_calibration(self, pixel_points, real_points):
        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.real_points = np.array(real_points, dtype=np.float32)
        self.M = cv2.getAffineTransform(self.pixel_points, self.real_points)
        self.mode = 'affine'
        if len(pixel_points) <=2:
            raise ValueError("최소 3개(affine), 4개(homography) 기준점 필요")

#        if len(pixel_points) == 3:
#            self.M = cv2.getAffineTransform(self.pixel_points, self.real_points)
#            self.mode = 'affine'
#        elif len(pixel_points) >= 4:
#            self.M, _ = cv2.findHomography(self.pixel_points, self.real_points)
#            self.mode = 'homography'



    def apply_axis_transform(self, x, y):
        if self.swap_xy:
            x, y = y, x
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return x, y

    def pixel_to_real(self, u, v):
        if self.M is None:
            raise ValueError("변환 행렬이 설정되지 않았습니다.")
        u, v = self.apply_axis_transform(u, v)
        pts = np.array([[[u, v]]], dtype=np.float32)
        if self.mode == 'affine':
            dst = cv2.transform(pts, self.M)
        else:
            dst = cv2.perspectiveTransform(pts, self.M)
        return tuple(dst[0][0])

    def real_to_pixel(self, x, y):
        if self.M is None:
            raise ValueError("변환 행렬이 설정되지 않았습니다.")
        x, y = self.apply_axis_transform(x, y)
        if self.mode == 'affine':
            M_inv = cv2.invertAffineTransform(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.transform(pts, M_inv)
        else:
            M_inv = np.linalg.inv(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pts, M_inv)
        return tuple(dst[0][0])

    def get_pixel_points_by_click(self, frame, num_points=4):
        """
        이미지에서 마우스로 기준점 좌표를 클릭하여 수집
        """
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < num_points:
                    print(f"[{len(points)+1}] 클릭 좌표: ({x}, {y})")
                    points.append([x, y])
                    cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Click Points", param)
                if len(points) == num_points:
                    cv2.destroyWindow("Click Points")

        clone = frame.copy()
        cv2.imshow("Click Points", clone)
        cv2.setMouseCallback("Click Points", click_event, clone)
        cv2.waitKey(0)

        return np.array(points, dtype=np.float32)

'''
import cv2
import numpy as np

class CameraCalibrator:
    def __init__(self, pixel_points, real_points, swap_xy=False, flip_x=False, flip_y=False):
        """
        pixel_points: [(u1,v1), (u2,v2), ...]  # 웹캠 기준점 픽셀좌표
        real_points:  [(x1,y1), (x2,y2), ...]  # 기준점의 실측(mm) 좌표
        swap_xy: x와 y축을 바꿀지 여부 (좌표계가 90도 회전된 경우)
        flip_x: x축을 뒤집을지 여부 (좌우 반전)
        flip_y: y축을 뒤집을지 여부 (상하 반전)
        """
        self.swap_xy = swap_xy
        self.flip_x = flip_x
        self.flip_y = flip_y

        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.real_points = np.array(real_points, dtype=np.float32)
        # Affine(3점) 또는 Homography(4점 이상) 자동 선택
        if len(pixel_points) == 3:
            self.M = cv2.getAffineTransform(self.pixel_points, self.real_points)
            self.mode = 'affine'
        elif len(pixel_points) >= 4:
            self.M, _ = cv2.findHomography(self.pixel_points, self.real_points)
            self.mode = 'homography'
        else:
            raise ValueError("최소 3개(affine), 4개(homography) 기준점 필요")

    def apply_axis_transform(self, x, y):
        if self.swap_xy:
            x, y = y, x
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return x, y

    def pixel_to_real(self, u, v):
        # 축 변환 적용
        u, v = self.apply_axis_transform(u, v)
        pts = np.array([[u, v]], dtype=np.float32)
        if self.mode == 'affine':
            pts = np.array([pts])
            dst = cv2.transform(pts, self.M)
            x, y = dst[0][0]
        else:  # homography
            pts = np.array([pts])
            dst = cv2.perspectiveTransform(pts, self.M)
            x, y = dst[0][0]
        return x, y

    def real_to_pixel(self, x, y):
        # 축 변환 적용
        x, y = self.apply_axis_transform(x, y)
        if self.mode == 'affine':
            M_inv = cv2.invertAffineTransform(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.transform(pts, M_inv)
            u, v = dst[0][0]
        else:
            M_inv = np.linalg.inv(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pts, M_inv)
            u, v = dst[0][0]
        return u, v
'''

import cv2
import numpy as np

class CameraCalibrator:
    def __init__(self, image_width, image_height,
                 swap_xy=False, flip_x=False, flip_y=False):
        """
        image_width, image_height: 입력 이미지 크기 (flip 보정시 필요)
        swap_xy: x/y축 전환
        flip_x: x축 flip (좌우반전)
        flip_y: y축 flip (상하반전)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.swap_xy = swap_xy
        self.flip_x = flip_x
        self.flip_y = flip_y

        self.pixel_points = None
        self.real_points = None
        self.M = None
        self.mode = None

    def _apply_axis_transform(self, u, v):
        # 기준점 입력시 flip/swap이 필요한 경우 변환
        x, y = u, v
        if self.swap_xy:
            x, y = y, x
        if self.flip_x:
            x = self.image_width - x
        if self.flip_y:
            y = self.image_height - y
        return x, y

    def set_calibration(self, pixel_points, real_points):
        # 기준점 쌍에 대해 flip/swap 적용 후 행렬 생성
        pixel_pts_adj = [self._apply_axis_transform(u, v) for u, v in pixel_points]
        self.pixel_points = np.array(pixel_pts_adj, dtype=np.float32)
        self.real_points = np.array(real_points, dtype=np.float32)

        if len(self.pixel_points) == 3:
            self.M = cv2.getAffineTransform(self.pixel_points, self.real_points)
            self.mode = 'affine'
        elif len(self.pixel_points) >= 4:
            self.M, _ = cv2.findHomography(self.pixel_points, self.real_points)
            self.mode = 'homography'
        else:
            raise ValueError("최소 3개(affine), 4개(homography) 기준점 필요")

    def pixel_to_real(self, u, v):
        # flip/swap은 이미 행렬 생성시 적용됐으므로, 변환만 수행
        if self.M is None:
            raise ValueError("변환 행렬이 설정되지 않았습니다.")
        pts = np.array([[[u, v]]], dtype=np.float32)
        if self.mode == 'affine':
            dst = cv2.transform(pts, self.M)
        else:
            dst = cv2.perspectiveTransform(pts, self.M)
        return tuple(dst[0][0])

    def real_to_pixel(self, x, y):
        if self.M is None:
            raise ValueError("변환 행렬이 설정되지 않았습니다.")
        if self.mode == 'affine':
            M_inv = cv2.invertAffineTransform(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.transform(pts, M_inv)
        else:
            M_inv = np.linalg.inv(self.M)
            pts = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pts, M_inv)
        return tuple(dst[0][0])

    def get_pixel_points_by_click(self, frame):
        """
        마우스 클릭으로 원하는 만큼 기준점 입력.  
        기준점 입력은 마우스 클릭 → ESC(27) 또는 Enter(13) 키로 종료.
        """
        points = []
        clone = frame.copy()

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"[{len(points)+1}] 클릭 좌표: ({x}, {y})")
                points.append([x, y])
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Click Points", param)

        cv2.imshow("Click Points", clone)
        cv2.setMouseCallback("Click Points", click_event, clone)

        print("마우스를 클릭하여 기준점을 입력하세요. (ESC 또는 Enter로 종료)")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == 13:  # ESC 또는 Enter
                break

        cv2.destroyWindow("Click Points")
        return np.array(points, dtype=np.float32)

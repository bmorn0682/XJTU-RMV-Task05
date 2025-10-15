import cv2
import numpy as np
import yaml


class Armor3DModel:
    def __init__(self, armor_width=135, armor_height=56):
        """
        定义装甲板的3D模型
        Args:
            armor_width: 装甲板宽度(mm)
            armor_height: 装甲板高度(mm)
        """
        self.width = armor_width
        self.height = armor_height

        # 定义装甲板四个角点的3D坐标（以装甲板中心为原点）
        # 坐标顺序：左上、右上、右下、左下
        half_w = armor_width / 2
        half_h = armor_height / 2

        self.obj_points = np.array([
            [-half_w, -half_h, 0],  # 左上
            [half_w, -half_h, 0],  # 右上
            [half_w, half_h, 0],  # 右下
            [-half_w, half_h, 0]  # 左下
        ], dtype=np.float32)

class PnPSolver:
    def __init__(self, camera_params, armor_model):
        self.camera_params = camera_params
        self.armor_model = armor_model

    def solve_pnp(self, image_points):
        """
        求解PnP问题
        Args:
            image_points: 装甲板四个角点的2D图像坐标 [左上, 右上, 右下, 左下]
        Returns:
            success: 是否求解成功
            rvec: 旋转向量
            tvec: 平移向量
        """
        if len(image_points) != 4:
            return False, None, None

        cameraMatrix = np.array(self.camera_params['IntrinsicMatrix'], dtype=np.float32)
        disCoeffs = np.array(self.camera_params['distCoeffs'],dtype=np.float32)
        # 将2D点转换为numpy数组
        img_pts = np.array(image_points, dtype=np.float32)

        try:
            # 使用SOLVEPNP_IPPE方法，适合于平面图像
            success, rvec, tvec = cv2.solvePnP(
                self.armor_model.obj_points,
                img_pts,
                cameraMatrix,
                disCoeffs,
                flags=cv2.SOLVEPNP_IPPE
            )

            return success, rvec, tvec

        except Exception as e:
            print(f"PnP求解错误: {e}")
            return False, None, None

    def calculate_3d_coordinates(self, rvec, tvec):
        """
        计算装甲板在相机坐标系下的3D坐标
        Args:
            rvec: 旋转向量
            tvec: 平移向量
        Returns:
            armor_center_3d: 装甲板中心在相机坐标系下的3D坐标
            corners_3d: 装甲板四个角点在相机坐标系下的3D坐标
        """
        if rvec is None or tvec is None:
            return None, None

        # 将旋转向量转换为旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)

        # 计算装甲板中心在相机坐标系下的坐标
        armor_center_3d = tvec.flatten()

        # 计算四个角点在相机坐标系下的坐标
        corners_3d = []
        for obj_point in self.armor_model.obj_points:
            # 将物体坐标系下的点转换到相机坐标系
            point_3d = rmat @ obj_point.reshape(3, 1) + tvec
            corners_3d.append(point_3d.flatten())

        corners_3d = np.array(corners_3d)

        return armor_center_3d, corners_3d

    def calculate_distance(self, tvec):
        """计算装甲板到相机的距离"""
        if tvec is None:
            return -1
        return np.linalg.norm(tvec)

    def calculate_angle(self, tvec):
        """计算装甲板相对于相机的角度（俯仰角和偏航角）"""
        if tvec is None:
            return None, None

        x, y, z = tvec.flatten()

        # 偏航角 (yaw)
        yaw = np.arctan2(x, z) * 180 / np.pi

        # 俯仰角 (pitch)
        pitch = np.arctan2(y, z) * 180 / np.pi

        return yaw, pitch

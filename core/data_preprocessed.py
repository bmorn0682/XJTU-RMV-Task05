import cv2
import numpy as np
from typing import Tuple,Optional,List

def read_video_frame(cap: cv2.VideoCapture, delay_ms: int = 1) -> Optional[np.ndarray]:
    """
    读取视频帧

    Args:
        cap: 视频捕获对象
        delay_ms: 延迟时间（毫秒）

    Returns:
        视频帧，如果读取失败或用户中断则返回None
    """
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        return None

    # 检查是否用户按键中断
    if cv2.waitKey(delay_ms) >= 0:
        return None

    return frame


def find_rect(frame: np.ndarray) -> List[cv2.RotatedRect]:
    """
    处理帧，检测旋转矩形

    Args:
        frame: 输入图像帧

    Returns:
        检测到的旋转矩形列表
    """
    rects = []

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, dst = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    dst = cv2.dilate(dst, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 过滤小面积轮廓
        if cv2.contourArea(contour) <= 100:
            continue

        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)

        # 计算矩形面积
        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height

        # 过滤面积比例不合适的矩形
        if rect_area > 0 and area / rect_area > 0.6:
            rects.append(rect)

    return rects


def find_similar_rect_pairs(
        rects: List[cv2.RotatedRect],
        angle_thresh: float = 10.0,
        y_thresh: float = 0.1,
        area_ratio_thresh: float = 2.0,
        aspect_min: float = 1.5,
        aspect_max: float = 8.0
) -> List[Tuple[int, int]]:
    """
    查找相似的矩形对

    Args:
        rects: 旋转矩形列表
        angle_thresh: 角度阈值
        y_thresh: Y坐标阈值
        area_ratio_thresh: 面积比例阈值
        aspect_min: 最小宽高比
        aspect_max: 最大宽高比

    Returns:
        匹配的矩形对索引列表
    """

    class RectFeature:
        def __init__(self, index: int, angle: float, x: float, y: float, area: float, rect: cv2.RotatedRect):
            self.index = index
            self.angle = angle
            self.x = x
            self.y = y
            self.area = area
            self.rect = rect

    matched_pairs = []
    feats = []

    # 提取矩形特征
    for i, rect in enumerate(rects):
        w, h = rect[1]
        area = w * h

        # 计算宽高比
        if min(w, h) > 0:
            aspect = max(w, h) / min(w, h)
        else:
            aspect = 0

        # 计算角度
        angle = rect[2]
        if w < h:
            angle += 90.0

        # 过滤宽高比不合适的矩形
        if aspect < aspect_min or aspect > aspect_max:
            continue

        center_x, center_y = rect[0]
        feats.append(RectFeature(i, angle, center_x, center_y, area, rect))

    # 按Y坐标排序
    feats.sort(key=lambda x: x.y)

    # 查找匹配对
    for i in range(len(feats)):
        r1 = feats[i]

        for j in range(i + 1, len(feats)):
            r2 = feats[j]

            # X坐标距离检查
            max_dim = max(r1.rect[1])
            if abs(r2.x - r1.x) > 5 * max_dim:
                break

            # Y坐标距离检查
            y_diff = r2.y - r1.y
            max_y = max(r1.y, r2.y)
            if max_y > 0 and y_diff / max_y > y_thresh * r1.rect[1][1]:
                break

            # 角度差和面积比例检查
            d_angle = abs(r1.angle - r2.angle)
            if r1.area > 0 and r2.area > 0:
                area_ratio = max(r1.area, r2.area) / min(r1.area, r2.area)
            else:
                area_ratio = float('inf')

            if d_angle <= angle_thresh and area_ratio <= area_ratio_thresh:
                matched_pairs.append((r1.index, r2.index))

    return matched_pairs


def sort_corners(pts: np.ndarray) -> List[Tuple[float,float]]:
    """
    对矩形的四个角点进行排序（左上、右上、右下、左下）

    Args:
        pts: 输入的四个角点

    Returns:
        排序后的角点列表
    """
    # 将点转换为列表
    points = [(float(pt[0]), float(pt[1])) for pt in pts]
    # 按Y坐标排序
    points.sort(key=lambda p: (p[1], p[0]))

    # 分离上下两部分
    top_points = points[:2]
    bottom_points = points[2:]

    # 左上和右上
    top_points.sort(key=lambda p: p[0])
    tl, tr = top_points

    # 左下和右下
    bottom_points.sort(key=lambda p: p[0])
    bl, br = bottom_points

    return [tl, tr, br, bl]


def combined_rect_points(
        image: np.ndarray,
        rects: List[cv2.RotatedRect],
        pairs: List[Tuple[int, int]]
) -> set[list[tuple[float, float]]] | None:
    """
    在图像上绘制组合矩形

    Args:
        image: 输入图像
        rects: 旋转矩形列表
        pairs: 矩形对列表
    """
    if image is None or len(image.shape) == 0 or len(rects) == 0 or len(pairs) == 0:
        return

    factor = 0.7
    combos = set()
    for pair in pairs:
        i, j = pair
        if i >= len(rects) or j >= len(rects):
            continue

        color = (0, 0, 255)  # 红色

        # 获取矩形的四个角点
        ptsA = cv2.boxPoints(rects[i])
        ptsB = cv2.boxPoints(rects[j])

        # 对角点进行排序
        cornersA = sort_corners(ptsA)
        cornersB = sort_corners(ptsB)

        # 计算扩展向量
        height_vec = (cornersA[3][0] - cornersA[0][0], cornersA[3][1] - cornersA[0][1])  # 左下 - 左上
        expand_vec = (height_vec[0] * factor, height_vec[1] * factor)

        # 扩展矩形A的上下边
        cornersA = [
            (cornersA[0][0] - expand_vec[0], cornersA[0][1] - expand_vec[1]),  # 上边向上扩展
            cornersA[1],
            cornersA[2],
            (cornersA[3][0] + expand_vec[0], cornersA[3][1] + expand_vec[1])  # 下边向下扩展
        ]

        # 扩展矩形B的上下边
        cornersB = [
            cornersB[0],
            (cornersB[1][0] - expand_vec[0], cornersB[1][1] - expand_vec[1]),  # 上边向上扩展
            (cornersB[2][0] + expand_vec[0], cornersB[2][1] + expand_vec[1]),  # 下边向下扩展
            cornersB[3]
        ]
        # 创建组合四边形
        combo = [
            cornersA[0],  # 左上
            cornersB[1],  # 右上
            cornersB[2],  # 右下
            cornersA[3]  # 左下
        ]

        combos.add(tuple(combo))


    return combos

def extract_rotated_rect(image, points):
    """
    通过透视变换提取倾斜矩形区域，并将其矫正为正面矩形
    points: 倾斜矩形的四个顶点，顺序为 [左上, 右上, 右下, 左下]
    """
    # 将点转换为float32
    points = np.array(points, dtype=np.float32)

    # 2. 计算平行四边形的实际宽度和高度（基于顶点距离）
    # 宽度：左上→右上的水平距离（或左下→右下的水平距离，平行四边形对边相等）
    width = int(np.linalg.norm(points[1] - points[0]))
    # 高度：左上→左下的垂直距离（或右上→右下的垂直距离）
    height = int(np.linalg.norm(points[3] - points[0]))

    # 定义目标矩形的四个点
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(points, dst_points)

    # 应用透视变换
    result = cv2.warpPerspective(image,M = matrix, dsize = (width, height))

    return result

def update_params(pixel, params):
    if pixel[0] < params[0]:
        params[0] = pixel[0]
    if pixel[1] < params[1]:
        params[1] = pixel[1]
    if pixel[0] > params[2]:
        params[2] = pixel[0]
    if pixel[1] > params[3]:
        params[3] = pixel[1]

    return params

def number_extraction(mat, initial = 0.1, step = 3):
    _shape = mat.shape[:2]
    _binary_data = mat[:,:,0]
    height, width = _shape

    # 计算中心坐标
    center_y, center_x = height // 2, width // 2

    # 计算矩形区域的半径（像素）
    radius_y = int(initial/2 * height)
    radius_x = int(initial/2 * width)

    # 计算切片边界
    y_start = max(0, center_y - radius_y)
    y_end = min(height-1, center_y + radius_y)
    x_start = max(0, center_x - radius_x)
    x_end = min(width-1, center_x + radius_x)

    # 提取区域
    initial_part = _binary_data[y_start:y_end, x_start:x_end]
    params = [20000, 20000, 0, 0]
    directions = [          (0,-step),
                  (-step,0),          (step, 0),
                            (0, step)]

    if np.sum(initial_part) == 0:
        return -1

    flag =False
    initial_pixel = center_x,center_y
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            if _binary_data[x,y] == 255:
                initial_pixel = x,y
                flag = True
                break
        if flag:
            break
    search_points = {initial_pixel}
    visited = set()
    while True:
        clone_setting = set()
        for i in search_points:
            for direction in directions:
                new_one_x = int(i[0]) + direction[0]
                new_one_y = int(i[1]) + direction[1]
                if new_one_x < 0 or new_one_x >= _shape[0] or new_one_y < 0 or new_one_y >= _shape[1]:
                    continue
                new_one = new_one_x,new_one_y
                if _binary_data[new_one] == 255 and new_one not in visited:
                    clone_setting.add(new_one)
                    visited.add(new_one)
                    update_params(new_one,params)
        if not clone_setting:
            break
        search_points.clear()
        search_points = clone_setting.copy()

    return (params[1],params[0]),(params[3],params[2])

def image_prepared(img) -> Tuple[bool, cv2.Mat | None]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    points = number_extraction(dst, 0.2)
    if points == -1:
        return False, None
    src = dst[points[0][1]:points[1][1], points[0][0]:points[1][0]]
    src = cv2.resize(src, (16, 22))
    src = cv2.copyMakeBorder(src, 4, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    return True,src
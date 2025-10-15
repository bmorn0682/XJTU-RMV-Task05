import cv2
import yaml
import data_preprocessed as pr
import Predictor
import solvePnP

# 使用示例
def main():
    # 打开摄像头或视频文件
    cap = cv2.VideoCapture(r"..\video\blue.mp4")
    # 加载摄像头参数
    with open(r"..\config\CameraParams.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    # 建立装甲板模型
    armor = solvePnP.Armor3DModel()
    # 创建PnP对象
    pnp = solvePnP.PnPSolver(params,armor)
    # 加载MLP模型
    predictor = Predictor.ModelPredictor(r"..\model\number_mlp_model.pth")

    while True:
        # 读取视频流
        frame = pr.read_video_frame(cap)
        if frame is None:
            break
        # 处理帧并检测矩形
        rects = pr.find_rect(frame)
        # 查找相似的矩形对
        pairs = pr.find_similar_rect_pairs(rects)
        # 获取组合矩形角点
        combos = pr.combined_rect_points(frame, rects, pairs)
        if combos is None:
            print("Error Occurred")
        # 遍历矩形角点
        for combo in combos:
            # 对角点排序，确保顺序正确
            combo = pr.sort_corners(combo)
            # 将倾斜矩阵通过透视变换正交化，提高数字识别准确度
            pre = pr.extract_rotated_rect(frame,combo)
            # BFS截取数字，并进行图像处理，将图像放缩到样本集尺寸(20，28)
            status,src = pr.image_prepared(pre)
            if not status:
                print("running error")
            # 进行图像类别预测
            result = predictor.predict_image(src)
            preclass = result['predicted_class'] + 1
            # print(preclass)

            if preclass == 9: # 若为负样本，则跳过后续步骤
                continue
            else:
                # 绘制组合四边形
                for k in range(4):
                    pt1 = tuple(map(int, combo[k]))
                    pt2 = tuple(map(int, combo[(k + 1) % 4]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                center_x = (combo[0][0] + combo[1][0]) * 0.5
                center_y = (combo[0][1] + combo[3][1]) * 0.5
                center_point = (int(center_x), int(center_y))
                label = f"class:{preclass}"
                cv2.putText(frame, label, center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # 进行PnP解算，获取旋转矩阵和平移向量
                success, rvec, tvec = pnp.solve_pnp(combo)
                if success:
                    # 获取装甲板中心，角点的实际坐标，距离和俯仰角和偏航角
                    armor_center_3d, corners_3d = pnp.calculate_3d_coordinates(rvec, tvec)
                    distance = pnp.calculate_distance(tvec)
                    cangle = pnp.calculate_angle(tvec)
                    # print(distance, cangle)


                    center_point = (int(center_x), int(center_y + 30))
                    label = f"distance:{distance:.4f},yaw:{cangle[0]:.4f}  pitch:{cangle[1]:.4f}"
                    cv2.putText(frame, label, center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
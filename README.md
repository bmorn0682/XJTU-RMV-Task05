# XJTU-RMV-Task05
Task05-jiaming bai

## 一、任务回顾

- [x] 1. 支持调用本地视频和相机实时取流，完成稳定识别装甲板
- [x] 2. 完成pnp结算，求出装甲板的在相机坐标系下的坐标
- [x] 3. 训练模型，检测装甲板数字，提高装甲板的识别精度
- [ ] 4. 使用ROS2实现上述功能

## 二、项目简述

第一阶段：本项目通过结合前四次作业成果，并结合实际需求对代码做出改进以完成第一阶段作业；

第二阶段：调用MATLAB的CameraCalibrator工具箱完成相机标定获得相机的内参矩阵和畸变系数，再通过装甲板标定，利用cv2的solvePnP解算旋转矩阵和平移向量，完成第二阶段作业；

第三阶段：利用Python的pytorch训练MLP模型，所用模型特征为输入层为20 * 28 * 3，输出层为九种类型的相似度，对数据集进行训练和测试。并根据数据集样本特征，开发相应的图像预处理函数，以提高识别准确率。

## 三、代码结构
```
ActionDetecion/
├── core/                           # 项目核心包
│   ├── ActionDetection.py          # 主程序入口（调用其他代码，完成装甲板识别、数字识别、PnP解算功能）
│   ├── solvePnP.py                 # PnP解算功能包
|   ├── data_preprocessed.py        # 图像预处理功能包（包含装甲板识别，图像处理等功能）
|   ├── data_preprocessed.py        # 图像预处理功能包（包含装甲板识别，图像处理等功能）
|   ├── Predictor.py                # 模型预测功能包（包含加载模型，图像分类预测等功能）
│   ├── Cpp/                        # C++头文件（包含装甲板识别功能）
│   │   ├── ActionDetection.hpp     # 装甲板识别头文件
│   │   └── ActionDetection.cpp     # 同名源代码
├── video/                          # 视频文件夹
│   │   ├── blue.mp4                # 原视频
│   │   ├── blue_Processed.mp4      # 处理后视频
├── config/                         # 配置相关
│   │   ├── calibrationSession.mat  # matlab参数文件
│   │   └── CameraParams.yaml       # 相机参数文件
├── image/                          # 图像相关
│   ├── training_loss_accuracy.png  # 模型训练效果图
│   └── test/                       # 若干测试图像
├── model/                          # 模型相关
│   ├── numberType.py               # 模型训练程序（需配置数据库位置）
│   └── number_mlp_model.pth        # 模型文件
├── requirements.txt                # 依赖清单
└── README.md                       # 项目说明（功能、安装、使用）
```

## 四、代码说明

- 1、在识别装甲板的时候，巧妙地通过两灯条倾斜角是否接近（角度差小于5°），并以灯条中心纵坐标差，长宽比等作为辅助判准，精准识别装甲板，并能在测试视频中无漏帧、错帧。
- 2、在截取装甲板范围时，为精准截取，采用了倾斜矩形框选（RotatedRect），因此在数字识别之前必须进行图像的变换，以得到正矩形方便后续处理，因此我采用了透视仿射变换，得到标准图像
- 3、为了避免灯条及其他噪声影响数字范围截取，采用BFS从第2步中得到的图像中心开始搜索所有白色区域，并获取边界值，以精确截取数字区域，提高模型预测准确性
- 4、训练模型采用的是3层线性层，激活函数采用RELU的MLP模型，模型训练效果良好，平均准确率达到了99.94%。且对测试视频的实时预测正确率达到了100%
- 5、代码的具体运行逻辑可参考ActionDetection.py中的main函数，每一步都进行了注释，能够很好的体现代码逻辑

## 五、模型训练结果展示

![模型训练损失函数和正确率](https://github.com/bmorn0682/XJTU-RMV-Task05/tree/main/image/training_loss_accuracy.png)

## 六、模型测试说明
对ActionDetection.py的main函数做如下修改即可
```python
def main():
    # 加载MLP模型
    predictor = Predictor.ModelPredictor(r"..\model\number_mlp_model.pth")
    frame = cv2.imread(path) #将path替换为图片真实路径
    status,src = pr.image_prepared(pre)
    if not status:
        print("running error")
    # 进行图像类别预测
    result = predictor.predict_image(src)
    preclass = result['predicted_class'] + 1
    print(preclass)
```

## 七、依赖说明
本项目依赖均为python库，主要依赖有pytorch, cv2, yaml, numpy, pillow, torchvision, matplotlib

依赖安装代码
```bash
sudo apt update  # 更新软件源
sudo apt install python3-pip  # 安装 pip3
sudo apt install python3-venv  # 安装虚拟环境工具
```
创建并激活虚拟环境
```bash
# 在项目目录创建虚拟环境
python3 -m venv AcitonDetection_env

# 激活虚拟环境（激活后命令行前会显示 (AcitonDetection_env)）
source AcitonDetection_env/bin/activate
```
安装依赖库
```bash
# 在虚拟环境中安装库（此时直接用 pip 即可，无需 pip3）
pip install opencv-python numpy torch pyyaml pillow matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 退出虚拟环境
deactivate
```

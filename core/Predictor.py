import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import NumberType as NT

class ModelPredictor:
    def __init__(self, model_path, device='auto'):
        """
        模型预测器
        model_path: 训练好的模型路径
        device: 运行设备
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.transform = None

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            # 加载检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 获取模型参数
            hidden_sizes = checkpoint.get('hidden_sizes', [512, 256, 128])
            num_classes = checkpoint.get('num_classes', 9)
            dropout_rate = checkpoint.get('dropout_rate', 0.5)

            # 创建模型（使用之前定义的MLP）
            self.model = NT.MLP(
                hidden_sizes=hidden_sizes,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            ).to(self.device)

            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # 设置为评估模式

            # 获取类别信息
            self.class_names = checkpoint.get('class_names', None)
            if self.class_names is None and 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                self.class_names = {v: k for k, v in class_to_idx.items()}

            print(f"模型加载成功: {self.model_path}")
            print(f"类别数量: {num_classes}")
            if self.class_names:
                print(f"类别名称: {self.class_names}")

            # 创建数据预处理（与训练时一致）
            self.transform = transforms.Compose([
                transforms.Resize((28, 20)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # 假设是三通道
            ])

        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def predict_image(self, image):
        """
        预测单张图像
        image: 可以是文件路径、PIL图像、numpy数组或OpenCV图像
        """
        # 预处理图像
        tensor_image = self._preprocess_image(image)
        if tensor_image is None:
            return None

        # 添加批次维度
        tensor_image = tensor_image.unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            result = {
                'predicted_class': predicted.item(),
                'confidence': confidence.item(),
                'all_probabilities': probabilities.cpu().numpy()[0]
            }

            # 添加类别名称（如果有）
            if self.class_names:
                result['class_name'] = self.class_names.get(result['predicted_class'],
                                                            f"Class_{result['predicted_class']}")

            return result

    def predict_batch(self, images):
        """
        批量预测
        images: 图像列表
        """
        if not images:
            return []

        # 预处理所有图像
        tensor_images = []
        valid_indices = []

        for i, img in enumerate(images):
            tensor_img = self._preprocess_image(img)
            if tensor_img is not None:
                tensor_images.append(tensor_img)
                valid_indices.append(i)

        if not tensor_images:
            return []

        # 堆叠成批次
        batch_tensor = torch.stack(tensor_images).to(self.device)

        # 批量预测
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            results = []
            for i, (pred, conf) in enumerate(zip(predictions.cpu().numpy(),
                                                 confidences.cpu().numpy())):
                result = {
                    'index': valid_indices[i],
                    'predicted_class': pred,
                    'confidence': conf
                }

                if self.class_names:
                    result['class_name'] = self.class_names.get(pred, f"Class_{pred}")

                results.append(result)

            return results

    def _preprocess_image(self, image):
        """预处理图像"""
        try:
            # 处理不同类型的输入
            if isinstance(image, str):
                # 文件路径
                pil_image = Image.open(image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            elif isinstance(image, Image.Image):
                # PIL图像
                pil_image = image
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            elif isinstance(image, np.ndarray):
                # OpenCV/numpy数组 (BGR格式)
                if len(image.shape) == 3:
                    # 转换BGR到RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    # 灰度图，转换为RGB
                    pil_image = Image.fromarray(image).convert('RGB')
            else:
                print(f"不支持的图像类型: {type(image)}")
                return None

            # 应用变换
            tensor_image = self.transform(pil_image)
            return tensor_image

        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None

    def get_top_predictions(self, image, top_k=3):
        """
        获取前K个预测结果
        """
        result = self.predict_image(image)
        if result is None:
            return None

        # 获取所有类别的概率
        all_probs = result['all_probabilities']

        # 获取前K个预测
        top_indices = np.argsort(all_probs)[-top_k:][::-1]
        top_predictions = []

        for idx in top_indices:
            pred = {
                'class': idx,
                'probability': all_probs[idx]
            }
            if self.class_names:
                pred['class_name'] = self.class_names.get(idx, f"Class_{idx}")
            top_predictions.append(pred)

        result['top_predictions'] = top_predictions
        return result

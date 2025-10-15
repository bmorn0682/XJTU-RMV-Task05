import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np




def count_class_samples(dataset, full_dataset):
    """统计数据集中每个类别的样本数"""
    labels = [full_dataset.imgs[i][1] for i in dataset.indices]
    class_counts = {cls: labels.count(cls) for cls in set(labels)}
    return class_counts




class MLP(nn.Module):
    def __init__(self, input_size=1680, hidden_sizes=[512, 256], num_classes=9, dropout_rate=0.5):
        super(MLP, self).__init__()

        # 构建隐藏层
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout防止过拟合
            prev_size = hidden_size

        # 隐藏层序列
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        # 展平输入 (batch_size, 3, 28, 20) -> (batch_size, 1560)
        x = x.view(x.size(0), -1)

        # 通过隐藏层
        x = self.hidden_layers(x)

        # 输出层（不使用softmax，因为CrossEntropyLoss会自动处理）
        x = self.output_layer(x)

        return x

    def train_epoch(self, train_loader, criterion, optimizer, device):
        """训练一个epoch"""
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        return train_loss, train_acc

    def validate_epoch(self, val_loader, criterion, device):
        """验证"""
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc




def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



 # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()


def evaluate_model(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    print(f'Test Results:')
    print(f'  Average Loss: {test_loss:.4f}')
    print(f'  Accuracy: {test_acc:.2f}%')

    return test_acc

def save_model(model, optimizer, epoch, path='best_model.pth'):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path='best_model.pth'):
    """加载模型"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}")
    return model, optimizer, epoch


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 定义变换
    transform = transforms.Compose([
        transforms.Resize((28, 20)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. 加载完整数据集
    full_dataset = datasets.ImageFolder(root='C:\\Users\\22949\\Desktop\\datasets', transform=transform)
    class_names = full_dataset.classes  # 类别名称列表
    class_to_idx = full_dataset.class_to_idx  # 类别到索引的映射

    # 定义划分比例
    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    # 存储所有类别的训练/验证样本索引
    train_indices = []
    val_indices = []

    # 遍历每个类别
    for class_name in class_names:
        # 获取当前类别的所有样本索引
        # full_dataset.imgs 是 (图像路径, 类别索引) 的列表
        class_idx = class_to_idx[class_name]  # 当前类别的索引（如0,1,2...）
        # 筛选出当前类别的样本索引
        indices = [i for i, (path, label) in enumerate(full_dataset.imgs) if label == class_idx]

        # 打乱当前类别的样本顺序（保证随机性）
        np.random.seed(42)  # 固定种子，确保可复现
        np.random.shuffle(indices)

        # 计算当前类别的训练/验证样本数量
        n = len(indices)
        n_train = int(n * train_ratio)

        # 划分当前类别的训练/验证索引
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    # 根据索引创建训练集和验证集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 训练集类别分布
    train_counts = count_class_samples(train_dataset, full_dataset)
    # 验证集类别分布
    val_counts = count_class_samples(val_dataset, full_dataset)

    print("训练集类别分布：", train_counts)
    print("验证集类别分布：", val_counts)

    # 实例化模型
    model = MLP(
        input_size=28 * 20 * 3,
        hidden_sizes=[512, 256, 128],
        num_classes=9,
        dropout_rate=0.3
    ).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 训练参数
    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("开始训练...")
    best_val_acc = 0
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = model.train_epoch(train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = model.validate_epoch(val_loader, criterion, device)

        # 学习率调度
        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, num_epochs, 'mnist_mlp_model.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)


    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # 评估模型
    test_accuracy = evaluate_model(model, val_loader, device)

    print(f"Final Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
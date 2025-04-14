import os
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.residual_denoising_diffusion_pytorch import (
    Trainer, UnetRes, set_seed
)
from src.adversarial_diffusion import AdversarialResidualDiffusion


class TargetClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结基础特征提取层
        for param in self.model.parameters():
            param.requires_grad = False

        # 替换分类层
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)

        # 只训练分类层
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def train_classifier(model, train_loader, val_loader=None, num_epochs=10):
    """训练分类器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # 验证阶段
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            acc = 100. * correct / total
            print(f'Validation Accuracy: {acc:.2f}%')

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), './pre-model/resnet18_cat_dog_best.pth')

        print(f'Epoch {epoch + 1} average loss: {running_loss / len(train_loader):.4f}')

    # 保存最终模型
    torch.save(model.state_dict(), './pre-model/resnet18_cat_dog_final.pth')
    return model


def prepare_data(data_dir, batch_size=32):
    """准备数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_classifier_main():
    """训练分类器的主函数"""
    # 确保模型保存目录存在
    os.makedirs('./pre-model', exist_ok=True)

    # 准备数据
    data_dir = "./database/cat_dog"  # 包含cat和dog两个子文件夹的目录
    train_loader, val_loader = prepare_data(data_dir)

    # 创建并训练分类器
    classifier = TargetClassifier(num_classes=2)
    train_classifier(classifier, train_loader, val_loader, num_epochs=10)


def main():
    """对抗样本生成的主函数"""
    # 基础环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sys.stdout.flush()
    set_seed(10)

    # 加载训练好的分类器
    target_classifier = TargetClassifier(num_classes=2)
    target_classifier.load_state_dict(torch.load('./pre-model/resnet18_cat_dog_best.pth'))
    target_classifier = target_classifier.cuda()
    target_classifier.eval()

    # 调试模式配置
    debug = True
    if debug:
        save_and_sample_every = 2
        sampling_timesteps = 10
        train_num_steps = 200
    else:
        save_and_sample_every = 1000
        sampling_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        train_num_steps = 100000

    # 模型配置
    condition = True
    input_condition = False
    input_condition_mask = False

    # 数据路径配置
    folder = [
        "./database/cat_dog/cat.flist",
        "./database/cat_dog/dog.flist",
        "./database/cat_dog/dog_test.flist"
    ]

    # 模型参数设置
    model_config = {
        'image_size': 64,
        'num_unet': 2,
        'train_batch_size': 16,
        'num_samples': 16,
        'sum_scale': 1,
        'objective': 'pred_res_noise',
        'test_res_or_noise': "res_noise",
        'epsilon_vis': 1e-3,
        'attack_config': {
            'target_class': 1,  # 狗的类别标签
            'attack_loss_weight': 1.0,
            'classifier': target_classifier
        }
    }

    # 初始化UnetRes模型
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=model_config['num_unet'],
        condition=condition,
        input_condition=input_condition,
        objective=model_config['objective'],
        test_res_or_noise=model_config['test_res_or_noise'],
        img_to_img_translation=True,
        adversarial_mode=True
    )

    # 初始化对抗残差扩散模型
    diffusion = AdversarialResidualDiffusion(
        model,
        image_size=model_config['image_size'],
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective=model_config['objective'],
        loss_type='l2',
        condition=condition,
        sum_scale=model_config['sum_scale'],
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise=model_config['test_res_or_noise'],
        img_to_img_translation=True,
        epsilon_vis=model_config['epsilon_vis'],
        target_class=model_config['attack_config']['target_class'],
        classifier=model_config['attack_config']['classifier'],
        attack_loss_weight=model_config['attack_config']['attack_loss_weight']
    )

    # 初始化训练器
    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=model_config['train_batch_size'],
        num_samples=model_config['num_samples'],
        train_lr=2e-4,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        equalizeHist=False,
        crop_patch=False,
        generation=True,
        num_unet=model_config['num_unet']
    )

    # 训练对抗模型
    trainer.train()

    # 测试对抗样本生成
    if trainer.accelerator.is_local_main_process:
        trainer.load(trainer.train_num_steps // save_and_sample_every)
        results_folder = f'./results/adv_test_resnet18_{sampling_timesteps}'
        trainer.set_results_folder(results_folder)
        trainer.test(last=True)


if __name__ == '__main__':
    # 设置多进程启动方式
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 如果需要训练分类器
    if not os.path.exists('./pre-model/resnet18_cat_dog_best.pth'):
        print("Training classifier first...")
        train_classifier_main()

    # 运行对抗样本生成
    main()
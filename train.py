import os
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob
from pathlib import Path
from src.residual_denoising_diffusion_pytorch import (
    Trainer, UnetRes, set_seed
)
from src.adversarial_diffusion import AdversarialResidualDiffusion
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

# 主函数开始前添加
torch.cuda.empty_cache()


class TargetedAttackDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform

        self.source_images = sorted(glob.glob(os.path.join(source_dir, '*.*')))
        self.target_images = sorted(glob.glob(os.path.join(target_dir, '*.*')))

        assert len(self.source_images) > 0, f"No source images found in {source_dir}"
        assert len(self.target_images) > 0, f"No target images found in {target_dir}"

        print(f"Found {len(self.source_images)} source images and {len(self.target_images)} target images")

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        source_path = self.source_images[idx]
        source_img = Image.open(source_path).convert('RGB')

        target_idx = idx % len(self.target_images)
        target_path = self.target_images[target_idx]
        target_img = Image.open(target_path).convert('RGB')

        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        return source_img, target_img


def custom_collate(batch):
    source_imgs = []
    target_imgs = []

    for source_img, target_img in batch:
        source_imgs.append(source_img)
        target_imgs.append(target_img)

    # 使用torch.stack，但不设置out参数
    source_imgs = torch.stack(source_imgs)
    target_imgs = torch.stack(target_imgs)

    return source_imgs, target_imgs

def prepare_targeted_attack_data(source_dir, target_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize(144),  # 稍大的尺寸
        transforms.RandomCrop(128),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = TargetedAttackDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

def main():
    # 确保分类器在正确的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置参数
    config = {
        'train': {
            'source': "./database/train/source_images",
            'target': "./database/train/target_images"
        },
        'test': {
            'source': "./database/test/source_images",
            'target': "./database/test/target_images"
        },
        'batch_size': 2
    }

    # 创建必要的目录
    for paths in config.values():
        if isinstance(paths, dict):
            for path in paths.values():
                os.makedirs(path, exist_ok=True)

    # 加载预训练的ResNet18
    classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).cuda()
    classifier = classifier.to(device)
    classifier.eval()

    # 模型配置
    model_config = {
        'image_size': 128,
        'num_unet': 2,
        'train_batch_size': 4,
        'num_samples': 4,
        'sum_scale': 0.5,
        'objective': 'pred_res_noise',
        'test_res_or_noise': "res_noise",
        'epsilon_vis': 0.01,
        'attack_config': {
            'use_target_image': True,
            'attack_loss_weight': 0.1,
            'classifier': classifier
        }
    }

    # 准备数据加载器
    train_loader = prepare_targeted_attack_data(
        source_dir='./database/train/source_images',
        target_dir='./database/train/target_images',
        batch_size=model_config['train_batch_size']
    )

    # 初始化模型
    model = UnetRes(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        num_unet=model_config['num_unet'],
        condition=True,
        input_condition=False,
        objective=model_config['objective'],
        test_res_or_noise=model_config['test_res_or_noise'],
        img_to_img_translation=True,
        adversarial_mode=True,
        resnet_block_groups=8,  # 增加ResNet块的组数
    )

    # 初始化对抗残差扩散模型
    diffusion = AdversarialResidualDiffusion(
        model,
        image_size=model_config['image_size'],
        timesteps=1000,
        sampling_timesteps=10,
        objective=model_config['objective'],
        loss_type='l2',
        condition=True,
        sum_scale=model_config['sum_scale'],
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise=model_config['test_res_or_noise'],
        img_to_img_translation=True,
        epsilon_vis=model_config['epsilon_vis'],
        classifier=model_config['attack_config']['classifier'],
        attack_loss_weight=model_config['attack_config']['attack_loss_weight']
    )

    # 确保所有参数都需要梯度
    for param in model.parameters():
        param.requires_grad = True

    # 确保所有参数都需要梯度
    for param in diffusion.parameters():
        param.requires_grad = True

    # 初始化训练器
    trainer = Trainer(
        diffusion_model=diffusion,
        train_dataloader=train_loader,
        train_batch_size=model_config['train_batch_size'],
        train_lr=1e-4,
        train_num_steps=2000,
        gradient_accumulate_every=2,
        ema_decay=0.999,  # 提高EMA衰减率
        amp=True,
        results_folder='./results/sample',
        save_and_sample_every=200,
        condition=True,
        num_unet=model_config['num_unet']
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
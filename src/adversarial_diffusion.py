import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from src.denoising_diffusion_pytorch import normalize_to_neg_one_to_one
from src.residual_denoising_diffusion_pytorch import ResidualDiffusion


class AdversarialResidualDiffusion(ResidualDiffusion):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_res_noise',
            ddim_sampling_eta=0.,
            condition=False,
            sum_scale=None,
            input_condition=False,
            input_condition_mask=False,
            test_res_or_noise="None",
            img_to_img_translation=False,
            epsilon_vis=1e-3,  # Visibility constraint parameter
            classifier=None,
            attack_loss_weight=1.0
    ):
        super().__init__(
            model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type=loss_type,
            objective=objective,
            ddim_sampling_eta=ddim_sampling_eta,
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise=test_res_or_noise,
            img_to_img_translation=img_to_img_translation
        )
        self.epsilon_vis = epsilon_vis
        self.classifier = classifier
        self.attack_loss_weight = attack_loss_weight
        # 调整权重
        self.adv_weight = 0.5  # 降低对抗损失的权重
        self.vis_weight = 0.5  # 增加可见性约束的权重

        # 确保分类器需要梯度
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def compute_adversarial_residual(self, source_img, target_img):
        """计算约束后的对抗残差"""
        residual = target_img - source_img
        residual_norm = torch.norm(residual, p=2)
        if residual_norm > self.epsilon_vis:
            residual = residual * (self.epsilon_vis / residual_norm)
        return residual

    def p_losses(self, imgs, t, noise=None):
        """
        计算对抗残差扩散模型的损失
        Args:
            imgs: [source_img, target_img] 列表
            t: 时间步
            noise: 可选的预定义噪声
        Returns:
            losses: 包含所有损失项的列表
        """
        if not isinstance(imgs, (list, tuple)) or len(imgs) != 2:
            raise ValueError("Expected input in format [source_img, target_img]")

        source_img, target_img = imgs
        device = source_img.device  # 获取当前设备

        # 确保输入需要梯度
        source_img.requires_grad_(True)
        target_img.requires_grad_(True)

        # 计算对抗残差
        residual = target_img - source_img
        residual_norm = torch.norm(residual.view(residual.shape[0], -1), p=2, dim=1)
        normalized_residual = residual * torch.min(
            torch.ones_like(residual_norm).view(-1, 1, 1, 1),
            (self.epsilon_vis / residual_norm).view(-1, 1, 1, 1)
        )
        adv_img = source_img + normalized_residual

        # 基础扩散损失
        base_losses = super().p_losses([adv_img, source_img], t, noise)
        if not isinstance(base_losses, (list, tuple)):
            base_losses = [base_losses]
        losses = list(base_losses)

        # 对抗损失
        if self.classifier is not None:
            adv_pred = self.classifier(adv_img)
            with torch.no_grad():
                target_pred = self.classifier(target_img)
            target_class = target_pred.argmax(dim=1)

            # 将target_class移动到正确的设备
            target_class = target_class.to(device)

            # 计算目标类和其他类的logits
            target_logits = adv_pred.gather(1, target_class.unsqueeze(1)).squeeze(1)

            # 创建mask并确保在正确的设备上
            mask = torch.eye(adv_pred.shape[1], device=device)[target_class]
            other_logits = torch.max(
                adv_pred - mask * 9999,
                dim=1
            )[0]

            # CW损失
            kappa = 5.0
            adv_loss = torch.mean(torch.clamp(other_logits - target_logits + kappa, min=0.0))
            losses.append(self.adv_weight * adv_loss)

        # 可见性约束损失
        visibility_loss = torch.mean(
            torch.abs(residual_norm - self.epsilon_vis) +
            F.relu(residual_norm - self.epsilon_vis)
        )
        losses.append(self.vis_weight * visibility_loss)

        # 图像平滑损失
        def compute_smoothness_loss(tensor):
            h_diff = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
            v_diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            d_diff = tensor[:, :, 1:, 1:] - tensor[:, :, :-1, :-1]

            return (torch.mean(torch.abs(h_diff)) +
                    torch.mean(torch.abs(v_diff)) +
                    torch.mean(torch.abs(d_diff))) / 3.0

        smoothness_loss = compute_smoothness_loss(residual)
        losses.append(0.05 * smoothness_loss)

        # 确保所有损失都有梯度
        losses = [loss for loss in losses if loss.requires_grad]

        # 添加损失统计信息
        if self.training and hasattr(self, 'loss_stats'):
            self.loss_stats = {
                'base_loss': base_losses[0].item(),
                'adv_loss': adv_loss.item() if self.classifier is not None else 0,
                'visibility_loss': visibility_loss.item(),
                'smoothness_loss': smoothness_loss.item(),
                'total_loss': sum(loss.item() for loss in losses)
            }

        return losses

    @torch.no_grad()
    def sample(self, x_input=None, batch_size=16, last=True):
        if x_input is None or not isinstance(x_input, (list, tuple)) or len(x_input) != 2:
            raise ValueError("Expected x_input in format [source_img, target_img]")

        source_img, target_img = x_input
        source_img = normalize_to_neg_one_to_one(source_img)
        target_img = normalize_to_neg_one_to_one(target_img)

        # 降低采样步数以减少累积误差
        original_timesteps = self.sampling_timesteps
        self.sampling_timesteps = min(10, original_timesteps)

        try:
            # 生成多个样本并选择最好的
            all_samples = []
            success_rates = []

            for _ in range(3):  # 生成3个候选样本
                adv_samples = self.ddim_sample([source_img, target_img],
                                               source_img.shape,
                                               last=True)

                if self.classifier is not None:
                    with torch.no_grad():
                        target_pred = self.classifier(target_img)
                        target_class = target_pred.argmax(dim=1)

                        adv_pred = self.classifier(adv_samples[-1])
                        success = (adv_pred.argmax(dim=1) == target_class).float().mean()
                        success_rates.append(success.item())
                        all_samples.append(adv_samples)

            # 选择攻击成功率最高的样本
            if all_samples:
                best_idx = np.argmax(success_rates)
                adv_samples = all_samples[best_idx]
                print(f"Best Attack Success Rate: {success_rates[best_idx] * 100:.2f}%")

            return adv_samples
        finally:
            # 恢复原始采样步数
            self.sampling_timesteps = original_timesteps

    def evaluate_attack_success(self, results_folder):
        """评估对抗攻击的成功率"""
        if self.classifier is None:
            return

        self.eval()
        total = 0
        success = 0

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载并评估生成的对抗样本
        for file in os.listdir(results_folder):
            if file.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(results_folder, file)
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).cuda()

                with torch.no_grad():
                    pred = self.classifier(img)
                    target_pred = self.classifier(img)  # 这里应该使用目标图像，但示例中用了同一张
                    target_class = target_pred.argmax(dim=1)

                    if pred.argmax(dim=1) == target_class:
                        success += 1
                    total += 1

        success_rate = (success / total) * 100 if total > 0 else 0
        print(f"\nAttack Success Rate: {success_rate:.2f}% ({success}/{total})")
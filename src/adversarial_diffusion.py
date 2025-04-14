import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

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
            target_class=None,  # Target class for adversarial attack
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
        self.target_class = target_class
        self.classifier = classifier
        self.attack_loss_weight = attack_loss_weight

    def compute_adversarial_residual(self, source_img, target_img):
        """
        Compute constrained adversarial residual between source and target images
        """
        # Calculate raw residual: I_res_attack = I_target - I_0
        residual = target_img - source_img

        # Constrain residual norm: ||I_res_attack||_2 ≤ epsilon_vis
        residual_norm = torch.norm(residual, p=2)
        if residual_norm > self.epsilon_vis:
            residual = residual * (self.epsilon_vis / residual_norm)

        return residual

    def generate_adversarial_sample(self, source_img, target_img, classifier=None):
        """
        Generate invisible adversarial sample using residual diffusion
        """
        # Compute constrained adversarial residual
        residual = self.compute_adversarial_residual(source_img, target_img)

        # Initialize timesteps for diffusion process
        batch_size = source_img.shape[0]
        t = torch.zeros(batch_size, device=source_img.device).long()

        # Generate adversarial sample using residual diffusion
        with torch.no_grad():
            # Predict noise and start point
            model_pred = self.model_predictions(source_img, source_img + residual, t)
            x_start = model_pred.pred_x_start

            # Apply diffusion process
            adv_img = x_start

            # Ensure sample stays within valid image range [0,1]
            adv_img = torch.clamp(adv_img, 0, 1)

            # Verify adversarial effect if classifier is provided
            if classifier is not None:
                pred = classifier(adv_img)
                if self.target_class is not None:
                    success = (torch.argmax(pred, dim=1) == self.target_class).float().mean()
                    print(f"Attack Success Rate: {success.item() * 100:.2f}%")

        return adv_img, residual

    def p_losses(self, imgs, t, noise=None):
        """
        Modified loss function to incorporate adversarial objectives
        """
        # 获取基础损失
        base_losses = super().p_losses(imgs, t, noise)

        if isinstance(imgs, list) and len(imgs) >= 2:
            source_img = imgs[1]  # 输入图像
            target_img = imgs[0]  # 目标图像

            # 计算对抗残差
            residual = self.compute_adversarial_residual(source_img, target_img)

            # 生成对抗样本
            adv_img = source_img + residual

            # 计算分类器损失
            if self.classifier is not None:
                target_labels = torch.full((adv_img.shape[0],),
                                           self.target_class,
                                           device=adv_img.device)
                classifier_output = self.classifier(adv_img)
                adv_loss = F.cross_entropy(classifier_output, target_labels)

                # 添加对抗损失
                base_losses.append(self.attack_loss_weight * adv_loss)

        return base_losses

    def sample(self, x_input=0, batch_size=16, target_img=None, last=True):
        """
        Modified sampling to support adversarial generation
        """
        if target_img is not None:
            adv_img, residual = self.generate_adversarial_sample(x_input[0], target_img)
            return [adv_img]
        else:
            return super().sample(x_input, batch_size, last)

    def evaluate_attack_success(self, results_folder):
        """
        评估对抗攻击的成功率
        """
        if self.classifier is None:
            return

        self.eval()
        total = 0
        success = 0

        # 加载生成的对抗样本
        for file in os.listdir(results_folder):
            if file.endswith(".png"):
                img_path = os.path.join(results_folder, file)
                img = Image.open(img_path).convert('RGB')
                img = transforms.ToTensor()(img).unsqueeze(0).cuda()

                # 获取分类器预测
                with torch.no_grad():
                    pred = self.classifier(img)
                    pred_class = pred.argmax(dim=1).item()

                    if pred_class == self.target_class:
                        success += 1
                    total += 1

        success_rate = (success / total) * 100 if total > 0 else 0
        print(f"\nAttack Success Rate: {success_rate:.2f}% ({success}/{total})")
# -*- coding: UTF-8 -*- #
"""
@filename:test.py
@author:JongAh
@time:2025-04-15
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm  # 添加 tqdm

class TargetClassifier(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 加载模型参数并去掉 key 前缀 "model."
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # 如有GPU，可移除 map_location
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        self.model.eval()

    def forward(self, x):
        return self.model(x)


def evaluate_attack_success(classifier, adversarial_samples_dir, target_class, image_size=224):
    """
    Evaluate the success rate of adversarial attack.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    total_samples = 0
    successful_attacks = 0

    classifier.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)

    # 获取所有图片文件列表
    img_files = [f for f in os.listdir(adversarial_samples_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg'))]

    # tqdm 添加进度条
    for img_file in tqdm(img_files, desc="Evaluating"):
        img_path = os.path.join(adversarial_samples_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(img_tensor)
            predicted_class = output.argmax(dim=1).item()

            if predicted_class != target_class:
                successful_attacks += 1
            total_samples += 1

    success_rate = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nAttack Success Rate: {success_rate:.2f}% ({successful_attacks}/{total_samples})")
    return success_rate


if __name__ == "__main__":
    model_path = './pre-model/resnet18_cat_dog_best.pth'
    adversarial_samples_dir = './results/adv_test_resnet18_1000'
    target_class = 1

    classifier = TargetClassifier(model_path=model_path)
    evaluate_attack_success(classifier, adversarial_samples_dir, target_class)

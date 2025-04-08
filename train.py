import os
import sys
import torch
from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)


def main():
    # 所有初始化代码必须放在main函数内部
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sys.stdout.flush()

    # 设置随机种子
    set_seed(10)

    # 调试模式配置
    debug = True
    if debug:
        save_and_sample_every = 2
        sampling_timesteps = 10
        sampling_timesteps_original_ddim_ddpm = 10
        train_num_steps = 200
    else:
        save_and_sample_every = 1000
        sampling_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        sampling_timesteps_original_ddim_ddpm = 250
        train_num_steps = 100000

    # 模型配置
    original_ddim_ddpm = False
    condition = not original_ddim_ddpm
    input_condition = False
    input_condition_mask = False

    # 数据路径配置
    if condition:
        folder = [
            "../../database/cat_dog/cat.flist",
            "../../database/cat_dog/dog.flist",
            "../../database/cat_dog/dog_test.flist"
        ]
        img_to_img_translation = True
        train_batch_size = 64
        num_samples = 16
    else:
        folder = 'xxx/CelebA/img_align_celeba'
        train_batch_size = 128
        num_samples = 64

    # 统一参数设置
    sum_scale = 1
    image_size = 64
    num_unet = 2
    objective = 'pred_res_noise'
    test_res_or_noise = "res_noise"

    # 模型初始化
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
        img_to_img_translation=img_to_img_translation
    )

    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l2',
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise=test_res_or_noise,
        img_to_img_translation=img_to_img_translation
    )

    # 强制单进程数据加载
    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
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
        num_unet=num_unet,
        # num_workers=0,  # 关键修改点1：禁用多进程加载
        # persistent_workers=False
    )

    # 训练流程
    trainer.train()

    # 测试流程
    if trainer.accelerator.is_local_main_process:
        trainer.load(trainer.train_num_steps // save_and_sample_every)
        trainer.set_results_folder(f'./results/test_timestep_{sampling_timesteps}')
        trainer.test(last=True)


if __name__ == '__main__':
    # 关键修改点2：设置正确的多进程启动方式
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from torch_utils.gen_utils import w_to_img, make_run_dir

# ----------------------------------------------------------------------------


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps: int = 1000,
        w_avg_samples: int = 10000,
        initial_learning_rate: float = 0.1,
        initial_noise_factor: float = 0.05,
        lr_rampdown_length: float = 0.25,
        lr_rampup_length: float = 0.05,
        noise_ramp_length: float = 0.75,
        regularize_noise_weight: float = 1e5,
        project_in_wplus: bool = False,
        device: torch.device) -> torch.Tensor:
    """Projecting into W+ as opposed as the original repo. Should yield better results in general."""
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    if project_in_wplus:
        w_avg = torch.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]
        w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    else:
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # Setup noise inputs.
    noise_buffs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_buffs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    best_dist = best_loss = np.inf
    best_dist_step = best_loss_step = 0
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        if project_in_wplus:
            ws = w_opt + w_noise
        else:
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_buffs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # See if these are the best loss and dist we've gotten so far
        if dist < best_dist:
            best_dist = dist
            best_dist_step = step + 1
        if loss < best_loss:
            best_loss = loss
            best_loss_step = step + 1
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Print in the same line (avoid cluttering the commandline)
        n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
        print(f'step {step + 1:{n_digits}d}/{num_steps}: dist {dist:7.4f} | loss {float(loss):11.4f} - '
              f'best dist step {best_dist_step:{n_digits}d} | best loss step {best_loss_step:{n_digits}d}', end='\r')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_buffs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    if project_in_wplus:
        return w_out  # [num_steps, L, C]
    return w_out.repeat([1, G.mapping.num_ws, 1])  # [num_steps, 1, C] => [num_steps, L, C]


# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', '-net', 'network_pkl', help='Network pickle filename', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--target', '-t', 'target_fname', help='Target image file to project to', type=click.Path(exists=True, dir_okay=False), required=True, metavar='FILE')
@click.option('--num-steps', '-nsteps', help='Number of optimization steps', type=click.IntRange(min=0), default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', '-video', help='Save an mp4 video of optimization progress', is_flag=True)
@click.option('--fps', help='FPS for the mp4 video of optimization progress', type=int, default=30, show_default=True)
@click.option('--project-in-wplus', '-wplus', help='Project in the W+ latent space', is_flag=True)
@click.option('--save-every-step', '-saveall', help='Save every step taken in the projection (npy and image).', is_flag=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
def run_projection(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        save_video: bool,
        seed: int,
        num_steps: int,
        fps: int,
        project_in_wplus: bool,
        save_every_step: bool
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --target=~/mytargetimg.png --project-in-wplus --save-video --num-steps=1000 --save-every-step \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
        num_steps=num_steps,
        project_in_wplus=project_in_wplus,
        device=device
    )
    print(f'\nElapsed: {(perf_counter()-start_time):.1f} s')

    # Make the run dir automatically
    desc = 'projection-wplus' if project_in_wplus else 'projection'
    run_dir = make_run_dir(outdir, desc)

    # Render debug output: optional video and projected image and W vector.
    result_name = os.path.join(run_dir, 'proj')
    npy_name = os.path.join(run_dir, 'projected')
    if project_in_wplus:
        result_name, npy_name = f'{result_name}_wplus', f'{npy_name}_wplus'

    if save_video:
        video = imageio.get_writer(f'{result_name}.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{result_name}.mp4"')
        for projected_w in projected_w_steps:
            synth_image = w_to_img(G, dlatent=projected_w, noise_mode='const')
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))  # left side target, right projection
        video.close()

    # Save the target image
    target_pil.save(f'{run_dir}/target.jpg')
    print('Saving projection results...')
    if save_every_step:
        # Save every projected frame and W vector.
        n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
        for step in range(num_steps):
            w = projected_w_steps[step]
            synth_image = w_to_img(G, dlatent=w, noise_mode='const')
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{result_name}_step{step:0{n_digits}d}.jpg')
            np.save(f'{npy_name}_step{step:0{n_digits}d}.npy', w.unsqueeze(0).cpu().numpy())
    else:
        # Save only the final projected frame and W vector.
        projected_w = projected_w_steps[-1]
        synth_image = w_to_img(G, dlatent=projected_w, noise_mode='const')
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{result_name}_final.jpg')
        np.save(f'{npy_name}_final.npy', projected_w.unsqueeze(0).cpu().numpy())

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------

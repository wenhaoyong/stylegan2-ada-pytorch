# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from typing import List, Optional, Union
from locale import atof
import click

import dnnlib
from torch_utils.gen_utils import num_range, parse_fps, compress_video, double_slowdown, \
    make_run_dir, z_to_img, w_to_img, get_w_from_file, create_image_grid

import scipy
import numpy as np
import PIL.Image
import torch

import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor


# ----------------------------------------------------------------------------

# We group the different types of generation (images, grid, video, wacky stuff) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='images')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file; can be either .npy or .npz files', type=click.Path(exists=True, dir_okay=False), metavar='FILE')
@click.option('--save-grid', help='Use flag to save image grid', is_flag=True, show_default=True)
@click.option('--grid-width', '-gw', type=int, help='Grid width (number of columns)', default=None)
@click.option('--grid-height', '-gh', type=int, help='Grid height (number of rows)', default=None)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Description name for the directory path to save results', default='generate-images', show_default=True)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        save_grid: bool,
        grid_width: int,
        grid_height: int,
        outdir: str,
        desc: str,
        class_idx: Optional[int],
        projected_w: Optional[Union[str, os.PathLike]]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py generate-images --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py generate-images --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py generate-images --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py generate-images --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Create the run dir with the given name description
    run_dir = make_run_dir(outdir, desc)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = get_w_from_file(projected_w)
        ws = torch.tensor(ws, device=device)
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        n_digits = int(np.log10(len(ws))) + 1  # number of digits for naming the .jpg images
        for idx, w in enumerate(ws):
            img = w_to_img(G, w, noise_mode)
            PIL.Image.fromarray(img, 'RGB').save(f'{run_dir}/proj{idx:0{n_digits}d}.jpg')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    images = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = z_to_img(G, z, label, truncation_psi, noise_mode)[0]
        if save_grid:
            images.append(img)
        PIL.Image.fromarray(img, 'RGB').save(f'{run_dir}/seed{seed:04d}.jpg')

    if save_grid:
        print('Saving image grid...')
        # We let the function infer the shape of the grid
        if (grid_width, grid_height) == (None, None):
            PIL.Image.fromarray(create_image_grid(np.array(images)), 'RGB').save(f'{run_dir}/grid.png')
        # The user tells the specific shape of the grid, but one value may be None
        elif None in (grid_width, grid_height):
            PIL.Image.fromarray(create_image_grid(np.array(images), (grid_width, grid_height)),
                                'RGB').save(f'{run_dir}/grid.png')


# ----------------------------------------------------------------------------


def _parse_slowdown(slowdown: Union[str, int]) -> int:
    """Function to parse the 'slowdown' parameter by the user. Will approximate to the nearest power of 2."""
    # TODO: slowdown should be any int
    if not isinstance(slowdown, int):
        slowdown = atof(slowdown)
    assert slowdown > 0
    # Let's approximate slowdown to the closest power of 2 (nothing happens if it's already a power of 2)
    slowdown = 2**int(np.rint(np.log2(slowdown)))
    return max(slowdown, 1)  # Guard against 0.5, 0.25, ... cases


@main.command(name='random-video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--grid-width', '-gw', type=int, help='Video grid width / number of columns', default=None, show_default=True)
@click.option('--grid-height', '-gh', type=int, help='Video grid height / number of rows', default=None, show_default=True)
@click.option('--slowdown', type=_parse_slowdown, help='Slow down the video by this amount; will be approximated to the nearest power of 2', default='1', show_default=True)
@click.option('--duration-sec', '-sec', type=float, help='Duration length of the video', default=30.0, show_default=True)
@click.option('--fps', type=parse_fps, help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Description name for the directory path to save results', default='random-video', show_default=True)
def random_interpolation_video(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        seeds: Optional[List[int]],
        truncation_psi: float,
        class_idx: Optional[int],
        noise_mode: str,
        grid_width: int,
        grid_height: int,
        slowdown: int,
        duration_sec: float,
        fps: int,
        outdir: Union[str, os.PathLike],
        desc: str,
        compress: bool,
        smoothing_sec: Optional[float] = 3.0  # for Gaussian blur; won't be a parameter, change at own risk
):
    """
    Generate a random interpolation video using a pretrained network.

    Examples:

    \b
    # Generate a 30-second long, untruncated MetFaces video at 30 FPS (3 rows and 2 columns; horizontal):
    python generate.py random-video --seeds=0-5 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate a 60-second long, truncated 1x2 MetFaces video at 60 FPS (2 rows and 1 column; vertical):
    python generate.py random-video --trunc=0.7 --seeds=10,20 --grid-width=1 --grid-height=2 \\
        --fps=60 -sec=60 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    """
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Create the run dir with the given name description; add slowdown if different than the default (1)
    desc = f'{desc}-{slowdown}xslowdown' if slowdown != 1 else desc
    run_dir = make_run_dir(outdir, desc)

    # Number of frames in the video and its total duration in seconds
    num_frames = int(np.rint(duration_sec * fps))
    total_duration = duration_sec * slowdown

    print('Generating latent vectors...')
    # TODO: let another helper function handle each case, we will use it for the grid
    # If there's more than one seed provided and the shape isn't specified by the user
    if (grid_width is None and grid_height is None) and len(seeds) >= 1:
        # TODO: this can be done by another function
        # Number of images in the grid video according to the seeds provided
        num_seeds = len(seeds)
        # Get the grid width and height according to num, giving priority to the number of columns
        grid_width = max(int(np.ceil(np.sqrt(num_seeds))), 1)
        grid_height = max((num_seeds - 1) // grid_width + 1, 1)
        grid_size = (grid_width, grid_height)
        shape = [num_frames, G.z_dim]  # This is per seed
        # Get the z latents
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)

    # If only one seed is provided, but the specific grid shape is specified:
    elif None not in (grid_width, grid_height) and len(seeds) == 1:
        grid_size = (grid_width, grid_height)
        shape = [num_frames, np.prod(grid_size), G.z_dim]
        # Since we have one seed, we use it to generate all latents
        all_latents = np.random.RandomState(*seeds).randn(*shape).astype(np.float32)

    # If more than one seed is provided, but the user also specifies the grid shape:
    elif None not in (grid_width, grid_height) and len(seeds) >= 1:
        # Case is similar to the first one
        num_seeds = len(seeds)
        grid_size = (grid_width, grid_height)
        available_slots = np.prod(grid_size)
        if available_slots < num_seeds:
            diff = num_seeds - available_slots
            click.secho(f'More seeds were provided ({num_seeds}) than available spaces in the grid ({available_slots})',
                        fg='red')
            click.secho(f'Removing the last {diff} seeds: {seeds[-diff:]}', fg='blue')
            seeds = seeds[:available_slots]
        shape = [num_frames, G.z_dim]
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)

    else:
        ctx.fail('Error: wrong combination of arguments! Please provide either a list of seeds, one seed and the grid '
                 'width and height, or more than one seed and the grid width and height')

    # Let's smooth out the random latents so that now they form a loop (and are correctly generated in a 512-dim space)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[smoothing_sec * fps, 0, 0], mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Name of the video
    mp4_name = f'{grid_width}x{grid_height}-slerp-{slowdown}xslowdown'

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    while slowdown > 1:
        all_latents, duration_sec, num_frames = double_slowdown(latents=all_latents,
                                                                duration=duration_sec,
                                                                frames=num_frames)
        slowdown //= 2

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        latents = torch.from_numpy(all_latents[frame_idx]).to(device)
        # Get the images with the labels
        images = z_to_img(G, latents, label, truncation_psi, noise_mode)
        # Generate the grid for this timestamp
        grid = create_image_grid(images, grid_size)
        # Grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(total_duration)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution)
    if compress:
        compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------

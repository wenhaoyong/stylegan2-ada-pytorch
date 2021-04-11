# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
from collections import OrderedDict
from typing import List, Union, Optional
import click

import dnnlib
from torch_utils.gen_utils import parse_fps, compress_video, make_run_dir, w_to_img

import numpy as np
import PIL.Image
import scipy
import torch

import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor

# ----------------------------------------------------------------------------


def num_range(s: str) -> List[int]:
    """
    Extended helper function from the original (original is contained here).
    Accept a comma separated list of numbers 'a,b,c', a range 'a-c', or a combination
    of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...', and return as a list of ints.

    Also accepted is the styles defined in the StyleGAN paper: 'coarse', 'middle', and
    'fine' styles, if the user wishes to use some pre-determined ones.
    """
    # coarse, middle, and fine style layers as defined in the StyleGAN paper
    if s == 'coarse':
        return list(range(0, 4))
    elif s == 'middle':
        return list(range(4, 8))
    elif s == 'fine':
        return list(range(8, 18))
    # Else, the user has defined specific styles (or seeds) to use
    else:
        str_list = s.split(',')
        nums = []
        for el in str_list:
            if '-' in el:
                # Get the lower and upper bounds of the range
                a, b = el.split('-')
                # Sanity check 0: only ints please
                try:
                    lower, upper = int(a), int(b)
                except ValueError:
                    print(f'One of the elements in "{s}" is not an int!')
                    raise
                # Sanity check 1: accept 'a-b' or 'b-a', with a<=b
                if lower <= upper:
                    r = [n for n in range(lower, upper + 1)]
                else:
                    r = [n for n in range(upper, lower + 1)]
                # We will extend nums (r is also a list)
                nums.extend(r)
            else:
                # It's a single number, so just append it
                nums.append(int(el))
        # Sanity check 2: delete repeating numbers, but keep order given by user
        nums = list(OrderedDict.fromkeys(nums))
        return nums


# ----------------------------------------------------------------------------


# We group the different types of style-mixing (grid and video) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='grid')
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--row-seeds', '-rows', 'row_seeds', type=num_range, help='Random seeds to use for image rows', required=True)
@click.option('--col-seeds', '-cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layers to use; can pass "coarse", "middle", "fine", or a list or range of ints', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Description name for the directory path to save results', default='stylemix-grid', show_default=True)
def generate_style_mix(
        network_pkl: str,
        row_seeds: List[int],
        col_seeds: List[int],
        col_styles: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        desc: str
):
    """Generate style-mixing images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py grid --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Sanity check: loaded model and selected styles must be compatible
    max_style = 2 * int(np.log2(G.img_resolution)) - 3
    if max(col_styles) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_styles[:] = [style for style in col_styles if style <= max_style]

    # Create the run dir with the given name description
    run_dir = make_run_dir(outdir, desc)

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = w_to_img(G, all_w, noise_mode)
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = w_to_img(G, w, noise_mode)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{run_dir}/{row_seed}-{col_seed}.jpg')

    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(os.path.join(run_dir, 'grid.jpg'))


# ----------------------------------------------------------------------------


@main.command(name='video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--row-seed', '-row', 'row_seed', type=int, help='Random seed to use for video row', required=True)
@click.option('--col-seeds', '-cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layers to use; can pass "coarse", "middle", "fine", or a list or range of ints', default='0-6', show_default=True)
@click.option('--only-stylemix', is_flag=True, help='Add flag to only show the style-mixed images in the video')
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file via ffmpeg-python (same resolution, lower file size)')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), help='Noise mode', default='const', show_default=True)
@click.option('--duration-sec', type=float, help='Duration of the video in seconds', default=30, show_default=True)
@click.option('--fps', type=parse_fps, help='Video FPS.', default=30, show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Description name for the directory path to save results', default='stylemix-video', show_default=True)
def random_stylemix_video(
        ctx: click.Context,
        network_pkl: str,
        row_seed: int,
        col_seeds: List[int],
        col_styles: List[int],
        only_stylemix: bool,
        compress: bool,
        truncation_psi: float,
        noise_mode: str,
        fps: int,
        duration_sec: float,
        outdir: Union[str, os.PathLike],
        desc: str,
        smoothing_sec: Optional[float] = 3.0  # for Gaussian blur; won't be a parameter, change at own risk
):
    """Generate random style-mixing video using pretrained network pickle.

        Examples:

        \b
        python style_mixing.py video --row=85 --cols=55,821,1789 --fps=60 \\
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

        \b
        python style_mixing.py video --row=0 --cols=7-10 --styles=fine --duration-sec=60 \\
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    # Calculate number of frames
    num_frames = int(np.rint(duration_sec * fps))
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Get the average dlatent
    w_avg = G.mapping.w_avg

    # Sanity check: loaded model and selected styles must be compatible
    max_style = 2 * int(np.log2(G.img_resolution)) - 3
    if max(col_styles) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_styles[:] = [style for style in col_styles if style <= max_style]

    # Create the run dir with the given name description
    run_dir = make_run_dir(outdir, desc)

    # First column (video) latents
    print('Generating source W vectors...')
    src_shape = [num_frames, G.z_dim]
    src_z = np.random.RandomState(row_seed).randn(*src_shape).astype(np.float32)
    src_z = scipy.ndimage.gaussian_filter(src_z, sigma=[smoothing_sec * fps, 0], mode='wrap')  # wrap to form a loop
    src_z /= np.sqrt(np.mean(np.square(src_z)))  # normalize

    # Map to W and do truncation trick
    src_w = G.mapping(torch.from_numpy(src_z).to(device), None)
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # First row (images) latents
    dst_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in col_seeds])
    dst_w = G.mapping(torch.from_numpy(dst_z).to(device), None)
    dst_w = w_avg + (dst_w - w_avg) * truncation_psi

    # Width and height of the generated image
    W = G.img_resolution
    H = G.img_resolution

    # Video name
    mp4_name = f'{len(col_seeds)}x1'
    # Add to the name the styles (from the StyleGAN paper) if they are being used
    if list(range(0, 4)) == col_styles:
        mp4_name = f'{mp4_name}-coarse_styles'
    elif list(range(4, 8)) == col_styles:
        mp4_name = f'{mp4_name}-middle_styles'
    elif list(range(8, max_style + 1)) == col_styles:
        mp4_name = f'{mp4_name}-fine_styles'

    # If user wishes to only show the style-transferred images (nice for 1x1 case)
    if only_stylemix:
        print('Generating style-mixing video (with only the style-transferred images)...')
        # We generate a canvas where we will paste all the generated images
        canvas = PIL.Image.new('RGB', (W * len(col_seeds), H * len([row_seed])), 'black')

        def make_frame(t):
            # Get the frame number according to time t
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            # For each of the column images
            for col, _ in enumerate(dst_w):
                # Select the pertinent latent w column
                w_col = dst_w[col].unsqueeze(0)  # [18, 512] -> [1, 18, 512]
                # Replace the values defined by col_styles
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate the style-mixed images
                col_images = w_to_img(G, w_col, noise_mode)
                # Paste them in their respective spot in the grid
                for row, image in enumerate(list(col_images)):
                    canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * H, row * W))

            return np.array(canvas)

        mp4_name = f'{mp4_name}-only-stylemix'
    else:
        print('Generating style-mixing video...')
        # Generate an empty canvas where we will paste all the generated images
        canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len([row_seed]) + 1)), 'black')

        # Generate all destination images (first row; static images)
        dst_images = w_to_img(G, dst_w, noise_mode)
        # Paste them in the canvas
        for col, dst_image in enumerate(list(dst_images)):
            canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), ((col + 1) * H, 0))

        def make_frame(t):
            # Get the frame number according to time t
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            # Get the image at this frame (first column; video)
            src_image = w_to_img(G, src_w[frame_idx], noise_mode)[0]
            # Paste it to the lower left
            canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, H))

            # For each of the column images (destination images)
            for col, _ in enumerate(list(dst_images)):
                # Select pertinent latent w column
                w_col = dst_w[col].unsqueeze(0)  # [18, 512] -> [1, 18, 512]
                # Replace the values defined by col_styles
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate these style-mixed images
                col_images = w_to_img(G, w_col, noise_mode)
                # Paste them in their respective spot in the grid
                for row, image in enumerate(list(col_images)):
                    canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * H, (row + 1) * W))

            return np.array(canvas)

        mp4_name = f'{mp4_name}-style-mixing'

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(duration_sec)

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

import os
from typing import List, Union
import click

import dnnlib
import legacy

import torch

import numpy as np
from torch_utils.gen_utils import parse_fps, compress_video, make_run_dir, w_to_img, create_image_grid, interpolate

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor


# ----------------------------------------------------------------------------


def _parse_seeds(s: str) -> List[int]:
    """
    Helper function for parsing seeds. With a, b, c,... as ints, then s can be either:
        * a comma-separated list of numbers: 'a,b,c,d,...'
        * a range-like of numbers: 'a-d'
        * a combination of both: 'a,b,c-d,a-e,f...'

    The returned list will be the numbers in this range, in order as the user entered
    them, without deleting repeated values.
    """
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if '-' in el:
            # Get the lower and upper bound of the range
            a, b = el.split('-')
            # Sanity check 0: only ints please
            try:
                lower, upper = int(a), int(b)
            except ValueError:
                print(f'Upper and lower bounds of "{el}" in "{s}" should be ints!')
                raise
            # Sanity check 1: accept ranges 'a-b' or 'b-a', with a<=b
            if lower <= upper:
                r = [n for n in range(lower, upper + 1)]
            else:
                r = [n for n in range(upper, lower + 1)]
            # Extend nums as r is also a list
            nums.extend(r)
        else:
            # It's a single number, so append it (if it's an int)
            try:
                nums.append(int(el))
            except ValueError:
                print(f'"{el}" in "{s}" is not an int!')
                raise
    return nums


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--network', '-net', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', '-s', type=_parse_seeds, help='List of seeds to visit in order ("a,b,c", "a-b", "a,b-c,d,e-f,a", ...', required=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--seed-sec', '-sec', type=float, help='Number of seconds between each seed transition', default=5.0, show_default=True)
@click.option('--interp-type', '-interp', type=click.Choice(['linear', 'spherical']), help='Type of interpolation in Z or W', default='spherical', show_default=True)
@click.option('--interp-in-z', is_flag=True, help='Add flag to interpolate in Z instead of in W')
@click.option('--smooth', is_flag=True, help='Add flag to smooth the transition between the latent vectors')
@click.option('--fps', type=parse_fps, help='Video FPS.', default=30, show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Additional description for the directory name where', default='', show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file via ffmpeg-python (same resolution, lower file size)')
def sightseeding(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        seeds: List[int],
        class_idx: int,
        truncation_psi: float,
        noise_mode: str,
        seed_sec: float,
        interp_type: str,
        interp_in_z: bool,
        smooth: bool,
        fps: int,
        outdir: Union[str, os.PathLike],
        desc: str,
        compress: bool,
):
    """
    Examples:

    # Will go from seeds 0 through 5, coming to the starting one in the end; the transition between each pair of seeds
    taking 7.5 seconds, spherically (and smoothly) interpolating in W, compressing the final video with ffmpeg-python

    python sightseeding.py --seeds=0-5,0 --seed-sec=7.5 --smooth --compress \
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl
    """
    # Sanity check:
    if len(seeds) < 2:
        ctx.fail('Please enter more than one seed to interpolate between!')

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Get the average dlatent
    w_avg = G.mapping.w_avg

    # Create the run dir with the given name description
    desc = f'{desc}-sightseeding' if len(desc) != 0 else 'sightseeding'
    desc = f'{desc}-{interp_type}-smooth' if smooth else f'{desc}-{interp_type}'
    desc = f'{desc}-in-Z' if interp_in_z else f'{desc}-in-W'
    run_dir = make_run_dir(outdir, desc)

    # Number of steps to take between each latent vector
    n_steps = int(np.rint(seed_sec * fps))
    # Total number of frames
    num_frames = int(n_steps * (len(seeds) - 1))
    # Total video length in seconds
    duration_sec = num_frames / fps

    # TODO: use labels (the following will be ignored for now)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate the random vectors from each seed
    print('Generating Z vectors...')
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim).astype(np.float32) for seed in seeds])
    # If user wants to interpolate in Z
    if interp_in_z:
        print(f'Interpolating in Z...(interpolation type: {interp_type})')
        src_z = np.empty([0] + list(all_z.shape[1:]), dtype=np.float32)
        for i in range(len(all_z) - 1):
            # We interpolate between each pair of latents
            interp = interpolate(all_z[i], all_z[i + 1], n_steps, interp_type, smooth)
            # Append it to our source
            src_z = np.append(src_z, interp, axis=0)
        # Convert to dlatent vectors
        print('Generating W vectors...')
        src_w = G.mapping(torch.from_numpy(src_z).to(device), None)

    # Otherwise, interpolation is done in W
    else:
        print(f'Interpolating in W... (interpolation type: {interp_type})')
        print('Generating W vectors...')
        all_w = G.mapping(torch.from_numpy(all_z).to(device), None).cpu()
        src_w = np.empty([0] + list(all_w.shape[1:]), dtype=np.float32)
        for i in range(len(all_w) - 1):
            # We interpolate between each pair of dlatents
            interp = interpolate(all_w[i], all_w[i + 1], n_steps, interp_type, smooth)
            # Append it to our source
            src_w = np.append(src_w, interp, axis=0)
        src_w = torch.from_numpy(src_w).to(device)

    # Do the truncation trick
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # Auxiliary function for moviepy
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        w = src_w[frame_idx].unsqueeze(0)  # [18, 512] -> [1, 18, 512]
        image = w_to_img(G, w, noise_mode)
        # Generate the grid for this timestamp
        grid = create_image_grid(image, (1, 1))
        # grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using make_frame
    print('Generating sightseeding video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(duration_sec)
    mp4_name = '-'.join(map(str, seeds))  # Make it clear by the file name what is the path taken
    mp4_name = f'{mp4_name}-sightseeding' if len(mp4_name) < 50 else 'sightseeding'  # arbitrary rule of mine

    # Set the video parameters (change if you like)
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution)
    if compress:
        compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    sightseeding()

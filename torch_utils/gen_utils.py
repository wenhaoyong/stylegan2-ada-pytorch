import os
import re
from typing import List, Tuple, Union
from collections import OrderedDict
from locale import atof

import click
import numpy as np
import torch


# ----------------------------------------------------------------------------


def parse_fps(fps: Union[str, int]) -> int:
    """Return FPS for the video; at worst, video will be 1 FPS, but no lower."""
    if isinstance(fps, int):
        return max(fps, 1)
    return max(int(atof(fps)), 1)


def num_range(s: str) -> List[int]:
    """
    Extended helper function from the original (original is contained here).
    Accept a comma separated list of numbers 'a,b,c', a range 'a-c', or a combination
    of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...', and return as a list of ints.
    """
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


def compress_video(
        original_video: Union[str, os.PathLike],
        original_video_name: Union[str, os.PathLike],
        outdir: Union[str, os.PathLike],
        ctx: click.Context) -> None:
    """ Helper function to compress the original_video using ffmpeg-python. moviepy creates huge videos, so use
        ffmpeg to 'compress' it (won't be perfect, 'compression' will depend on the video dimensions). ffmpeg
        can also be used to e.g. resize the video, make a GIF, save all frames in the video to the outdir, etc.
    """
    try:
        import ffmpeg
    except (ModuleNotFoundError, ImportError):
        ctx.fail('Missing ffmpeg! Install it via "pip install ffmpeg-python"')

    print('Compressing the video...')
    resized_video_name = os.path.join(outdir, f'{original_video_name}-compressed.mp4')
    ffmpeg.input(original_video).output(resized_video_name).run(capture_stdout=True, capture_stderr=True)
    print('Success!')


# ----------------------------------------------------------------------------


def interpolation_checks(
        t: Union[float, np.ndarray],
        v0: np.ndarray,
        v1: np.ndarray) -> Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:
    """Tests for the interpolation functions"""
    # Make sure 0.0<=t<=1.0
    assert np.min(t) >= 0.0 and np.max(t) <= 1.0
    # Guard against v0 and v1 not being NumPy arrays
    if not isinstance(v0, np.ndarray) or not isinstance(v1, np.ndarray):
        v0 = np.array(v0)
        v1 = np.array(v1)
    assert v0.shape == v1.shape, f'Incompatible shapes! v0: {v0.shape}, v1: {v1.shape}'
    return t, v0, v1


def lerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray]) -> np.ndarray:
    """
    Linear interpolation between v0 (starting) and v1 (final) vectors; for optimal results,
    use t as an np.ndarray to return all results at once via broadcasting
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    v2 = (1.0 - t) * v0 + t * v1
    return v2


def slerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        dot_threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical linear interpolation between v0 (starting) and v1 (final) vectors; for optimal
    results, use t as an np.ndarray to return all results at once via broadcasting.

    dot_threshold is the threshold for considering if the two vectors are collinear (not recommended to alter).

    Adapted from the Python code at: https://en.wikipedia.org/wiki/Slerp
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    # Copy vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't always use np.dot, so we use the definition)
    dot = np.sum(v0 * v1)
    # If it's ~1, vectors are ~colineal, so use lerp
    if np.abs(dot) > dot_threshold:
        return lerp(t, v0, v1)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Divide the angle into t steps
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def interpolate(
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        n_steps: int,
        interp_type: str = 'spherical',
        smooth: bool = False) -> np.ndarray:
    """
    Interpolation function between two vectors, v0 and v1. We will either do a 'linear' or 'spherical' interpolation,
    taking n_steps. The steps can be 'smooth'-ed out, so that the transition between vectors isn't too drastic.
    """
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False)
    # TODO: have a dictionary with easing functions that contains my 'smooth' one (might be useful for someone else)
    if smooth:
        # Smooth out the interpolation with a polynomial of order 3 (cubic function f)
        # Constructed f by setting f'(0) = f'(1) = 0, and f(0) = 0, f(1) = 1 => f(t) = -2t^3+3t^2 = t^2 (3-2t)
        t_array = t_array ** 2 * (3 - 2 * t_array)  # One line thanks to NumPy arrays
    # TODO: this might be possible to optimize by using the fact they're numpy arrays, but haven't found a nice way yet
    funcs_dict = {'linear': lerp, 'spherical': slerp}
    vectors = np.array([funcs_dict[interp_type](t, v0, v1) for t in t_array], dtype=np.float32)
    return vectors


# ----------------------------------------------------------------------------


def double_slowdown(latents: np.ndarray, duration: float, frames: int) -> Tuple[np.ndarray, float, int]:
    """
    Auxiliary function to slow down the video by 2x. We return the new latents, duration, and frames of the video
    """
    # Make an empty latent vector with double the amount of frames, but keep the others the same
    z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)
    # In the even frames, populate it with the latents
    for i in range(len(latents)):
        z[2 * i] = latents[i]
    # Interpolate in the odd frames
    for i in range(1, len(z), 2):
        # slerp between (t=0.5) even frames; for the last frame, we loop to the first one
        z[i] = slerp(0.5, z[i - 1], z[i + 1]) if i != len(z) - 1 else slerp(0.5, z[0], z[i - 1])
    # TODO: we could change this to any slowdown: slerp(1/slowdown, ...), and we return z, slowdown * duration, ...
    # Return the new latents, and the respective new duration and number of frames
    return z, 2 * duration, 2 * frames


# ----------------------------------------------------------------------------


def w_to_img(G, dlatent: torch.Tensor, noise_mode: str = 'const') -> np.ndarray:
    synth_image = G.synthesis(dlatent.unsqueeze(0), noise_mode=noise_mode)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return synth_image


# ----------------------------------------------------------------------------


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir

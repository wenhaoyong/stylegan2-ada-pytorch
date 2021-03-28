import os
from typing import List, Union
from collections import OrderedDict
from locale import atof

import click
import numpy as np


# ----------------------------------------------------------------------------


def parse_fps(fps: str) -> int:
    """Return FPS for the video; at worst, video will be 1 FPS, but no lower."""
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


def lerp():
    pass


def slerp():
    pass


def double_slowdown(latents: np.ndarray, duration: float, frames: int):
    """Auxiliary function to slow down the video by 2x. We return the latents, duration, and frames of the video
    """
    # Make an empty latent vector with double the amount of frames
    z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)
    # In the even frames, populate it with the latents
    for i in range(len(latents)):
        z[2 * i] = latents[i]
    # Interpolate in the odd frames
    for i in range(1, len(z), 2):
        if i != len(z) - 1:
            z[i] = (z[i - 1] + z[i + 1]) / 2  # TODO: slerp here
        # For the last frame, we loop to the first one
        else:
            z[i] = (z[0] + z[i - 1]) / 2
    # We also need to double the duration_sec and num_frames
    duration *= 2
    frames *= 2
    # Return the new latents, and the new duration and frames
    return z, duration, frames

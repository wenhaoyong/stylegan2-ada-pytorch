import os
from typing import Union
import click
import numpy as np


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

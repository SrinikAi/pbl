"""General utility functionality.

Working directory
-----------------

This module provides a `hiwi.WorkingDirectory` instance as singleton that
is used throughout the package to write/cache intermediate values and debug
information.
"""


import logging
import numpy as np
import SimpleITK as sitk

from collections import defaultdict
from hiwi import WorkingDirectory, Image, find_anatomical_orientation
from matplotlib.colors import to_rgb
from typing import List, Optional, Tuple


log = logging.getLogger(__name__)


working_dir = WorkingDirectory()
"""The singleton instance of `WorkingDirectory` used in this project."""


def find_image_mode(images: List[Image]) \
        -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Finds the dominant image orientation and spacing.

    Args:
        images: The images to find the mode for.

    Returns: A tuple consisting of the orientation and spacing.
    """
    orientations = defaultdict(lambda: 0)
    spacings = []

    for image in images:
        image_itk = sitk.ReadImage(str(image.path))

        if image_itk.GetDimension() == 3:
            orientation = find_anatomical_orientation(image_itk)
            orientations[orientation] += 1

        if image.spacing is not None:
            spacings.append(image.spacing[::-1])

    # defaults
    try:
        orientation = next(iter(orientations.keys()))
    except StopIteration:
        orientation = None
    try:
        spacing = next(iter(spacings))
    except StopIteration:
        spacing = None

    if len(orientations) > 1:
        orientation = sorted(orientations.items(), key=lambda x: -x[1])[0]
        log.warning('Images have more than one orientation (%s), using the '
                    'most dominant one %s',
                    ', '.join('%s: %i' % p for p in orientations.items()),
                    orientation)
        assert False, 'implement the conversion in training properly'

    if len(spacings) > 1 and not np.allclose(spacings[0], spacings[1:]):
        spacing = np.median(spacings, axis=0)
        log.warning('Images have more than one spacing (min: %s, max: %s), '
                    'using the median spacing %s',
                    np.amin(spacings, axis=0), np.amax(spacings, axis=0),
                    spacing)
        assert False, 'implement the conversion in training properly'

    return orientation, spacing


def format_time(seconds):
    """Formats the given `seconds` as a H:MM:SS string."""
    hours, remainder = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{:02}:{:02}'.format(int(hours), int(minutes),
                                   int(seconds))


def complement_color(c):
    """Creates the complement color of `c`, returned as ` (r, g, b)`."""
    r, g, b = to_rgb(c)
    k = _hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))


def _hilo(a, b, c):
    if c < b:
        b, c = c, b
    if b < a:
        a, b = b, a
    if c < b:
        b, c = c, b
    return a + c

"""Various **potential functions** including localizers.

This module provides the abstract `Potential` (`LearnablePotentialMixin`)
type as well as concrete implementations to be used in a `Graph`.

We provide three abstract classes to inherit from in order to define a new
potential:

- `UnaryPotential`
- `BinaryPotential`
- `TernaryPotential`

If your potential supports learning of parameters, it should implement
the `LearnablePotentialMixin` as well.

Localizer potentials
--------------------

There exist two kinds of localizer potentials, providing candidate positions
as well as unary energies.

**`RteLocalizer`**: A simple localizer based on regression tree ensembles. It
uses the implementation from the `rfl` package and is merely a simple wrapper.

**`MultiPartCNN`**: A multi-part localizer (not a potential though, these are
accessable via the `potentials` attribute) that uses a simple convolutional
neural network. These potentials are learnable.
"""


import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import rfl
import tensorflow as tf
import time

from abc import ABC, ABCMeta, abstractmethod
from collections import deque
from hiwi import Image, ImageList
from hiwi import PatchExtractor, batchize, place_gaussian
from hiwi import transform_elastically, LocalMaxLocator
from humanfriendly import format_size, format_timespan
from itertools import chain
from logging import getLogger
from multiprocessing import Pool as ProcessPool
from numbers import Real
from operator import methodcaller
from pathlib import Path
from scipy import stats
from scipy.spatial import distance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from typing import Any, Dict, List, Optional, Tuple, Union

from .evaluation import Criterion, plot_results
from .utils import working_dir


log = getLogger(__name__)


class Potential(ABC):
    """A potential in our `Graph` that is able to compute energies.

    This is the base class and provides entry points for you (by overriding) to
    optionally `train()` the potential, provide some candidates with
    `propose_candidates()` (if it can) and (mandatory) `compute()` some
    energies.

    Note that this is not a potential in the classical definition, but rather
    a broader definition that catches all the semantics we need.

    Args:
        parts: The string-based parts this potential depends on, i.e., its
            `compute` method gets passed coordinates for these parts only.
        weight: The weight associated with this potential.
        unknown_energy: An optional energy that can be taken in case of
            `unknown`, i.e., missing parts etc.
    """

    def __init__(self, parts: List[str], weight: float = 1.0,
                 unknown_energy: Optional[float] = None) \
            -> None:
        assert len(parts) == self.arity()
        assert weight >= 0

        self.parts = parts
        """A list of parts this potential depends on."""

        self.weight = weight
        """The weight of this potential."""

        self.unknown_energy = unknown_energy
        """The energy that should be used when we can't make an informed
        decision.
        """

    def name(self) -> str:
        """Returns the name of the potential."""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def arity() -> int:
        """Returns the arity of the potential."""
        raise NotImplementedError

    def train(self, train_images: List[Image]) -> None:
        """Trains the potential.

        Some potentials have to be trained, e.g., train a random forest,
        estimate some statistics, etc. This method is called for this.

        Override this method when your potential needs training. The default
        implementation does nothing.

        Args:
            train_images: The training images.
        """
        pass

    def propose_candidates(self, image: Image) -> Dict[str, np.ndarray]:
        """Generates like candidate positions for zero or more parts.

        Some potentials are able to generate a set of positions that are
        likely for certain parts. Consider for instance a potential based on a
        ML-localizer. Exploiting this information allows to greatly reduce
        the search space during CRF inference from the image domain to a set of
        few likely positions.

        Override this method when your potential supports generating likely
        positions. The default implementation returns an empty `dict()`.

        Args:
            image: The image we are currently looking at.

        Returns:
            A mapping (`dict`) from part to a list of candidate positions.
        """
        return dict()

    @abstractmethod
    def compute(self, image: Image, positions: List[np.ndarray],
                **kwargs) -> Union[float, np.ndarray]:
        """Computes the energy for the given image at the given position(s).

        Args:
            image: The image to compute the energy for.
            positions: The positions of the various depending parts
                inside the image.

        Returns:
            Either a float, if a single configuration, or a `np.ndarray` in
            case multiple positions were given.
        """
        raise NotImplementedError

    def plot_energies(self, image: Image, output_path: Path,
                      candidates: Optional[Dict[str, np.ndarray]] = None,
                      max_dist: float = 10.) \
            -> None:
        """Visualizes the potential's energies values w.r.t. the given image.

        The default implementation fixes all except one part position to the
        true positions and plots the energies w.r.t. this part changing.

        Args:
            image: The image we want to visualize energies for.
            output_path: Path to an image file where to store the result.
            candidates: Optional set of candidates to localize as well if
                possible.
            max_dist: The maximal dist (treated as mm if image has spacing)
                to visualize as localization tolerance.
        """
        plot_results(image, output_path, candidates=candidates,
                     max_dist=max_dist, potential=self)

    def __str__(self):
        return f'{self.name()}({", ".join(self.parts)})'


def select_reference_images(images: List[Image], potentials: List[Potential]) \
        -> List[Tuple[Image, List[Potential]]]:
    """Selects reference images to visualize the given potentials.

    If missing parts are involved, multiple reference image might be taken
    from the list of `images` in case there are annotations missing. The
    images are chosen such that a minimal amount of reference images is
    needed to plot all potentials.

    Args:
        images: The pool of available images.
        potentials: All potentials that we want to visualize later on.

    Returns: A list in which each tuple maps an image to multiple potentials.
    """
    # get images w.r.t. the amount of pots that they could plot
    image_to_pots = [(image, [pot for pot in potentials
                              if all(p in image.parts for p in pot.parts)])
                     for image in images]
    image_to_pots = sorted(image_to_pots,
                           key=lambda x: (-len(x[1]), x[0].name))

    reference_images = []

    assigned_pots = set()
    for image, pots in image_to_pots:
        unassigned_pots = [pot for pot in pots if pot not in assigned_pots]
        if unassigned_pots:
            reference_images.append((image, unassigned_pots))
            assigned_pots.update(unassigned_pots)

    unassigned_pots = [pot for pot in potentials if pot not in assigned_pots]
    if unassigned_pots:
        log.error('Could not find a reference plot image to compute '
                  'energies for for the following potentials: %s',
                  ', '.join(map(str, unassigned_pots)))

    return reference_images


class LearnablePotentialMixin(metaclass=ABCMeta):
    """Represents a learnable potential.

    The potential is expected to learn certain key parameters by using SGD
    w.r.t. the key parameters.

    See: `pbl.learning.FullSgdMaxMarginLearning`
    """

    def feed_values(self, feed_dict: Dict, image: Image) -> None:
        """Called prior an optimization step to provide feed values.

        All required placeholders with values are supposed to be added to
        `feed_dict` for the supplied `image`.

        Args:
            feed_dict: The current set of tensors to feed values.
            image: The image we want to optimize for.
        """
        pass

    def auxiliary_loss_tf(self):
        """Can provide auxiliary loss function that is added to the original
        learned loss function.
        """
        return tf.constant(0., tf.float64)

    @abstractmethod
    def compute_tf(self, positions_tf: List[tf.Tensor]) -> tf.Tensor:
        """Creates the value tensor for `compute()`.

        Args:
            positions_tf: One position tensor for each depending part. Note
                that the position tensors contain multiple positions (first
                axis).

        Return: An output tensor with one value for each (set of) position(s).
        """
        raise NotImplementedError

    def changed(self) -> None:
        """Called after an optimization step.

        This is useful to clear/reset cached values, for instance.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Any:
        """Returns the current set of parameters."""
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, parameters: Any) -> None:
        """Sets the parameters.

        Args:
            parameters: A set of parameters obtained by calling
                `get_parameters()` on the same potential.
        """
        raise NotImplementedError


class BasePotentialMixin(Potential, metaclass=ABCMeta):
    """A potential in our graph.

    Note that this is not a potential in the classical definition, but rather
    a broader definition that catches all the semantics we need.
    """

    def positions(self, images: ImageList, use_mm: bool = False,
                  random_positions: int = 0,
                  max_distance: Union[float, np.ndarray] = 5) -> List:
        """Helper method to get all part positions or logging and ignoring an
        object when not all are given.

        Args:
            images: The images to iterate.
            use_mm: Whether to convert the positions to mm beforehand.
            random_positions: How many additional position pairs to generate
                for one true pair. Use uniform sampling around the true
                position in combination with `max_distance` to generate the
                additional positions.
            max_distance: Maximal distance between a true position and a
                randomly generated one. Specified in mm if `use_mm` is True.

        Returns:
            An iterator yielding the positions in a tuple.
        """
        pairs = tuple([] for _ in self.parts)
        total = 0
        matching = 0

        for image in images:
            for i, obj in enumerate(image.objects):
                total += 1
                missing = False
                positions = []

                for part in self.parts:
                    try:
                        position = obj.parts[part].position
                    except KeyError:
                        position = None

                    if position is None:
                        missing = True
                        break

                    positions.append(position[::-1])

                if missing:
                    continue

                matching += 1

                if use_mm:
                    spacing = image.spacing

                    assert spacing is not None, \
                        'Unable to compute position in mm due to missing ' \
                        'spacing'

                    spacing = spacing[::-1]
                    positions = [p * spacing for p in positions]

                for container, position in zip(pairs, positions):
                    container.append(position)

                    if random_positions > 0:
                        sampled = _sample_around_point(position, max_distance,
                                                       random_positions)
                        container.extend(sampled)

        if matching < total:
            log.warning('Only %i of %i objects over all images contain the '
                        'parts %s', matching, total, ', '.join(self.parts))

        pairs = tuple(np.array(p) for p in pairs)
        if len(self.parts) == 1:
            pairs = pairs[0]

        return pairs


class UnaryPotential(Potential, metaclass=ABCMeta):
    """An abstract potential that depends on **one** node.

    Args:
        part: The single part name.
    """
    def __init__(self, part: str, **kwargs):
        super().__init__([part], **kwargs)

    @staticmethod
    def arity() -> int:
        return 1

    @property
    def part(self):
        """Convenience function to get the parts name."""
        return self.parts[0]


class BinaryPotential(Potential, metaclass=ABCMeta):
    """An abstract potential that depends on **two** nodes."""

    @staticmethod
    def arity() -> int:
        return 2


class TernaryPotential(Potential, metaclass=ABCMeta):
    """An abstract potential that depends on **three** nodes."""

    @staticmethod
    def arity() -> int:
        return 3


class RteLocalizer(UnaryPotential):
    """Localizer based on regression tree ensembles.

    This is basically a wrapper around the `RandomForestLocalizer` provided
    by the `rfl` package.

    Args:
        part: The part to localize.
        rfl_kw: The settings to pass to `RandomForestLocalizer`.
        peak_finder: Used to find the local maxima inside the heatmap.
        cache: Helper class to reduce amount of repeated work.
    """

    OCL = None
    """A global `OCL` instance that is used to speed up the processing."""

    def __init__(self, part: str, n_dims: int,
                 rfl_kw: dict, peak_finder: LocalMaxLocator,
                 cache: Optional['RteLocalizer.Cache'] = None,
                 **kwargs) -> None:
        super().__init__(part, **kwargs)
        assert n_dims in (2, 3)

        self.n_dims = n_dims
        """Image dimensionality"""

        self.model = None
        """The `RandomForestLocalizer` instance that is being used. Only
        available after `train()`-ing.
        """
        self.rfl_kw = rfl_kw
        """The parameters passed to `RandomForestLocalizer`."""
        self.peak_finder = peak_finder
        """Used to find the local maxima inside the heatmap."""
        self.cache = cache
        """Helper container to reduce amount of repeated work."""

        # cached values for faster runtime
        self._cached_path: Optional[Path] = None
        self._cached_heatmap: Optional[np.ndarray] = None

        if RteLocalizer.OCL is None:
            RteLocalizer.OCL = rfl.init_opencl()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_cached_path'] = None
        state['_cached_heatmap'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if RteLocalizer.OCL is None:
            RteLocalizer.OCL = rfl.init_opencl()

    def train(self, train_images: ImageList) -> None:
        @working_dir.cache('rfl_' + self.part)
        def train():
            train_samples = []

            for image in train_images:
                obj = image.objects[0]

                if self.part not in obj.parts:
                    continue

                radius = 5.
                if image.spacing is not None:
                    radius = radius / image.spacing[::-1]

                xs, ys = rfl.create_gaussian_samples(
                    obj.parts[self.part].position[::-1],
                    radius,
                    image.data.shape[:self.n_dims])

                ys *= 1000

                train_samples.append((image.data, xs, ys))

            if len(train_samples) == 0:
                log.critical('No annotations for part %s', self.part)

            model = rfl.RandomForestLocalizer(**self.rfl_kw)
            model.train(train_samples, self.OCL)

            return model

        self.model = train()

    def propose_candidates(self, image: Image) -> Dict[str, np.ndarray]:
        heatmap = self.get_heatmap(image)
        candidates = self.peak_finder.locate(
            heatmap,
            spacing=None if image.spacing is None else image.spacing[::-1],
            target_shape=image.data.shape[:heatmap.ndim])
        return {self.part: candidates}

    def compute(self, image: Image, positions: List[np.ndarray],
                **kwargs) -> Union[float, np.ndarray]:
        pos, = positions
        heatmap = self.get_heatmap(image)

        # FIXME: Interpolation!
        pos = np.round(pos).astype(int)

        values = heatmap[tuple(pos.T)]

        values /= 1000.

        with np.errstate(divide='ignore'):
            values = -np.log(values)

        return values

    def get_heatmap(self, image: Image) -> np.ndarray:
        """Helper function to create or re-use existing heatmap."""

        # we keep the most recent heatmap in memory
        if self._cached_path == image.path:
            return self._cached_heatmap

        # @working_dir.cache(f'rfl_heatmaps/{self.part}/{image.name}')
        # def compute():
        #     return self.model.test(image.data, self.OCL)
        # heatmap = compute()

        if self.cache is None:
            heatmap = self.model.test(image.data, self.OCL)
        else:
            heatmap = self.cache.test(self.model, image, self.OCL)

        self._cached_heatmap = heatmap
        self._cached_path = image.path

        return heatmap

    class Cache:
        """Helper object to not redo unnecessary work."""

        def __init__(self):
            self.image_path = None
            self.prepared_image = None
            self.prepared_buffer = None
            self.process_pool = None

        def __getstate__(self):
            state = dict(self.__dict__)
            state['image_path'] = None
            state['prepared_image'] = None
            state['prepared_buffer'] = None
            state['process_pool'] = None
            return state

        def test(self, model: rfl.RandomForestLocalizer,
                 image: Image, ocl: Optional[rfl.OCL]) -> np.ndarray:
            """Repliactes the test method of the `rfl.RandomForestLocalizer`.

            This method however, caches the output of the smoothing as well as
            the buffer of the output on the GPU.
            """
            if self.image_path != image.path:
                self.prepared_image = model._prepare_image(image.data)
                self.prepared_buffer = None

            start_time = time.time()

            if ocl is None:
                if self.process_pool is None:
                    self.process_pool = ProcessPool()

                values = np.zeros(self.prepared_image.shape[:3], np.float32)

                for local_values in self.process_pool.imap(
                        methodcaller('test', self.prepared_image),
                        model.trees):
                    values += local_values

            else:
                if self.prepared_buffer is None:
                    self.prepared_buffer = cl.Buffer(
                        ocl.ctx, cl.mem_flags.READ_ONLY
                        | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=self.prepared_image)

                values = np.empty(self.prepared_image.shape[:3], np.float32)
                values_cl = cl.Buffer(ocl.ctx, cl.mem_flags.WRITE_ONLY,
                                      values.nbytes)
                cl.enqueue_fill_buffer(ocl.queue, values_cl, b'\x00', 0,
                                       values.nbytes)

                # call the actual kernel for each tree
                kernel_events = [tree.test_ocl(ocl, self.prepared_buffer,
                                               values_cl,
                                               self.prepared_image.shape,
                                               ocl.kernel)
                                 for tree in model.trees]

                cl.enqueue_copy(ocl.queue, values, values_cl,
                                wait_for=kernel_events)

            values /= len(model.trees)
            if model.n_dims == 2:
                values = values[0]

            test_time = time.time() - start_time
            if test_time > 5.:
                log.warning('Performing RTE inference took very long for '
                            'image of shape %s: %.2fs',
                            values.shape, test_time)

            self.image_path = image.path

            return values


class DistancePotential(BinaryPotential, BasePotentialMixin):
    """Estimates a Gaussian distribution to model the distance between
    two nodes. Works in 2D as well as in 3D.
    """

    def __init__(self, *args, use_mm: bool = False,
                 normalize: Optional[str] = None, **kwargs) -> None:
        """Initializes a new :class:`DistancePotential`.

        :param use_mm: Whether to estimate the distances in the voxel domain or
                       in the mm domain.
        """
        assert normalize in (None, 'sum', 'mean')

        super().__init__(*args, **kwargs)

        #: Whether the distances are estimated in voxel domain or mm domain.
        self.use_mm = use_mm

        self.normalize = normalize

        #: The distances of all vectors used to estimate the distribution. Only
        #: valid after calling :meth:`train`.
        self.distances = None

        #: The normal distribution as provided by `scipy.stats`.
        self.dist = None

    def train(self, images: ImageList, random_positions: int = 0,
              max_distance: float = 5.) -> None:
        """Estimates the required Gaussian distribution using the given
        `images`.
        """
        pos_a, pos_b = self.positions(images, self.use_mm, random_positions,
                                      max_distance)

        distances = np.sqrt(np.sum((pos_a - pos_b)**2, axis=1))

        dist = stats.norm(np.mean(distances), np.std(distances))

        log.info('Estimated a Gaussian for the distance between %s and %s '
                 'using %i samples: mean=%.2f, std_dev=%.2f', *self.parts,
                 len(distances), dist.mean(), dist.std())

        self.distances = distances
        self.dist = dist
        self._norm = self.dist.pdf(np.mean(distances))

    def compute(self, image: Image, positions: List[np.ndarray],
                scaling: Optional[float] = None,
                **kwargs) -> Union[Real, np.ndarray]:
        pos_a, pos_b = positions

        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing

        distance = np.sqrt(np.sum((pos_a - pos_b)**2,
                                  axis=None if pos_a.ndim == 1 else 1))

        if scaling is not None:
            distance *= scaling

        # probs = np.abs(self.dist.cdf(distance - self.cdf_delta) -
        #                self.dist.cdf(distance + self.cdf_delta))
        probs = self.dist.pdf(distance)
        if self.normalize == 'sum':
            probs /= probs.sum()
        elif self.normalize == 'mean':
            probs /= self._norm

        # assert (probs >= 0).all() and (probs <= 1).all()

        with np.errstate(divide='ignore'):
            return -np.log(probs)

    def plot(self, ax=None) -> None:
        """Plots the histogram of the training data overlayed by the density
        function.

        Args:
            ax: An optional axes to plot to, if `None` we use `plt.gca()`.
        """
        if ax is None:
            ax = plt.gca()

        min_dist, max_dist = self.distances.min(), self.distances.max()
        pad = (max_dist - min_dist) * 0.05

        x = np.linspace(min_dist - pad, max_dist + pad, 1000)
        y = -np.log(self.dist.pdf(x))

        ax.set_title('Distance from {} to {}'.format(*self.parts))
        ax.set_xlabel('Distance / {}'.format('mm' if self.use_mm else 'px'))
        ax.set_ylabel('Counts')
        ax.hist(self.distances, bins='auto')
        ax = ax.twinx()
        ax.set_ylabel('Energy')
        ax.set_yscale('log')
        ax.plot(x, y, 'r-')

        if self.unknown_energy is not None:
            ax.plot(*np.transpose([[x[0], self.unknown_energy],
                                   [x[-1], self.unknown_energy]]), 'r-')


class AnglePotential(BinaryPotential, BasePotentialMixin):
    """A """

    def __init__(self, *args, use_mm: bool = False,
                 view: List[int] = [0, 1], normalize: Optional[str] = None,
                 **kwargs) \
            -> None:
        """Initializes a new :class:`AnglePotential`.

        Args:
            use_mm: Whether to estimate the angles from the positions
                converted to mm.
            axes:
        """
        assert normalize in (None, 'sum', 'mean')

        super().__init__(*args, **kwargs)

        assert len(view) == 2

        #: Whether the distances are estimated in voxel domain or mm domain.
        self.use_mm = use_mm

        #: The axes selector to allow computation of angles in 3D space of a
        #: certan viewing plane.
        self.view = np.asarray(view)

        self.normalize = normalize

        #: List of angles used to estimate the distribution's parameters.
        self.angles = None

        #: The underlying distribution, only valid after calling :meth:`train`.
        self.dist = None

    def train(self, images: ImageList) -> None:
        """Estimates the required von Mises distribution using the given
        `images`.
        """
        pos_a, pos_b = self.positions(images, self.use_mm)

        angles = pos_b[:, self.view] - pos_a[:, self.view]
        angles = np.arctan2(angles[:, 0], angles[:, 1])

        assert (angles >= -np.pi).all() and (angles <= np.pi).all()

        # convert angles to their coordinate on the unit circle
        angles_uc = np.vstack((np.sin(angles), np.cos(angles))).T

        # estimate dist parameters given same samples
        r = np.sum(angles_uc, axis=0)
        r2 = np.sqrt(np.sum(r**2))
        r_ = r2 / len(angles)

        # the parameters of the von Mises distribution
        mu = r / r2
        mu = np.arctan2(mu[0], mu[1])
        k = (r_*2 - r_**3) / (1 - r_**2)

        # a to large kappa results in overflows in the exp function in the
        # pdf, thus we got to enforce an upper bound
        k = min(k, 600)

        log.info('Estimated a von Mises for the angle between %s and %s using '
                 '%i samples: mean=%.2f, k=%.2f', *self.parts, len(angles), mu,
                 k)

        self.angles = angles
        self.dist = stats.vonmises(loc=mu, kappa=k)
        self._norm = self.dist.pdf(mu)

    def compute(self, image: Image, positions: List[np.ndarray],
                **kwargs):
        pos_a, pos_b = positions

        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing

        pos_a = pos_a[:, self.view]
        pos_b = pos_b[:, self.view]
        delta = pos_b - pos_a

        angle = np.arctan2(delta[:, 0], delta[:, 1])

        assert (angle >= -np.pi).all() and (angle <= np.pi).all()

        probs = self.dist.pdf(angle)
        if self.normalize == 'sum':
            probs /= probs.sum()
        elif self.normalize == 'mean':
            probs /= self._norm

        # assert (probs >= 0).all() and (probs <= 1).all()

        with np.errstate(divide='ignore'):
            return -np.log(probs)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        pad = (self.angles.max() - self.angles.min()) * 0.01
        x = np.linspace(self.angles.min() - pad, self.angles.max() + pad, 1000)
        y = -np.log(self.dist.pdf(x))
        x = x / np.pi * 180

        ax = plt.gcf().add_subplot(111)

        ax.hist(self.angles / np.pi * 180, bins='auto')
        ax.set_title('Angle going from {} to {}'.format(
            self.part_a, self.part_b))
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Counts')
        ax.set_xlim(-180, 180)
        ax = ax.twinx()
        ax.set_ylabel('Energy')
        ax.set_yscale('log')
        # ax.set_ylim(bottom=np.floor(np.log10(y.min())))
        ax.plot(x, y, 'r-')

        if self.unknown_energy is not None:
            ax.plot(*np.transpose([[x[0], self.unknown_energy],
                                   [x[-1], self.unknown_energy]]), 'r-')


class VectorPotential(BinaryPotential, BasePotentialMixin):
    """Estimates a Gaussian distribution for a multi-dimensional vector,
    allowing you to combine distance and angle in one potential.
    """

    def __init__(self, parts, use_mm: bool = False,
                 normalize: Optional[str] = 'mean',
                 **kwargs):
        """Initializes the :class:`VectorPotential`.

        Args:
            use_mm: Whether to use vectors in mm space.
        """
        assert normalize in (None, 'sum', 'mean')

        super().__init__(parts, **kwargs)

        #: Whether to use vectors in mm space.
        self.use_mm = use_mm

        self.normalize = normalize

        #: An array of the vectors used to estimate the
        #: distribution, or `None` in case :meth:`train` hasn't been
        #: called yet.
        self.vectors: Optional[np.ndarray] = None

        #: A frozen `multivariate_normal` distribution, or `None`
        #: in case :meth:`train` hasn't been called yet.
        self.dist: Optional[Any] = None

    def name(self):
        return 'vector'

    def train(self, images: ImageList, random_positions: int = 0,
              max_distance: float = 5.):
        pos_a, pos_b = self.positions(images, self.use_mm, random_positions,
                                      max_distance)
        vectors = pos_b - pos_a

        mean = np.mean(vectors, axis=0)
        cov = np.cov(vectors, rowvar=False)

        self.dist = stats.multivariate_normal(mean=mean, cov=cov)
        self.vectors = vectors

        log.info('Estimated a Gaussian for the vector between %s and '
                 '%s using %i samples: mean=%s', *self.parts, len(vectors),
                 self.dist.mean[::-1])

    def compute(self, image: Image, positions: List[np.ndarray],
                scaling: Optional[float] = None,
                **kwargs) -> Union[Real, np.ndarray]:
        pos_a, pos_b = positions

        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing

        vector = pos_b - pos_a

        if scaling is not None:
            vector *= scaling

        # if self.ROTATION is not None:
        #     transform = sitk.Euler3DTransform()
        #     transform.SetRotation(*(self.ROTATION[::-1]))
        #     vector = [
        #         transform.TransformPoint((v.tolist()[::-1]))[::-1]
        #         for v in vector]

        probs = self.dist.pdf(vector)
        if self.normalize == 'sum':
            probs /= probs.sum()
        elif self.normalize == 'mean':
            probs /= self.dist.pdf(self.dist.mean)
            assert (probs >= 0).all() and (probs <= 1).all()

        with np.errstate(divide='ignore'):
            energies = -np.log(probs)

        # fix single values although we wanted to have multiple ones
        if pos_a.ndim == 2 and energies.ndim == 0:
            energies = np.array([energies])

        return energies

    def plot(self, ax=None):
        """Illustrates the estimated distribution including the
        underlying samples.

        Args:
            image: The image we use to illustrate the properties with.
            ax: An `Axes` to plot to, if `None`, the current active
                `Axes` is used.
        """
        if ax is None:
            ax = plt.gca()

        min_pos = np.amin(np.vstack((self.vectors, [0, 0])), axis=0)
        max_pos = np.amax(np.vstack((self.vectors, [0, 0])), axis=0)
        pad = (max_pos - min_pos) * 0.15
        min_pos = (min_pos - pad).astype(int)
        max_pos = np.ceil(max_pos + pad).astype(int)

        x = np.linspace(min_pos[1], max_pos[1], 1000)
        y = np.linspace(min_pos[0], max_pos[0], 1000)
        X, Y = np.meshgrid(x, y)

        Z = -np.log(self.dist.pdf(np.transpose([Y.reshape(Y.size),
                                                X.reshape(X.size)])))
        Z.shape = X.shape

        Z = np.ma.masked_where(~np.isfinite(Z), Z)

        if self.unknown_energy is not None:
            ax.contour(X, Y, Z, levels=[self.unknown_energy])

        cnt = plt.imshow(Z, cmap=plt.cm.inferno_r,
                         extent=(x[0], x[-1], y[-1], y[0]))
        ax.set_autoscale_on(False)

        ax.set_aspect('equal')
        ax.plot(0, 0, 'g.')
        ax.plot(*np.transpose(self.vectors[:, ::-1]), 'r.')

        ax.set_title('Vector going from {} to {}'.format(
            self.part_a, self.part_b))
        ax.set_xlabel('mm' if self.use_mm else 'px')
        ax.set_ylabel('mm' if self.use_mm else 'px')

        plt.gcf().colorbar(cnt)


class KdeVectorPotential(BinaryPotential, BasePotentialMixin):
    """Uses Kernel Density Estimation to find a distribution matching the
    data.

    Args:
        use_mm: Whether to use vectors in mm space.
    """

    def __init__(self, *parts, use_mm: bool = False, **kwargs):
        super().__init__(*parts, **kwargs)

        #: Whether to use vectors in mm space.
        self.use_mm = use_mm

        #: An array of the vectors used to estimate the
        #: distribution, or `None` in case :meth:`train` hasn't been
        #: called yet.
        self.vectors: Optional[np.ndarray] = None

        #: A frozen `multivariate_normal` distribution, or `None`
        #: in case :meth:`train` hasn't been called yet.
        self.dist: Optional[Any] = None

    def train(self, images: ImageList, random_positions: int = 0,
              max_distance: float = 5.):
        pos_a, pos_b = self.positions(images, self.use_mm, random_positions,
                                      max_distance)
        vectors = pos_b - pos_a

        @working_dir.cache(self.name() + str(self.parts))
        def train():
            grid = GridSearchCV(KernelDensity(),
                                {'bandwidth': 100**np.linspace(-1, 1, 500)},
                                cv=min(20, len(vectors)), iid=False)
            grid.fit(vectors)

            log.debug('Best bandwidth is %f', grid.best_params_['bandwidth'])

            return grid.best_estimator_

        self.kde = train()
        self.vectors = vectors

    def compute(self, image: Image, positions: np.ndarray,
                scaling: Optional[float] = None,
                **kwargs) -> Union[Real, np.ndarray]:
        pos_a, pos_b = positions

        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing

        vector = pos_b - pos_a

        if scaling is not None:
            vector *= scaling

        # if self.ROTATION is not None:
        #     transform = sitk.Euler3DTransform()
        #     transform.SetRotation(*(self.ROTATION[::-1]))
        #     vector = [
        #         transform.TransformPoint((v.tolist()[::-1]))[::-1]
        #         for v in vector]

        n = 1 if vector.ndim == 1 else len(vector)

        return -(self.kde.score_samples(vector) + np.log(n))


class UnitVectorPotential(BinaryPotential):
    """Estimates a Gaussian distribution for a multi-dimensional vector,
    allowing you to grasb distance and angle in one potential.
    """

    def __init__(self, *parts, use_mm: bool = False, **kwargs):
        """Initializes the :class:`VectorPotential`.

        Args:
            use_mm: Whether to use vectors in mm space.
        """
        super().__init__(*parts, **kwargs)

        #: Whether to use vectors in mm space.
        self.use_mm = use_mm

        #: An array of the vectors used to estimate the
        #: distribution, or `None` in case :meth:`train` hasn't been
        #: called yet.
        self.vectors: Optional[np.ndarray] = None

        #: A frozen `multivariate_normal` distribution, or `None`
        #: in case :meth:`train` hasn't been called yet.
        self.dist: Optional[Any] = None

    def train(self, images: ImageList, random_positions: int = 0,
              max_distance: float = 5.):
        """Trains te
        """
        pos_a, pos_b = self.positions(images, self.use_mm, random_positions,
                                      max_distance)
        vectors = pos_b - pos_a

        magnitude = np.sqrt(np.sum(vectors**2, axis=1))
        magnitude.shape = (magnitude.size, 1)

        vectors /= magnitude

        mean = np.mean(vectors, axis=0)
        cov = np.cov(vectors, rowvar=False)

        self.dist = stats.multivariate_normal(mean=mean, cov=cov)
        self.vectors = vectors
        self._norm = self.dist.pdf(mean)

        log.info('Estimated a Gaussian for the unit vector between %s and '
                 '%s using %i samples: mean=%s', *self.parts, len(vectors),
                 self.dist.mean[::-1])

    def compute(self, image: Image, pos_a: np.ndarray,
                pos_b: np.ndarray, **kwargs) -> Union[Real, np.ndarray]:
        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing

        vector = pos_b - pos_a

        magnitude = np.sqrt(np.sum(vector**2, axis=1 if vector.ndim == 2
                                   else None))
        magnitude.shape = (magnitude.size, 1)

        magnitude[magnitude == 0] = 1

        vector /= magnitude

        probs = self.dist.pdf(vector)
        # probs /= self._norm

        # assert (probs >= 0).all() and (probs <= 1).all()

        with np.errstate(divide='ignore'):
            return -np.log(probs)


class DistanceRatioPotential(TernaryPotential, BasePotentialMixin):
    """A ternary potential evaluating the ratio of two distances."""

    def __init__(self, parts, use_mm: bool = False, **kwargs):
        """Initializes the :class:`DistanceRatioPotential`.

        Args:
            use_mm: Whether to use distances in mm space.
        """
        assert len(parts) == 3

        super().__init__(parts, **kwargs)

        #: Whether the distances are estimated in voxel domain or mm domain.
        self.use_mm = use_mm

        #: The normal distribution as provided by `scipy.stats`.
        self.dist = None

    def train(self, images: ImageList, random_positions: int = 0,
              max_distance: float = 5.) \
            -> None:
        """Estimates the required Gaussian distribution using the given
        `images`.
        """
        pos_a, pos_b, pos_c = self.positions(images, self.use_mm,
                                             random_positions,
                                             max_distance)

        distances_a = np.sqrt(np.sum((pos_a - pos_b)**2, axis=1))
        distances_b = np.sqrt(np.sum((pos_b - pos_c)**2, axis=1))

        ratio = distances_a / distances_b

        dist = stats.norm(np.mean(ratio), np.std(ratio))

        log.info('Estimated a Gaussian for the distance ratio between %s, %s '
                 'and %s using %i samples: mean=%.2f, std_dev=%.2f',
                 *self.parts, len(distances_a), dist.mean(), dist.std())

        self.dist = dist

    def compute(self, image: Image, pos) -> Union[Real, np.ndarray]:
        pos_a, pos_b, pos_c = pos
        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing
            pos_c = pos_c * spacing

        distance_a = np.sqrt(np.sum((pos_a - pos_b)**2,
                                    axis=None if pos_a.ndim == 1 else 1))
        distance_b = np.sqrt(np.sum((pos_b - pos_c)**2,
                                    axis=None if pos_a.ndim == 1 else 1))

        print('d1', distance_a, 'd2', distance_b)
        ratio = distance_a / distance_b

        probs = self.dist.pdf(ratio)
        probs[(distance_a == 0) | (distance_b == 0)] = 0

        with np.errstate(divide='ignore'):
            return -np.log(probs)


class RelativeAnglePotential(TernaryPotential, BasePotentialMixin):
    """A ternary potential evaluating the angle between two vectors formed by
    three positions."""

    def __init__(self, parts, use_mm: bool = False,
                 view: List[int] = [0, 1], **kwargs):
        """Initializes the :class:`DistanceRatioPotential`.

        Args:
            use_mm: Whether to use distances in mm space.
        """
        assert len(parts) == 3

        super().__init__(parts, **kwargs)

        #: Whether the distances are estimated in voxel domain or mm domain.
        self.use_mm = use_mm

        #: In which plane to compute the vectors.
        self.view = np.asarray(view, dtype=int)

        #: The normal distribution as provided by `scipy.stats`.
        self.dist = None

    def train(self, images: ImageList) -> None:
        """Estimates the required von Mises distribution using the given
        `images`.
        """
        pos_a, pos_b, pos_c = self.positions(images, self.use_mm)

        pos_a = pos_a[:, self.view]
        pos_b = pos_b[:, self.view]
        pos_c = pos_c[:, self.view]

        vec1 = pos_b - pos_a
        vec2 = pos_c - pos_b

        angles = _compute_angle(vec1, vec2)

        # convert angles to their coordinate on the unit circle
        angles_uc = np.vstack((np.sin(angles), np.cos(angles))).T

        # estimate dist parameters given same samples
        r = np.sum(angles_uc, axis=0)
        r2 = np.sqrt(np.sum(r**2))
        r_ = r2 / len(angles)

        # the parameters of the von Mises distribution
        mu = r / r2
        mu = np.arctan2(mu[0], mu[1])
        k = (r_*2 - r_**3) / (1 - r_**2)

        # a to large kappa results in overflows in the exp function in the
        # pdf, thus we got to enforce an upper bound
        k = min(k, 600)

        log.info('Estimated a von Mises for the angle between vector %s->%s '
                 'and %s->%s using %i samples: mean=%.2f, k=%.2f',
                 *self.parts[:2], *self.parts[1:], len(angles), mu, k)

        self.dist = stats.vonmises(loc=mu, kappa=k)
        self._norm = self.dist.pdf(mu)

    def compute(self, image: Image, pos):
        pos_a, pos_b, pos_c = pos
        if self.use_mm:
            spacing = image.spacing[::-1]
            pos_a = pos_a * spacing
            pos_b = pos_b * spacing
            pos_c = pos_c * spacing

        pos_a = pos_a[:, self.view]
        pos_b = pos_b[:, self.view]
        pos_c = pos_c[:, self.view]

        vec1 = pos_b - pos_a
        vec2 = pos_c - pos_b

        angle = _compute_angle(vec1, vec2)

        probs = self.dist.pdf(angle)
        probs[np.isnan(angle)] = 0

        with np.errstate(divide='ignore'):
            return -np.log(probs)


def _compute_angle(vec1: np.ndarray, vec2: np.ndarray) \
        -> Union[float, np.ndarray]:
    """Computes the angle between `vec1` and `vec2`. The returned angle is
    signed and depends on the order of the vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Either a float if only one vector pair is given or an array in case
        multiple vectors are given.
    """
    just_one = False
    if vec1.ndim == 1:
        just_one = True
        vec1 = np.array([vec1])
        vec2 = np.array([vec2])

    l1 = np.sqrt(np.sum(vec1**2, axis=1))
    l2 = np.sqrt(np.sum(vec2**2, axis=1))

    # FIXME: That's probably really slow, can we somewhat improve it?
    dot = np.array([np.dot(v1, v2) for v1, v2 in zip(vec1, vec2)])
    cross = np.cross(vec2, vec1)
    prod = dot / (l1 * l2)
    prod[prod > 1] = 1

    angle = np.sign(cross) * np.arccos(prod)
    assert (((angle >= -np.pi) & (angle <= np.pi)) | np.isnan(angle)).all()

    if just_one:
        angle = angle[0]

    return angle


def _sample_around_point(center, max_distance: float,
                         count: int = 1, seed: int = 31337) -> np.ndarray:
    """Uniformly samples points around a pivot with a certain maximal
    Euclidean distance.

    Args:
        center: The origin to estimate the distance to.
        max_distance: The maximal Euclidean distance between the sampled point
            and the `center` point.
        count: Number of points to sample.
        seed: A number to initialize the RNG.

    Returns:
        A list (or one) sampled point.
    """
    rng = np.random.RandomState(seed)

    samples = []

    while len(samples) < count:
        sample = np.array([rng.uniform(c - max_distance, c + max_distance)
                           for c in center])

        if distance.euclidean(sample, center) > max_distance:
            continue

        samples.append(sample)

    return samples[0] if count == 1 else np.array(samples)


class MultiPartCNN:
    """Multi part key point generator based on convolutional neural networks.

    Uses a CNN to generate heatmaps for multiple parts at once, i.e., multiple
    final feature maps. This should be very effective in terms of reusing
    intermediate features w.r.t. repetitive structures.

    Pickling this instance is supported, including reloading the weights.
    When unpickling such an instance, the default TF session is used. Make
    sure to to change the default session appropriately if you want to use
    a specific one.

    Args:
        session: The TensorFlow session (and associated graph) to create the
            CNN and the weights in.
        parts: Names of the parts.
        n_dims: Input image dimensions, i.e., 2 or 3.
        n_channels: Input image channels. Images with no channel axis are
            treated as having one channel.
        peak_finder: Used to find the local maxima inside the heatmaps.
        target_value: The heatmap peak target value.
        l2_norm: Amount of weight L2 norm to use as auxiliary loss.
        dropout_rate: If given, uses dropout in training. Good value is
            within 0.2-0.5.
    """
    def __init__(self, session: tf.Session, parts: List[str], n_dims: int,
                 n_channels: int, input_shape: np.ndarray,
                 peak_finder: LocalMaxLocator,
                 target_value: float = 10_000,
                 l2_norm: float = 0.,
                 dropout_rate: Optional[float] = None):
        assert len(parts) > 0
        assert n_dims in (2, 3)
        assert n_channels > 0

        self.session = session
        """The TF session we operate in."""

        self.parts = parts
        """The name of the parts for the corresponding output feature maps."""
        self.n_parts = len(parts)
        """Number of parts to localize."""
        self.n_dims = n_dims
        """Number of image dimensions"""
        self.n_channels = n_channels
        """Number of image channels."""
        self.input_shape = np.array(input_shape)
        """Use this to indicate a maximal image input shape, i.e., keep
        a constant input image size.
        """
        self.l2_norm = l2_norm
        """Amount of L2 norm to use for the weights."""
        self.dropout_rate = dropout_rate
        """Amount of dropout to use, if given."""
        self.target_value = target_value
        """The target value to reach in the heatmap."""
        self.peak_finder = peak_finder
        """Used to find the local maxima inside the heatmaps."""

        self.input_tf = None
        """The input image placeholder."""
        self.output_tf = None
        """The output of the final layer without activation."""
        self.normalized_output_tf = None
        """Normalized version (energy) of the final layer."""
        self.dropout_rate_tf = None
        """The dropout rate to use."""
        self.trainables_tf = None
        """List of all trainable variables of the network."""
        self.weights_l2_tf = None
        """L2 loss applied to all weights (_NOT_ biases!) and summed."""
        self.is_training_tf = None
        """Boolean indicating whether we are currently in training mode."""
        self.downsampling = None
        """The downsampling applied by the network. The final output is free
        of this downsampling.
        """
        self.output_shape = None
        """Defined later, since it might be different than input shape due
        to downsampling."""
        self.channels_first = None
        """Whether channels are before or after the image axes."""

        # create the network graph
        self.create_graph()

        self.potentials = [MultiPartCNN.Potential(self, i, name)
                           for i, name in enumerate(parts)]
        """One unary potential for each part. These are just shallow proxies
        that make use of the joint CNN.
        """

        # for better performance between propose candidates and compute calls
        # we cache the output for the most recent image
        self._cache_image = None
        self._cache_output = None

    def __getstate__(self):
        state = dict(self.__dict__)

        # drop unpicklable and unneeded things
        state['_cache_image'] = None
        state['_cache_output'] = None
        state['session'] = None
        state['input_tf'] = None
        state['output_tf'] = None
        state['normalized_output_tf'] = None
        state['is_training_tf'] = None
        state['dropout_rate_tf'] = None
        state['trainables_tf'] = None
        state['weights_l2_tf'] = None
        # state['input_shape'] = None
        state['channels_first'] = None

        # get weights for later restore
        state['_weights'] = self.get_weights()

        return state

    def __setstate__(self, state):
        weights = state.pop('_weights')

        self.__dict__ = state

        # we make use of the default session
        self.session = tf.get_default_session()
        assert self.session is not None, 'please set a default TF session'

        self.create_graph(initialize_weights=False)

        self.set_weights(weights)

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Gets the current weights for all trainable TF variables."""
        assert self.session is not None
        return dict(zip([var.name for var in self.trainables_tf],
                        self.session.run(self.trainables_tf)))

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Sets the given weights for all trainable TF variables."""
        assert self.session is not None
        for var in self.trainables_tf:
            try:
                var.load(weights[var.name], self.session)
            except KeyError as e:
                raise RuntimeError('the given weights do not fit our model') \
                    from e

    def load_vgg19_weights(self):
        with tf.Graph().as_default():
            with tf.Session().as_default():
                from keras.applications.vgg19 import VGG19

                vgg19 = VGG19(weights='imagenet')

                for layer in range(1, 3):
                    for name, values in zip(['weights', 'bias'],
                                            vgg19.layers[layer].get_weights()):
                        var_tf = next(
                            var for var in self.trainables_tf if
                            var.name == f'multipartcnn/conv{layer}_{name}:0')
                        var_tf.load(values, self.session)

    def create_graph(self, initialize_weights: bool = True):
        """Create the TF graph structure for the network."""
        assert self.session is not None

        # the data representation we use
        channels_first = any(d.device_type == 'GPU'
                             for d in self.session.list_devices())
        data_format = ('NCx' if channels_first else 'NxC').replace(
            'x', 'HW' if self.n_dims == 2 else 'DHW')

        layer = 1
        receptive_field, jump = 1, 1
        input_channels = self.n_channels
        downsampling = np.ones(self.n_dims, dtype=int)
        weights_l2 = []

        def max_pool(x, size, stride):
            nonlocal layer, downsampling
            nonlocal receptive_field, jump

            receptive_field += (size - 1) * jump
            jump *= stride
            log.debug('Layer %i: MaxPool(kernel=%i stride=%i '
                      'receptive_field=%i)',
                      layer, size, stride, receptive_field)

            def shapify(v):
                return (1, 1) + (v,) * self.n_dims if channels_first else \
                    (1,) + (v,) * self.n_dims + (1,)

            y = (tf.nn.max_pool3d if self.n_dims == 3 else
                 tf.nn.max_pool)(x, shapify(size), shapify(stride),
                                 padding='SAME', data_format=data_format)

            layer += 1
            downsampling *= stride

            return y

        def batch_norm(x, scale=False):
            return tf.contrib.layers.batch_norm(
                x, data_format=data_format.replace('D', ''),
                trainable=True, scale=scale,
                is_training=self.is_training_tf,
                variables_collections=['pbl_batch_norm_coll'])

        def conv(x, size, channels, final=False, stride=1, dilation=1):
            nonlocal layer, input_channels, downsampling
            nonlocal receptive_field, jump

            receptive_field += (size + (size - 1) * (dilation - 1) - 1) * jump
            jump *= stride
            log.debug('Layer %i: Conv(kernel=%i channels=%i stride=%i '
                      'dilation=%i receptive_field=%i)',
                      layer, size, channels, stride, dilation, receptive_field)

            initializer = None
            if initialize_weights:
                # paper "delving deep into rectifiers" recommends this
                # initialization for ReLU activated layers, alternative is
                # xavier
                initializer = tf.contrib.layers.variance_scaling_initializer(
                    seed=layer)

            weights = tf.get_variable(shape=(size,) * self.n_dims +
                                            (input_channels, channels),
                                      initializer=initializer,
                                      name=f'conv{layer}_weights',
                                      dtype=tf.float32)
            bias = tf.get_variable(shape=(channels,),
                                   initializer=tf.zeros_initializer()
                                   if initialize_weights else None,
                                   name=f'conv{layer}_bias',
                                   dtype=tf.float32)

            y = tf.nn.convolution(x, weights, strides=(stride,) * self.n_dims,
                                  dilation_rate=(dilation,) * self.n_dims,
                                  data_format=data_format, padding='SAME')
            y = tf.nn.bias_add(y, bias,
                               data_format=data_format.replace('D', ''))

            if not final:
                y = tf.nn.relu(y)

            layer += 1
            input_channels = channels
            downsampling *= stride
            weights_l2.append(tf.nn.l2_loss(weights))

            return y

        with self.session.graph.as_default():
            with tf.variable_scope('multipartcnn'):
                assert self.input_shape is not None

                self.is_training_tf = tf.placeholder_with_default(
                    False, shape=())
                self.dropout_rate_tf = tf.placeholder_with_default(
                    0.0, shape=())

                # image input
                input_shape = [None] + [None] * self.n_dims
                input_shape.insert(1 if channels_first else len(input_shape),
                                   self.n_channels)
                input_tf = tf.placeholder(tf.float32, input_shape,
                                          name='input')

                # lazy data normation
                output_tf = batch_norm(input_tf, scale=True)

                output_tf = conv(output_tf, 9, 128)
                output_tf = max_pool(output_tf, 2, 2)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 9, 128)
                output_tf = max_pool(output_tf, 2, 2)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 9, 128)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 5, 32)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 9, 128)
                output_tf = batch_norm(output_tf)

                output_tf = conv(output_tf, 11, 128)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 11, 128)
                output_tf = batch_norm(output_tf)
                output_tf = conv(output_tf, 1, 512)
                output_tf = batch_norm(output_tf)

                # dropout should be placed after ALL batchnorm stuff!
                # for spatial dropout define the noise_shape param
                if self.dropout_rate is not None:
                    output_tf = tf.nn.dropout(
                        output_tf, rate=self.dropout_rate_tf)

                # output_tf = conv(input_tf, 3, 64)
                # output_tf = conv(output_tf, 9, 64)
                # output_tf = conv(output_tf, 3, 128, dilation=2)  # r=9
                # output_tf = conv(output_tf, 3, 128, dilation=2)  # r=13
                # output_tf = conv(output_tf, 7, 256, dilation=2)  # r=25
                # output_tf = conv(output_tf, 7, 256, dilation=2)  # r=37
                # output_tf = conv(output_tf, 1, self.n_parts, final=True)

                output_tf = conv(output_tf, 1, self.n_parts, final=True)

                # for the output, we always make sure channels are first
                if not channels_first:
                    output_tf = tf.transpose(
                        output_tf, perm=[0, self.n_dims + 1]
                        + list(range(1, self.n_dims + 1)))

                # Gibbs measure to model probability distribution over
                # image domain, we use log-sum-exp trick to prevent
                # under/overflow
                # image_start = 2 if channels_first else 1
                # axis = np.arange(image_start, image_start + self.n_dims)
                # max_output_tf = tf.reduce_max(output_tf, axis=axis,
                #                               keepdims=True)
                # exp_tf = tf.exp(output_tf - max_output_tf)
                # sum_tf = tf.reduce_sum(exp_tf, axis=axis, keepdims=True)
                # logsumexp_tf = max_output_tf + tf.log(sum_tf)
                # normalized_output_tf = -(output_tf - logsumexp_tf)

                target_value_tf = tf.constant(self.target_value, tf.float32)
                normalized_output_tf = -tf.log(
                    tf.minimum(tf.maximum(output_tf,
                                          tf.constant(1e-30, tf.float32)),
                               target_value_tf) / target_value_tf)

            self.input_tf = input_tf
            self.output_tf = output_tf
            self.normalized_output_tf = normalized_output_tf
            self.trainables_tf = tf.trainable_variables('multipartcnn') \
                + tf.get_collection('pbl_batch_norm_coll')
            assert len(tf.get_collection('pbl_batch_norm_coll')) > 0

            self.weights_l2_tf = tf.reduce_sum(weights_l2)

        self.downsampling = downsampling
        self.channels_first = channels_first

        self.input_shape = (np.ceil(self.input_shape / self.downsampling)
                            * self.downsampling).astype(int)
        self.output_shape = self.input_shape // self.downsampling

        if initialize_weights:
            self.session.run(tf.variables_initializer(self.trainables_tf))

        n_weights = sum(np.prod(var.shape.as_list())
                        for var in self.trainables_tf)
        log.info('MultiPartCNN has %i trainable parameters (%s) and a '
                 'receptive field size of %i using an input shape of %s',
                 n_weights, format_size(n_weights * 4), receptive_field,
                 self.input_shape)

    def infer(self, image: Image) -> np.ndarray:
        """Infers the energy maps for the supplied image.

        Args:
            image: The image that we want to infer.

        Return: All energy maps for the image.
        """
        if self._cache_image == image:
            return self._cache_output

        input_stride = (np.round((np.array(self.input_shape) / 3 * 2)
                                 / self.downsampling)
                        * self.downsampling).astype(int)
        input_patcher = PatchExtractor(self.input_shape, stride=input_stride)

        output_stride = input_stride // self.downsampling
        output_patcher = PatchExtractor(self.output_shape,
                                        stride=output_stride,
                                        overlap_reduction='max')

        image_shape = image.data.shape[:self.n_dims]
        n_patches = np.maximum(np.ceil((image_shape - self.input_shape)
                                       / input_stride), 0).astype(int) + 1
        heatmap_shape = output_stride * n_patches + (self.output_shape
                                                     - output_stride)
        # the image might be smaller than the patch size, thus we have to
        # get the actual part
        valid_heatmap_shape = np.ceil(
            image_shape / self.downsampling).astype(int)

        output_patches = []
        for patch in input_patcher.extract(image.data):
            input_patch = self.create_input(patch)
            output_patch = self.session.run(
                self.output_tf, feed_dict={self.input_tf: input_patch})
            output_patches.append(output_patch[0])

        # stitch the output patches together
        heatmap = np.empty((self.n_parts,) + tuple(valid_heatmap_shape),
                           np.float32)
        for i in range(self.n_parts):
            tmp_hm = output_patcher.reconstruct(
                heatmap_shape, [p[i] for p in output_patches], np.nan)
            heatmap[i] = tmp_hm[tuple(map(slice, valid_heatmap_shape))]

        assert not np.isnan(heatmap).any()

        self._cache_image = image
        self._cache_output = heatmap

        return heatmap

    def propose_candidates(self, image: Image) -> Dict[str, np.ndarray]:
        """Returns the best candidates for all parts."""
        heatmaps = self.infer(image)

        all_candidates = {}

        for part_idx, part in enumerate(self.parts):
            candidates = self.peak_finder.locate(
                heatmaps[part_idx],
                spacing=None if image.spacing is None else image.spacing[::-1],
                downsampled=self.downsampling,
                target_shape=image.data.shape[:self.n_dims])

            all_candidates[part] = candidates

        return all_candidates

    def create_input(self, input_: np.ndarray) -> np.ndarray:
        """Creates a properly formatted input array for `self.input_tf`.

        Args:
            input_: The image we want to get the data for.

        Return: A properly shaped array.
        """
        assert ((len(input_.shape) == self.n_dims + 1
                 and input_.shape[-1] == self.n_channels)
                or (len(input_.shape) == self.n_dims
                    and self.n_channels == 1)), \
            f'invalid input image with shape {input_.shape}: network is ' \
            f'setup for n_dims={self.n_dims} and n_channels={self.n_channels}'

        # eventually missing channels dim
        if input_.ndim == self.n_dims:
            input_ = input_.reshape(input_.shape + (1,))

        if self.input_shape is not None:
            assert (self.input_shape >= input_.shape[:self.n_dims]).all(), \
                'input image is larger than input shape size!'

        padded_size = input_.shape[:self.n_dims] if self.input_shape is None \
            else self.input_shape
        padded_size = (np.ceil(padded_size / self.downsampling)
                       * self.downsampling).astype(int)

        padding = [(0, i - s) for i, s in zip(padded_size, input_.shape)] \
            + [(0, 0)]
        input_ = np.pad(input_, padding, mode='constant')

        if self.channels_first:
            input_ = np.transpose(input_,
                                  [self.n_dims] + list(range(self.n_dims)))

        return input_.reshape((1,) + input_.shape).astype(np.float32)

    def train(self, criterion: Criterion, train_images: List[Image],
              val_images: List[Image], batch_size: int = 8,
              relative_loss: bool = False,
              max_train_time: float = 24. * 60 * 60,
              elastic_transform: Optional[float] = None,
              metrics_every: int = 100,
              flip_axis: Tuple[int, float] = (0, 0),
              value_aug: Tuple[float, float, float] = (0.25, 0.25, 0)) -> None:
        """Trains the CNN.

        Args:
            criterion: The criterion we are optimizing against.
            train_images: The training images to use.
            val_images: The validation images to use.
            batch_size: Number of images in one batch.
            relative_loss: Whether to use a new relative loss formulation
                instead of the MSE over target heatmaps.
            max_train_time: Maximal amount to spend on the training in
                seconds.
            elastic_transform: Chance of using elastic transform.
            metrics_every: Computes the metris every i-th iteration.
            flip_axis: If given, specifies the axis (0-th item) and the
                chance (2-nd item) of this axis getting flipped.
            value_aug: Optional value augmentation, encoded as
                `(shift, 1+scale, chance)`.
        """
        if elastic_transform is None:
            elastic_transform = 0.

        flip_axis, flip_chance = flip_axis
        value_shift, value_scale, value_chance = value_aug
        assert 0 <= value_chance <= 1

        assert 0 <= elastic_transform <= 1
        assert self.input_shape is not None

        @working_dir.cache('cnn_weights')
        def train_weights():
            log.info('Training CNN with %i training and %i validation images '
                     'and batch size %i', len(train_images),
                     len(val_images), batch_size)

            with self.session.graph.as_default():
                # the normal SSE error loss between heatmap and true one
                true_output_tf = tf.placeholder(tf.float32,
                                                self.output_tf.shape)
                valid_output_area_tf = tf.placeholder(tf.int32,
                                                      (self.n_dims, 2))
                part_weights_tf = tf.placeholder(tf.float32, self.n_parts)

                diff_tf = self.output_tf - true_output_tf
                diff_tf = diff_tf[(0, slice(None),) +
                                  tuple(slice(valid_output_area_tf[i, 0],
                                              valid_output_area_tf[i, 1])
                                        for i in range(self.n_dims))]

                loss_tf = tf.reduce_mean(tf.square(diff_tf),
                                         axis=np.arange(1, self.n_dims + 1))
                loss_tf = tf.reduce_mean(loss_tf * part_weights_tf)

                # a different point-based max-margin loss
                true_pos_tf = tf.placeholder(tf.float32, (self.n_parts,
                                                          self.n_dims))

                downsampling_tf = tf.convert_to_tensor(self.downsampling,
                                                       dtype=tf.float32)
                out_true_idx_tf = tf.cast((true_pos_tf + tf.constant(.5))
                                          // downsampling_tf, tf.int32)

                terms = []

                mesh = np.meshgrid(
                    *[(np.arange(s) + 0.5) * d - 0.5 for s, d in
                      zip(self.output_shape, self.downsampling)],
                    indexing='ij')
                mesh_tf = tf.convert_to_tensor(np.array(mesh),
                                               dtype=tf.float32)

                for idx in range(self.n_parts):
                    tp_tf = tf.reshape(true_pos_tf[idx],
                                       (self.n_dims,) + (1,) * self.n_dims)
                    margin_tf = tf.sqrt(tf.reduce_sum(tf.square(
                        mesh_tf - tp_tf), axis=0))
                    margin_tf = tf.minimum(margin_tf / tf.constant(10.),
                                           tf.constant(1.))
                    margin_tf *= tf.constant(100.)

                    hm_tf = self.output_tf[0, idx]
                    tp_value_tf = hm_tf[tuple(out_true_idx_tf[idx, i]
                                              for i in range(self.n_dims))]
                    term_tf = tf.square(tf.maximum(
                        hm_tf - tp_value_tf + margin_tf, tf.constant(0.)))
                    term_tf = tf.reduce_mean(term_tf) * part_weights_tf[idx]

                    terms.append(tf.where(tf.is_nan(true_pos_tf[idx, 0]),
                                          tf.constant(np.nan),
                                          term_tf))

                terms = tf.stack(terms)
                terms = tf.boolean_mask(terms,
                                        tf.logical_not(tf.is_nan(terms)))

                loss2_tf = tf.reduce_mean(terms)

                if relative_loss:
                    loss_tf = loss2_tf

                global_step_tf = tf.Variable(0, trainable=False)
                increase_global_step_tf = tf.assign_add(global_step_tf, 1)

                learning_rate = 1.0e-5
                # learning_rate_tf = tf.train.exponential_decay(
                #     learning_rate, global_step_tf,
                #     int(np.ceil(len(train_images) / batch_size) * 15), 0.95,
                #     staircase=True)
                learning_rate_tf = learning_rate

                # trainable tensors and our optimizer
                trainable_vars = tf.trainable_variables()
                with tf.name_scope('adam_optimizer'):
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=learning_rate_tf)

                    accum_gradients_tf = [
                        tf.Variable(tf.zeros_like(tv.initialized_value()),
                                    trainable=False)
                        for tv in trainable_vars]
                    zero_accum_gradients_tf = [agv.assign(tf.zeros_like(agv))
                                               for agv in accum_gradients_tf]
                    gradients_tf = optimizer.compute_gradients(loss_tf,
                                                               trainable_vars)
                    # needed for e.g. batch norm
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        add_gradients_tf = [agv.assign_add(gv[0]) for agv, gv
                                            in zip(accum_gradients_tf,
                                                   gradients_tf)]

                    update_weights_tf = optimizer.apply_gradients(
                        list(zip(accum_gradients_tf, trainable_vars)))

            optimizer_variables_tf = [optimizer.get_slot(var, name)
                                      for name in optimizer.get_slot_names()
                                      for var in trainable_vars]
            # hack to get Adam specific variables (beta)
            optimizer_variables_tf.extend(optimizer._non_slot_dict.values())
            optimizer_variables_tf = [var for var in optimizer_variables_tf
                                      if var is not None]

            self.session.run(tf.variables_initializer(
                optimizer_variables_tf + [global_step_tf]))

            rng = np.random.RandomState(42)
            part_counter = np.zeros(self.n_parts, dtype=int)

            # generate, possible random/artificial, training samples
            def generate_training_sample(image):
                nonlocal part_counter

                data = image.data
                data_size = data.shape[:self.n_dims]

                all_pos = [obj.position[::-1] for obj in image.parts.values()]
                min_pos = np.min(all_pos, axis=0)
                max_pos = np.max(all_pos, axis=0)

                # constrains the region where to randomly place the origin
                # in order to have landmarks in the patch
                # ALT:
                # min_origin = np.maximum((min_pos - self.input_shape * .6),
                #                         0).astype(int)
                # max_origin = np.minimum((max_pos - self.input_shape * .4),
                #                         data_size
                #                         - self.input_shape).astype(int)
                min_origin = (min_pos - self.input_shape * .5)
                max_origin = (max_pos - self.input_shape * .5)

                # however, we have to use a different strategy if the data
                # is smaller than the patch
                # ALT: rng.randint(ds - fs, 0 + 1) if mi > ma else
                origin = np.array([rng.randint(mi, ma + 1)
                                   for mi, ma, ds, fs
                                   in zip(min_origin, max_origin, data_size,
                                          self.input_shape)])

                # valid region inside the original data volume
                data_start = np.maximum(origin, 0)
                data_end = np.minimum((origin + self.input_shape), data_size)
                data_slice = tuple(slice(*a) for a in zip(data_start,
                                                          data_end))

                data_sliced = data[data_slice]
                if len(data_sliced.shape) == self.n_dims:
                    assert self.n_channels == 1
                    data_sliced.shape += (1,)

                # we also add randomly a bit of Gaussian noise
                # if rng.rand() < .25:
                #     data_sliced = data_sliced \
                #         + rng.normal(size=data_sliced.shape)
                # sigma seems too large, check that!

                # the corresponding region in our new patch
                new_data_slice = tuple(slice(s - o, (s - o) + e - s)
                                       for s, o, e in
                                       zip(data_start, origin, data_end))

                new_data = np.zeros(tuple(self.input_shape)
                                    + (self.n_channels,), dtype=data.dtype)
                new_data[new_data_slice] = data_sliced
                new_data = new_data.astype(np.float32)

                # this is the valid region in the output heatmap, that is
                # actually caused by image input rather than padded area around
                valid_output_area = np.array([
                    (max(r.start // d, 0), min(int(np.ceil(r.stop / d)), o))
                    for r, d, o in zip(new_data_slice, self.downsampling,
                                       self.output_shape)])

                # the new true positions inside the patch
                true_pos = np.full((self.n_parts, self.n_dims), np.nan)
                for part_idx, part in enumerate(self.parts):
                    try:
                        pos = image.parts[part].position[::-1] - origin
                    except KeyError:
                        continue

                    if (pos >= 0).all() \
                            and (pos <= self.input_shape - 1).all():
                        true_pos[part_idx] = pos

                # class specific weights
                part_counter = part_counter + ~np.isnan(true_pos[:, 0])

                # part_weights = 1 / part_counter
                # part_weights = part_weights / part_weights.min()
                # part_weights[np.isnan(true_pos[:, 0])] = 1.
                part_weights = np.ones(part_counter.size)

                # we use elastic transformation by chance
                if rng.rand() < elastic_transform:
                    valid_true_pos = ~np.isnan(true_pos[:, 0])
                    new_data, new_true_data = transform_elastically(
                        new_data, positions=true_pos[valid_true_pos],
                        alpha=200, sigma=15, seed=None)
                    # TODO: alpha and sigma should be probably scaled by mm res
                    true_pos[valid_true_pos] = new_true_data

                if self.channels_first:
                    new_data = new_data.transpose(
                        [self.n_dims] + list(range(self.n_dims)))

                # build the corresponding true output map
                true_output = np.zeros((self.n_parts,)
                                       + tuple(self.output_shape),
                                       dtype=np.float32)

                # radius is either 10px or 10mm if spacing is given
                radius = 10.
                if image.spacing is not None:
                    radius = radius / image.spacing[::-1]
                radius = radius / self.downsampling

                for part_idx, part in enumerate(self.parts):
                    pos = (true_pos[part_idx] + 0.5) / self.downsampling - 0.5

                    if np.isnan(pos[0]):
                        continue

                    place_gaussian(true_output[part_idx], pos, radius,
                                   border_value=0.01, scale=self.target_value)

                if rng.rand() < flip_chance:
                    new_data = np.flip(new_data, axis=flip_axis)
                    for i in range(self.n_parts):
                        true_output[i] = np.flip(true_output[i],
                                                 axis=flip_axis)
                    last_index = true_output.shape[flip_axis + 1] - 1
                    true_pos[flip_axis] = last_index - true_pos[flip_axis]
                    valid_output_area[flip_axis] = last_index + 1 - \
                        valid_output_area[flip_axis][::-1]

                # value augmentation
                if rng.rand() < value_chance:
                    scale = rng.uniform(low=1 - value_scale,
                                        high=1 + value_scale,
                                        size=new_data.shape)
                    shift = rng.uniform(low=-value_shift, high=value_shift,
                                        size=new_data.shape)
                    new_data = (new_data + shift) * scale

                return new_data, true_output, true_pos, valid_output_area, \
                    part_weights

            # computation of the actual important metrics
            def evaluate(images):
                first_results = []
                best_results = []

                for image in images:
                    all_candidates = self.propose_candidates(image)

                    # sometimes during training there is no candidate
                    for part in self.parts:
                        if len(all_candidates[part]) == 0:
                            all_candidates[part] = \
                                np.array([(0,) * self.n_dims])

                    true_pos = criterion.true_pos(image, self.parts,
                                                  self.n_dims)

                    first_pos = np.copy(true_pos)
                    for i, pos in enumerate(first_pos):
                        if np.isnan(pos).any():
                            continue
                        first_pos[i] = all_candidates[self.parts[i]][0]

                    best_pos = criterion.best_configuration(
                        image, self.parts, all_candidates)

                    first_result = criterion.evaluate(
                        image, self.parts, first_pos, true_pos)
                    best_result = criterion.evaluate(
                        image, self.parts, best_pos, true_pos)

                    first_results.append(first_result)
                    best_results.append(best_result)

                def comp(results):
                    def mean(f):
                        return np.mean(list(chain.from_iterable(
                            f(x) for x in results)))

                    rate = mean(lambda r: r.correct[~r.true_unknown])
                    err = mean(lambda r: (r.dist if r.dist_mm is None else
                                          r.dist_mm)[~r.true_unknown])
                    all_rate = mean(lambda r:
                                    [r.correct[~r.true_unknown].all()])

                    return rate * 100, err, all_rate * 100

                return comp(best_results) + comp(first_results)

            losses = deque([], maxlen=int(len(train_images) / batch_size))

            best_values = (0, np.inf, 0, np.inf)
            best_weights = None
            best_evaluation_idx = 0

            train_start = time.time()
            train_time = 0

            try:
                # the actual training loop starts here
                for iteration, (epoch, batch_indices) in enumerate(
                        batchize(np.arange(len(train_images)),
                                 batch_size=batch_size)):

                    t1 = time.time()

                    self.session.run(zero_accum_gradients_tf)

                    iter_loss = 0

                    for image_idx in batch_indices:
                        input_, true_output, true_pos, valid_output_area, \
                            part_weights = \
                            generate_training_sample(train_images[image_idx])

                        input_.shape = (1,) + input_.shape
                        true_output.shape = (1,) + true_output.shape

                        feed_dict = {true_pos_tf: true_pos,
                                     true_output_tf: true_output,
                                     valid_output_area_tf: valid_output_area,
                                     part_weights_tf: part_weights,
                                     self.input_tf: input_,
                                     self.is_training_tf: True}

                        if self.dropout_rate is not None:
                            feed_dict[self.dropout_rate_tf] = self.dropout_rate

                        _, loss = self.session.run(
                            [add_gradients_tf, loss_tf],
                            feed_dict=feed_dict)

                        iter_loss += loss / batch_size

                    losses.append(iter_loss)

                    self.session.run(update_weights_tf)
                    self.session.run(increase_global_step_tf)

                    train_time += time.time() - t1

                    log.debug('Iteration %i/%.2f: loss=%f iter-loss=%f',
                              iteration + 1, (iteration + 1) * batch_size
                              / len(train_images), np.mean(losses), iter_loss)

                    # evaluation time!
                    if iteration > 0 and iteration % metrics_every == 0:
                        log.debug('Computing evaluation metrics...')

                        evaluation_idx = iteration // metrics_every
                        train_results = evaluate(train_images)
                        val_results = evaluate(val_images)

                        if working_dir:
                            for name, images in [('val', val_images),
                                                 ('train', train_images)]:
                                reference_images = select_reference_images(
                                    images, self.potentials)

                                for image, potentials in reference_images:
                                    for pot in potentials:
                                        path = working_dir \
                                            / (f'cnn_progress_{name}/'
                                               + f'{iteration}/{pot}_'
                                                 f'{image.name}.jpg')
                                        path.parent.mkdir(parents=True,
                                                          exist_ok=True)
                                        pot.plot_energies(image, path)

                        for prefix, results in [('Training', train_results),
                                                ('Validation', val_results)]:
                            log.debug(' - %s: best_part_rate=%.1f%% '
                                      'best_err=%.2f best_img_rate=%.1f%% '
                                      'first_part_rate=%.1f%% first_err=%.2f '
                                      'first_img_rate=%.1f%% ',
                                      prefix, *results)

                        total_train_time = time.time() - train_start
                        log.debug(' - Training time: %s (minus eval.: %s)',
                                  format_timespan(total_train_time),
                                  format_timespan(train_time))

                        cur_values = (-val_results[0], val_results[1],
                                      -val_results[3], val_results[4])
                        if cur_values < best_values:
                            log.debug(' ! This is a new best!')
                            best_values = cur_values
                            best_weights = self.get_weights()
                            best_evaluation_idx = evaluation_idx

                        # we stop after five evaluation runs with no impro.
                        if evaluation_idx - best_evaluation_idx == 5:
                            log.info('No improvement for 5 consecutive '
                                     'evaluation runs, stopping now.')
                            break

                        # max over all training time
                        if total_train_time > max_train_time:
                            log.debug('Trained for more than %s, stopping.',
                                      format_timespan(max_train_time))
                            break

            except KeyboardInterrupt:
                log.warning('Caught keyboard interrupt, using latest best '
                            'weights')

            assert best_weights is not None, 'did not generate a best weights!'

            return best_weights

        self.set_weights(train_weights())

    class Potential(UnaryPotential, LearnablePotentialMixin):
        """A unary potential proxying the output of the multi part CNN.

        Args:
            net: The underlying network.
            index: The index of the final feature map belonging to this
                potential.
            part: Name of the corresponding part.
        """

        def __init__(self, net: 'MultiPartCNN', index: int, part: str) -> None:
            super().__init__(part)

            self.net = net
            """The underlying network."""

            self.index = index
            """Index of the part's feature map."""

        def name(self):
            return 'nclass-cnn-fm' + str(self.index)

        def propose_candidates(self, image: Image) -> Dict[str, np.ndarray]:
            if self.index == 0:
                return self.net.propose_candidates(image)

            return super().propose_candidates(image)

        def compute(self, image: Image, positions: List[np.ndarray],
                    **kwargs) -> Union[float, np.ndarray]:
            position, = positions

            heatmap = self.net.infer(image)[self.index]

            pos = ((position + 0.5) / self.net.downsampling) - 0.5
            assert pos.ndim == 2
            idx = np.round(pos).astype(int)

            # find the positions that are off-grid
            # needs_interp = (np.abs(pos - idx) > 0.02).any(axis=1)

            idx = np.minimum(np.maximum(idx, 0), np.array(heatmap.shape) - 1)
            values = heatmap[tuple(idx.T)]
            assert np.isfinite(values).all()

            # TODO: Add interpolation!
            # if self.net.n_dims == 2 and needs_interp.any():
            # from scipy.interpolate import RectBivariateSpline
            #     spline = RectBivariateSpline(np.arange(heatmap.shape[0]),
            #                                  np.arange(heatmap.shape[1]),
            #                                  heatmap,
            #                                  kx=2, ky=2)
            #     values[needs_interp] = spline.ev(*pos[needs_interp].T)

            if values.max() > self.net.target_value:
                log.warning('Got a CNN heatmap value larger than the trained '
                            'target value: %f > %f',
                            values.max(), self.net.target_value)

            values = np.minimum(np.maximum(values, 0), self.net.target_value)
            values /= self.net.target_value
            assert (values >= 0).all() and (values <= 1).all()

            with np.errstate(divide='ignore'):
                values = -np.log(values)

            return values

        def feed_values(self, feed_dict: Dict, image: Image) -> None:
            if self.net.input_tf in feed_dict:
                return

            feed_dict[self.net.input_tf] = self.net.create_input(image.data)

        def auxiliary_loss_tf(self):
            if self.index == 0 and self.net.l2_norm > 0:
                return self.net.weights_l2_tf * tf.constant(self.net.l2_norm)

            return super().auxiliary_loss_tf()

        def compute_tf(self, positions_tf: List[tf.Tensor]) -> tf.Tensor:
            position_tf, = positions_tf

            # TODO: Interpolated!!
            downsampling_tf = tf.constant(self.net.downsampling,
                                          dtype=tf.float64)
            pos_tf = tf.cast((position_tf + tf.constant(0.5, dtype=tf.float64))
                             / downsampling_tf, tf.int32)

            energymap_tf = self.net.normalized_output_tf[0, self.index]
            return tf.gather_nd(energymap_tf, pos_tf)

        def changed(self) -> None:
            self.net._cache_image = None
            self.net._cache_output = None

        def get_parameters(self) -> Optional[Dict[str, np.ndarray]]:
            if self.index == 0:
                return self.net.get_weights()

            return None

        def set_parameters(self, parameters: Optional[Dict[str, np.ndarray]]) \
                -> None:
            if self.index == 0:
                self.net.set_weights(parameters)


class GaussianVector(BinaryPotential, LearnablePotentialMixin,
                     BasePotentialMixin):
    """



    """
    def __init__(self, session, parts: List[str], n_dims: int = 2, plot=False,
                 **kwargs) -> None:
        super().__init__(parts, **kwargs)
        assert n_dims in (2, 3)

        self.n_dims = n_dims

        self.plot = plot

        self.session = session

        self.mu_tf = None
        self.sigma_tf = None

        self.create_graph(initialize_weights=True)

    def create_graph(self, initialize_weights: bool = True):
        with self.session.graph.as_default():
            with tf.variable_scope(f'gaussian_vector_{"-".join(self.parts)}'):
                self.mu_tf = tf.get_variable(
                    'mu', shape=(self.n_dims,), dtype=tf.float64,
                    initializer=tf.constant_initializer(0.)
                    if initialize_weights else None)
                self.sigma_tf = tf.get_variable(
                    'sigma', shape=(self.n_dims, self.n_dims),
                    dtype=tf.float64, initializer=tf.initializers.identity()
                    if initialize_weights else None)

        if initialize_weights:
            self.session.run(tf.variables_initializer(
                [self.mu_tf, self.sigma_tf]))

    def __getstate__(self):
        state = dict(self.__dict__)

        # drop unpicklable and unneeded things
        state['mu_tf'] = None
        state['sigma_tf'] = None
        state['session'] = None

        # get weights for later restore
        state['_weights'] = self.get_parameters()

        return state

    def __setstate__(self, state):
        weights = state.pop('_weights')

        self.__dict__ = state

        # we make use of the default session
        self.session = tf.get_default_session()
        assert self.session is not None, 'please set a default TF session'

        self.create_graph(initialize_weights=False)
        self.set_parameters(weights)

    def train(self, train_images: List[Image]) -> None:
        pos_a, pos_b = self.positions(train_images)
        vectors = pos_b - pos_a

        mean = np.mean(vectors, axis=0)
        cov = np.cov(vectors, rowvar=False)

        self.mu_tf.load(mean, self.session)
        self.sigma_tf.load(cov, self.session)

        log.info('Estimated a Gaussian for the vector between %s and '
                 '%s using %i samples: mean=%s', *self.parts, len(vectors),
                 mean[::-1])

    def compute_tf(self, positions_tf: List[tf.Tensor]) -> tf.Tensor:
        pos_a_tf, pos_b_tf = positions_tf
        # FIXME: single pos! Is that acutally an issue here? think not
        # FIXME: SPACING!

        vec_tf = pos_b_tf - pos_a_tf
        delta_tf = vec_tf - self.mu_tf

        return tf.reduce_sum(tf.multiply(delta_tf @ self.sigma_tf,
                                         delta_tf), 1)

    def compute(self, image: Image, positions: List[np.ndarray],
                **kwargs) -> Union[float, np.ndarray]:
        pos_a, pos_b = positions

        # FIXME: SPACING!

        mu, sigma = self.session.run([self.mu_tf, self.sigma_tf])

        vec = pos_b - pos_a
        delta = vec - mu

        return np.sum((delta @ sigma) * delta, axis=1)

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return tuple(self.session.run([self.mu_tf, self.sigma_tf]))

    def set_parameters(self, parameters: Tuple[np.ndarray, np.ndarray]) \
            -> None:
        mu, sigma = parameters
        self.mu_tf.load(mu, self.session)
        self.sigma_tf.load(sigma, self.session)

"""Provides common **evaluation metrics**.

Localization result evaluation
------------------------------

A localization result can be evaluated using one of the `Criterion`
subclasses, producing a `LocResult` with all kinds of interesting metrics.

Currently, there is only the `MaxDistance` criterion, which should drive most
common needs.

Metrics
-------

In order to simplify the logging and plotting of common metrics, we use
a dedicated `Metric` class (`PercentMetric`). This class is used to define
and specify metrics, common ones are stored in `Metrics`.
"""


import hiwi
import logging
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from colorama import Fore
from collections import OrderedDict
from copy import copy
from enum import Enum
from hiwi import Image, guess_image_shape
from hiwi.plot import distinct_colors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse
from pathlib import Path
from sklearn.utils.extmath import cartesian
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


log = logging.getLogger(__name__)


class PartCase(str, Enum):
    """The potential result cases."""
    LOCALIZED = 'localized'
    MISLOCALIZED = 'mislocalized'
    FP = 'fp'
    FN = 'fn'
    TN = 'tn'


class LocResult:
    """A localization result including evaluation metrics.

    This should not be created directly, but rather by a `Criterion`, which is
    able to populate the `correct` and `error` fields. All other fields are
    safe to use though.

    Args:
        true_pos: The true positions in voxel, `NaN` indicating absence.
        pred_pos: The predicted positions in voxel, `NaN` indicating absence.
        spacing: An optional spacing.
    """
    def __init__(self, true_pos: np.ndarray, pred_pos: np.ndarray,
                 spacing: Optional[np.ndarray] = None):
        self.true_pos = true_pos
        """The true position, `NaN`s indicating a missing position."""

        self.pred_pos = pred_pos
        """The predicted position, `NaN`s indicating a missing position."""

        self.true_pos_mm = None if spacing is None else true_pos * spacing
        """The true position, in mm if spacing is given, else `None`."""

        self.pred_pos_mm = None if spacing is None else pred_pos * spacing
        """The predicted position, in mm if spacing is given, else `None`."""

        self.dist = hiwi.dist(true_pos, pred_pos)
        """Euclidean distances, `NaN` if any position is `NaN`."""

        self.dist_mm = None if spacing is None else \
            hiwi.dist(self.true_pos_mm, self.pred_pos_mm)
        """Euclidean distances in mm if spacing is given, else `None`."""

        self.true_unknown = np.isnan(self.true_pos[:, 0])
        """Boolean array indicating the missing true positions."""

        self.pred_unknown = np.isnan(self.pred_pos[:, 0])
        """Boolean array indicating the missing predicted positions."""

        self.correct: np.ndarray = None
        """Part-specific boolean array that specifies whether the part was
        correctly localized or not.
        """

        self.error: np.ndarray = None
        """A part-specific error value."""

        self.order: np.ndarray = np.arange(len(true_pos))
        """The order for part-specific string logging."""

    @property
    def valid_dist(self):
        """Filters `dist` for true positive results."""
        return self.dist[~np.isnan(self.dist)]

    @property
    def valid_dist_mm(self):
        """Filters `dist_mm` for true positive results."""
        return None if self.dist_mm is None \
            else self.dist_mm[~np.isnan(self.dist_mm)]

    @property
    def cases(self) -> List[PartCase]:
        """A list describing the different result cases."""
        cases = []
        for c, tu, pu in zip(self.correct, self.true_unknown,
                             self.pred_unknown):
            if c:
                case = PartCase.TN if tu else PartCase.LOCALIZED
            elif tu and not pu:
                case = PartCase.FP
            elif not tu and pu:
                case = PartCase.FN
            elif not tu and not pu:
                case = PartCase.MISLOCALIZED
            else:
                raise RuntimeError('should never happen')
            cases.append(case)
        return cases

    def __str__(self):
        return ('correct = {loc_per:5.1f}% ({loc_abs:{digits}d}/'
                '{total:{digits}d}), mean_dist = {error}, '
                'parts = {parts}').format(
                digits=digits(len(self.correct)),
                loc_per=self.correct.sum() / len(self.correct) * 100,
                loc_abs=self.correct.sum(), total=len(self.correct),
                error=format_error(self.valid_dist.mean(),
                                   None if self.dist_mm is None else
                                   self.valid_dist_mm.mean()),
                parts=''.join(format_correct(self.correct[i],
                                             np.isnan(self.true_pos[i, 0]),
                                             not np.isnan(self.pred_pos[i, 0]))
                              for i in self.order))


class SetLocResult:
    """Provides statistics over a set of localization results."""
    def __init__(self, results: List[LocResult]) -> None:
        self.results = results
        """The individual localization results."""

        self.correct = np.array([r.correct for r in results])
        """Array indicating which result part was correct."""

        self.fully_correct = self.correct.all(axis=1)
        """Array indicating which result was fully correct."""

        self.avg_dist = np.mean(sum([r.valid_dist.tolist()
                                     for r in results], []))
        """Average localization distance."""

        self.avg_dist_mm = np.mean(sum([r.valid_dist_mm.tolist()
                                        for r in results], [])) \
            if results[0].dist_mm is not None else None
        """Average localization distance in mm, if available."""

    def __str__(self):
        return ('correct = {loc_per:5.1f}% ({loc_abs:{digits}d}/'
                '{total:{digits}d}), fully correct = {floc_per:5.1f}% '
                '({floc_abs:{digits2}d}/{total2:{digits2}d}), '
                'mean_dist = {error}').format(
                digits=digits(self.correct.size),
                loc_per=self.correct.mean() * 100,
                loc_abs=self.correct.sum(), total=self.correct.size,
                digits2=digits(self.fully_correct.size),
                floc_per=self.fully_correct.mean() * 100,
                floc_abs=self.fully_correct.sum(),
                total2=self.fully_correct.size,
                error=format_error(self.avg_dist, self.avg_dist_mm))


class Criterion(ABC):
    """Evaluates a localization result.

    This class allows to drive the optimization w.r.t. a specific error
    and to compute evaluation metrics.

    Not that `NaN` values are used to indicate the absence of a position,
    i.e., the "unknown" label.
    """

    @abstractmethod
    def evaluate(self, image: Image, parts: List[str], pred_pos: np.ndarray,
                 true_pos: Optional[np.ndarray] = None) -> LocResult:
        """Evaluates a localization result
        Args:
            image: The image we want to evaluate results for.
            parts: The parts we are providing results for.
            pred_pos: The list of predicted positions (2D).
            true_pos: Optional list of respective true positions.
        Return: A localization result with errors and indication of correct.
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, image: Image, part: str, pred_pos: np.ndarray) \
            -> Union[float, np.ndarray]:
        """Computes the error for one or multiple part-specific predictions.

        Args:
            image: The image we want to compute an error for.
            part: The part we want to compute an error for.
            pred_pos:  One (1D) or multiple (2D) prediction results.

        Return: One or multiple errors.
        """
        raise NotImplementedError

    @abstractmethod
    def correct_configurations(self, image: Image, parts: List[str],
                               candidates: Dict[str, np.ndarray]) \
            -> np.ndarray:
        """Finds all correct configurations in a set of candidates.

        Uses `NaN` to indicate the absence of a part's position.

        Args:
            image: The image we want to take the annotations from.
            parts: The list of parts we are interested in.
            candidates: The available candidates for each part.

        Returns: The (potentially empty) list of all correct configurations.
        """
        raise NotImplementedError

    @staticmethod
    def true_pos(image: Image, parts: List[str],
                 n_dims: Optional[int] = None) -> np.ndarray:
        """Derives the position array from the image annotations.

        Args:
            image: The image we want to take the annotations from.
            parts: The list of parts we are interested in.
            n_dims: Optional image dimensions, if not given, they are derived
                from the annotated positions. Note that this might fail if
                all parts are missing.

        Return:
            A 2D array with [z]yx coordinates in pixel, with `NaN` values
            indicating the absence of the part.
        """
        if n_dims is None:
            n_dims = next(len(obj.position) for obj
                          in image.objects[0].parts.values())

        true_pos = np.full((len(parts), n_dims), np.nan)

        for i, part in enumerate(parts):
            try:
                true_pos[i] = image.objects[0].parts[part].position[::-1]
            except (IndexError, KeyError):
                pass

        return true_pos

    @abstractmethod
    def best_configuration(self, image: Image, parts: List[str],
                           candidates: Dict[str, np.ndarray],
                           ret_indices: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Selects the best configuration out of a set of candidates.

        Uses `NaN` to indicate the absence of a part's position as well
        as -1 for the corresponding candidate index.

        Args:
            image: The image we want to take the annotations from.
            parts: The list of parts we are interested in.
            candidates: The available candidates for each part.
            ret_indices: Whether to return the indices of the best candidates
                as well.

        Returns:
            Either just the best configuration or the best configuration plus
            the vector of the corresponding indices (state).
        """

        # TODO: We could provide a default implementation using
        # TODO: correct_configurations i guess
        raise NotImplementedError


class MaxDistance(Criterion):
    """Simple Euclidean-distance-based thresholding criterion.

    The error defaults to the (potentially capped) Euclidean distance. If a
    classification error occurred (e.g., `NaN` vs position), a dedicated
    error value is used.

    Args:
        max_dist: The maximal distance after which a prediction is treated as
            incorrect.
        dist_cap: Used to cap the error distance.
        classification_error: A dedicated value for classification errors,
            if not given, this defaults to `max_dist`.
        use_mm: Whether to compute distances in mm or pixel.
    """
    def __init__(self, max_dist: float, dist_cap: float = np.inf,
                 classification_error: Optional[float] = None,
                 use_mm: bool = False) -> None:
        assert dist_cap >= max_dist

        self.max_dist = max_dist
        """The maximally allowed distance to treat a prediction as correct."""

        self.dist_cap = dist_cap
        """Distance errors are capped to this value."""

        self.classification_error = \
            max_dist if classification_error is None else classification_error
        """Dedicated error value to be used for classification errors."""

        self.use_mm = use_mm
        """Whether to compute distances in mm or pixel."""

    def evaluate(self, image: Image, parts: List[str], pred_pos: np.ndarray,
                 true_pos: Optional[np.ndarray] = None) -> LocResult:
        if true_pos is None:
            true_pos = Criterion.true_pos(image, parts)

        spacing = image.spacing
        if spacing is not None:
            spacing = spacing[::-1]

        result = LocResult(true_pos, pred_pos, spacing)
        result.correct = self.compute_correct(result)
        result.error = self.compute_error(result)

        return result

    def error(self, image: Image, part: str, pred_pos: np.ndarray) \
            -> Union[float, np.ndarray]:
        is_one = pred_pos.ndim == 1
        if is_one:
            pred_pos = pred_pos.reshape((1, pred_pos.size))

        try:
            true_pos = image.objects[0].parts[part].position[::-1]
        except KeyError:
            true_pos = np.full(pred_pos.shape[1], np.nan)
        true_pos.shape = (1, len(true_pos))

        spacing = image.spacing
        if spacing is not None:
            spacing = spacing[::-1]

        error = self.compute_error(LocResult(true_pos, pred_pos, spacing))

        if is_one:
            error = error[0]

        return error

    def correct_configurations(self, image: Image, parts: List[str],
                               candidates: Dict[str, np.ndarray]) \
            -> np.ndarray:
        n_parts = len(parts)
        n_dims = next(len(obj.position) for obj
                      in image.objects[0].parts.values())

        bests = []
        errors = []

        for i, part in enumerate(parts):
            if part in image.parts:
                cands = candidates[part]
                true_pos = image.parts[part].position[::-1]

                if self.use_mm:
                    cands_mm = cands * image.spacing[::-1]
                    true_pos_mm = true_pos * image.spacing[::-1]
                    dists = hiwi.dist(true_pos_mm, cands_mm)
                else:
                    dists = hiwi.dist(true_pos, cands)

                bests.append([i for i, dist in enumerate(dists)
                              if dist <= self.max_dist])
                errors.append(dists)
            else:
                bests.append([-1])
                errors.append([-1])

        n_configs = np.prod([len(x) for x in bests])
        if n_configs > 10_000:
            log.warning('There are %i correct configurations!', n_configs)

        if n_configs == 0:
            return np.array([]).reshape((0, n_parts, n_dims))

        configurations = np.full((n_configs, n_parts, n_dims), np.nan)
        error_sum = np.zeros(n_configs)

        for part_idx, cand_indices in enumerate(cartesian(bests).T):
            if cand_indices[0] != -1:
                cands = candidates[parts[part_idx]][cand_indices]
                configurations[:, part_idx, :] = cands
                error_sum += errors[part_idx][cand_indices]

        return configurations[np.argsort(error_sum)]

    def best_configuration(self, image: Image, parts: List[str],
                           candidates: Dict[str, np.ndarray],
                           ret_indices: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        n_parts = len(parts)
        n_dims = next(len(obj.position) for obj
                      in image.objects[0].parts.values())

        best_idx = np.full(n_parts, -1, dtype=int)
        best_cands = np.full((n_parts, n_dims), np.nan)

        for i, part in enumerate(parts):
            if part not in image.parts:
                continue

            true_pos = image.parts[part].position[::-1]
            cands = candidates[part]

            if self.use_mm:
                cands = cands * image.spacing[::-1]
                true_pos = true_pos * image.spacing[::-1]

            dists = hiwi.dist(true_pos, cands)

            # do not use cands here because it might be in mm
            best_idx[i] = np.argmin(dists)
            best_cands[i] = candidates[part][best_idx[i]]

        if ret_indices:
            return best_cands, best_idx

        return best_cands

    def compute_correct(self, result: LocResult) -> np.ndarray:
        dist = result.dist_mm if self.use_mm else result.dist

        # we eventually compare to NaN here, which is safe
        with np.errstate(invalid='ignore'):
            loc_correct = dist <= self.max_dist

        return (result.true_unknown & result.pred_unknown) \
            | (~result.true_unknown & ~result.pred_unknown & loc_correct)

    def compute_error(self, result: LocResult) -> np.ndarray:
        dist = result.dist_mm if self.use_mm else result.dist

        error = np.minimum(dist, self.dist_cap)
        error[result.true_unknown & result.pred_unknown] = 0
        error[result.true_unknown ^ result.pred_unknown] = \
            self.classification_error
        assert not np.isnan(error).any()

        return error


class Metric:
    """An optimization metric.

    Args:
        name: Name of the metric, these are unique!
        minimize: Whether the metric should be minimized or maximized.
        unit: An optional unit for plotting.
        ylim: Optional bounds for the y-axis.
        plot_kwargs: Additional plotting params to pass to `pyplot.plot()`.
        plot_with: Plots all specified metrics, including this one, in one
            single graph.
    """
    def __init__(self, name: str, minimize: bool = True,
                 unit: Optional[str] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 plot_kwargs: Optional[Dict] = None,
                 plot_with: Optional[List['Metric']] = None):
        self.name = name
        """Unique name of the metric."""

        self.minimize = minimize
        """Whether this metric should be minimized or maximized."""

        self.unit = unit
        """The unit of this metric."""

        self.ylim = ylim
        """Set axis bounds for y-coordinate manually."""

        self.tol = 1e-10
        """Absolute tolerance when doing checks."""

        self.grouped_metrics = set()
        """Other metrics to form groups with during plotting."""

        plot_kwargs_updated = dict(ls='-')
        plot_kwargs_updated.update(plot_kwargs or {})

        self.plot_kwargs = plot_kwargs_updated
        """Additional plotting params to pass to `pyplot.plot()`."""

        Metric.add(self)

        for metric in (plot_with or []):
            self.grouped_metrics.add(metric)
            metric.grouped_metrics.add(self)

    VAL_PREFIX = 'val-'
    """Prefix that indicates whether this was computed on the validation
    or training data.
    """

    BY_NAME = dict()
    """Map of existing metrics."""

    ALL = list()
    """All existing metrics."""

    @classmethod
    def add(cls, metric: 'Metric') -> None:
        assert metric.name not in Metric.BY_NAME, 'metric already exists'
        Metric.BY_NAME[metric.name] = metric
        Metric.ALL = list(Metric.BY_NAME.values())

    @property
    def val(self):
        """Same metric but for validation samples."""
        assert not self.name.startswith(Metric.VAL_PREFIX)

        val_name = Metric.VAL_PREFIX + self.name

        try:
            return Metric.BY_NAME[val_name]
        except KeyError:
            metric = copy(self)
            metric.name = val_name
            Metric.add(metric)
            return metric

    @property
    def org(self):
        """Reference to the original metric, if this is a validation metric."""
        assert self.is_val
        return Metric.BY_NAME[self.name.replace(self.VAL_PREFIX, '')]

    @property
    def is_val(self):
        """Whether this is a validation or training metric."""
        return self.name.startswith(self.VAL_PREFIX)

    def is_better(self, old_value: float, new_value: float) -> bool:
        """Whether the `new_value` is better than the `old_value`."""
        if self.minimize:
            return new_value < old_value - self.tol
        else:
            return new_value > old_value + self.tol

    def is_equal(self, old_value: float, new_value: float) -> bool:
        """Whether the `new_value` is equal to the `old_value`."""
        return old_value - self.tol <= new_value <= old_value + self.tol

    def format(self, value: float) -> str:
        return f'{value:.2f}{self.unit or ""}'

    def __str__(self):
        return self.name


class PercentMetric(Metric):
    """Simple percent metric."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, minimize=False, unit='%', **kwargs)


class ImageResult(NamedTuple):
    """Corresponds to the localization results on a specific image."""

    predicted: LocResult
    """The actual localization results."""

    localizer: Optional[LocResult]
    """Optionally, the results achieved just using the localizer."""

    best: Optional[LocResult]
    """Optionally, the theoretical achievable localization results."""

    best_cands: Optional[np.ndarray]
    """Optionally, indices (1-based) of the best candidates."""

    candidates: Optional[Dict[str, np.ndarray]]
    """Optionally, the available candidates."""


class Metrics(OrderedDict):
    """Common metrics that are generated for all learners."""

    LOSS = Metric('loss')
    """"The normal loss of the optimization function."""

    REL_WEIGHTS_UPDATE = PercentMetric('rel_weights_update')
    """The percentage change of the norm of the weights with respect to the
    average of old and new value.
    """

    REL_UNKNOWN_ENERGIES_UPDATE = PercentMetric('rel_unknown_energies_update',
                                                plot_with=[REL_WEIGHTS_UPDATE])
    """The percentage change of the norm of the unknown energies with respect
    to the average of old and new value.
    """

    ERROR = Metric('error')
    """The objective error, given by the `Criterion` error."""

    REL_ERROR = Metric('rel_error')
    """The relative objective error w.r.t. the best achievable result, thus,
    this metric allows to go towards 0 despite still erroneous results.
    """

    BEST_CANDIDATE = Metric('best_candidate')
    """The position of the best candidate, i.e., ideally it should be 1 or very
    low to indicate good performance.
    """

    CORRECT_PARTS = PercentMetric('correct_parts',
                                  plot_kwargs=dict(c='C0'))
    """Amount of 'correct' parts as defined by the `Criterion`."""

    CORRECT_PARTS_LOCALIZER = PercentMetric('correct_parts_localizer',
                                            plot_with=[CORRECT_PARTS],
                                            plot_kwargs=dict(ls='--', c='C0'))
    """Amount of 'correct' parts as defined by the `Criterion` when using
    the first best candidate (i.e., just using the localizer).
    """

    CORRECT_PARTS_BOUND = PercentMetric('correct_parts_bound',
                                        plot_with=[CORRECT_PARTS],
                                        plot_kwargs=dict(ls=':', c='C0'))
    """The upper bound of 'correct' parts created by a limited set of
    candidate positions.
    """

    CORRECT_IMAGES = PercentMetric('correct_images',
                                   plot_kwargs=dict(c='C0'))
    """Amount of fully 'correct' images, i.e., all parts correct, as defined by
    the `Criterion`.
    """

    CORRECT_IMAGES_LOCALIZER = PercentMetric('correct_images_localizer',
                                             plot_with=[CORRECT_IMAGES],
                                             plot_kwargs=dict(ls='--', c='C0'))
    """Amount of fully 'correct' images, i.e., all parts correct, as defined by
    the `Criterion` when using the first best candidates (i.e., just using the
    localizer).
    """

    CORRECT_IMAGES_BOUND = PercentMetric('correct_images_bound',
                                         plot_with=[CORRECT_IMAGES],
                                         plot_kwargs=dict(ls=':', c='C0'))
    """The upper bound of 'correct' images created by a limited set of
    candidate positions.
    """

    AVG_LOC_DIST = Metric('avg_loc_dist', unit='px',
                          plot_kwargs=dict(c='C0'))
    """The average localization distance in pixel/voxel."""

    AVG_LOC_DIST_LOCALIZER = Metric('avg_loc_dist_localizer', unit='px',
                                    plot_with=[AVG_LOC_DIST],
                                    plot_kwargs=dict(ls='--', c='C0'))
    """The average localization distance in pixel/voxel when using the first
    best candidates (i.e., just using the localizer).
    """

    AVG_LOC_DIST_BOUND = Metric('avg_loc_dist_bound', unit='px',
                                plot_with=[AVG_LOC_DIST],
                                plot_kwargs=dict(ls=':', c='C0'))
    """Lower bound of the average localization distance in pixel/voxel created
    by a limited set of candidate positions.
    """

    AVG_LOC_DIST_MM = Metric('avg_loc_dist_mm', unit='mm',
                             plot_kwargs=dict(c='C0'))
    """The average localization distance in mm."""

    AVG_LOC_DIST_MM_LOCALIZER = Metric('avg_loc_dist_mm_localizer', unit='mm',
                                       plot_with=[AVG_LOC_DIST_MM],
                                       plot_kwargs=dict(ls='--', c='C0'))
    """The average localization distance in mm when using the first
    best candidates (i.e., just using the localizer).
    """

    AVG_LOC_DIST_MM_BOUND = Metric('avg_loc_dist_mm_bound', unit='mm',
                                   plot_with=[AVG_LOC_DIST_MM],
                                   plot_kwargs=dict(ls=':', c='C0'))
    """Lower bound of the average localization distance in mm created by a
    limited set of candidate positions.
    """

    USED_POTENTIALS = Metric('used_potentials', unit='#')
    """Number of effectively used potentials, i.e., where the potential
    weight is larger than 0.
    """

    TRUE_POSITIVE = PercentMetric('true_positive',
                                  plot_kwargs=dict(ls='-'))
    """Amount of correctly detected and localized parts."""

    TRUE_NEGATIVE = PercentMetric('true_negative',
                                  plot_kwargs=dict(ls='--'))
    """Amount of missing parts correctly detected as being missing."""

    FALSE_POSITIVE = PercentMetric('false_positive',
                                   plot_kwargs=dict(ls=(0, (3, 5, 1, 5))))
    """Amount of missing parts incorrectly detected."""

    FALSE_NEGATIVE = PercentMetric('false_negative',
                                   plot_kwargs=dict(ls=':'))
    """Amount of present parts incorrectly detected as being missing."""

    MISLOCALIZED_POSITIVE = PercentMetric('mislocalized_positive',
                                          plot_kwargs=dict(ls='-.'),
                                          plot_with=[TRUE_POSITIVE,
                                                     TRUE_NEGATIVE,
                                                     FALSE_POSITIVE,
                                                     FALSE_NEGATIVE])
    """Amount of present parts correctly detected but incorrectly localized."""

    ID_RATE = PercentMetric('id_rate')
    """Id.Rate as defined by Glocker et al."""

    @staticmethod
    def compute(results: List[ImageResult]) -> 'Metrics':
        """Computes all possible metrics given the localization results."""
        vals = Metrics()
        vals[Metrics.ERROR] = 0
        vals[Metrics.REL_ERROR] = 0
        vals[Metrics.BEST_CANDIDATE] = []
        vals[Metrics.CORRECT_PARTS] = []
        vals[Metrics.CORRECT_PARTS_LOCALIZER] = []
        vals[Metrics.CORRECT_PARTS_BOUND] = []
        vals[Metrics.CORRECT_IMAGES] = []
        vals[Metrics.CORRECT_IMAGES_LOCALIZER] = []
        vals[Metrics.CORRECT_IMAGES_BOUND] = []
        vals[Metrics.AVG_LOC_DIST] = []
        vals[Metrics.AVG_LOC_DIST_LOCALIZER] = []
        vals[Metrics.AVG_LOC_DIST_BOUND] = []
        vals[Metrics.AVG_LOC_DIST_MM] = []
        vals[Metrics.AVG_LOC_DIST_MM_LOCALIZER] = []
        vals[Metrics.AVG_LOC_DIST_MM_BOUND] = []
        vals[Metrics.TRUE_POSITIVE] = []
        vals[Metrics.MISLOCALIZED_POSITIVE] = []
        vals[Metrics.FALSE_NEGATIVE] = []
        vals[Metrics.TRUE_NEGATIVE] = []
        vals[Metrics.FALSE_POSITIVE] = []
        vals[Metrics.ID_RATE] = []

        for img_result in results:
            result = img_result.predicted
            result_localizer = img_result.localizer
            result_best = img_result.best

            vals[Metrics.ERROR] += result.error.sum()
            vals[Metrics.REL_ERROR] += (result.error -
                                        result_best.error).sum()

            if img_result.best_cands is not None:
                vals[Metrics.BEST_CANDIDATE].extend(img_result.best_cands)

            vals[Metrics.CORRECT_PARTS].extend(result.correct)
            if result_localizer:
                vals[Metrics.CORRECT_PARTS_LOCALIZER].extend(
                    result_localizer.correct)
            if result_best:
                vals[Metrics.CORRECT_PARTS_BOUND].extend(result_best.correct)

            vals[Metrics.CORRECT_IMAGES].append(result.correct.all())
            if result_localizer:
                vals[Metrics.CORRECT_IMAGES_LOCALIZER].append(
                    result_localizer.correct.all())
            if result_best:
                vals[Metrics.CORRECT_IMAGES_BOUND].append(
                    result_best.correct.all())

            vals[Metrics.AVG_LOC_DIST].extend(result.valid_dist)
            if result_localizer:
                vals[Metrics.AVG_LOC_DIST_LOCALIZER].extend(
                    result_localizer.valid_dist)
            if result_best:
                vals[Metrics.AVG_LOC_DIST_BOUND].extend(result_best.valid_dist)

            vals[Metrics.TRUE_POSITIVE].extend(
                ~result.true_unknown & ~result.pred_unknown & result.correct)
            vals[Metrics.MISLOCALIZED_POSITIVE].extend(
                ~result.true_unknown & ~result.pred_unknown & ~result.correct)
            vals[Metrics.FALSE_NEGATIVE].extend(
                ~result.true_unknown & result.pred_unknown)
            vals[Metrics.TRUE_NEGATIVE].extend(
                result.true_unknown & result.pred_unknown)
            vals[Metrics.FALSE_POSITIVE].extend(
                result.true_unknown & ~result.pred_unknown)

            if result.dist_mm is not None:
                vals[Metrics.AVG_LOC_DIST_MM].extend(result.valid_dist_mm)
                if result_localizer:
                    vals[Metrics.AVG_LOC_DIST_MM_LOCALIZER].extend(
                        result_localizer.valid_dist_mm)
                if result_best:
                    vals[Metrics.AVG_LOC_DIST_MM_BOUND].extend(
                        result_best.valid_dist_mm)

                for i in range(len(result.correct)):
                    if result.true_unknown[i] or result.pred_unknown[i]:
                        continue

                    ided = True

                    if result.dist_mm[i] > 20:
                        ided = False
                    else:
                        tp = result.true_pos_mm[i]
                        dists = np.sqrt(np.sum((tp - result.pred_pos_mm)**2,
                                               axis=1))
                        dists[np.isnan(dists)] = np.inf
                        if (dists < dists[i]).any():
                            ided = False

                    vals[Metrics.ID_RATE].append(ided)

        for metric, value in list(vals.items()):
            if isinstance(value, list):
                if value:
                    vals[metric] = np.mean(value)
                else:
                    del vals[metric]

        for metric in list(vals.keys()):
            if isinstance(metric, PercentMetric):
                vals[metric] *= 100.

        return vals


def format_error(error, error_mm=None, unit='px'):
    """Creates a nicely formatted error string with optional error in mm."""
    errors = ['{:5.1f}{}'.format(error, unit)]

    if error_mm is not None:
        if abs(error_mm - error) < 1e-3:
            errors = []
        errors.insert(0, '{:5.1f}mm'.format(error_mm))

    return ' / '.join(errors)


def format_correct(correct: bool, missing: bool = False,
                   detected: bool = True) -> str:
    """Creates a nicely colored cross or checkmark for `localized`."""
    positive = '✅' if not missing else '➖'
    negative = '❎' if not missing else '❌'
    if correct:
        s = Fore.GREEN + positive
    elif not detected:
        s = Fore.BLUE + negative
    else:
        s = Fore.RED + negative
    return s + Fore.RESET


def digits(number: int) -> int:
    """Counts the digits in the given `number`."""
    if number > 0:
        return int(np.log10(number)) + 1
    elif number == 0:
        return 1
    else:
        return int(np.log10(-number)) + 2


def plot_results(image: Image, output_path: Path,
                 potential: Optional['Potential'] = None,
                 pot_pos: Optional[np.ndarray] = None,
                 parts: Optional[List[str]] = None,
                 pred_pos: Optional[np.ndarray] = None,
                 correct: Optional[np.ndarray] = None,
                 candidates: Optional[Dict[str, np.ndarray]] = None,
                 max_dist: float = 10.) -> None:
    """Visualizes the potential's energies values w.r.t. the given image.

    The default implementation fixes all except one part position to the
    true positions and plots the energies w.r.t. this part changing.

    Args:
        image: The image we want to visualize energies for.
        output_path: Path to an image file where to store the result.
        potential: If a potential is given, the potential energies are plotted
            next to the image.
        pot_pos: Used as anchor points for the potential energy computation
            if `potential` is given. This defaults to the annotations in the
            `image` if not provided.
        parts: The list of parts.
        pred_pos: As given by a `LocResult`.
        correct: 1D array indicating which `pred_pos` is correct.
        candidates: Optional set of candidates to localize as well if
            possible.
        max_dist: The maximal dist (treated as mm if image has spacing)
            to visualize as localization tolerance.
    """
    assert parts is not None or potential is not None

    n_dims, n_channels = guess_image_shape(image.data)
    shape = image.data.shape[:n_dims]

    spacing = np.ones(n_dims) if image.spacing is None else \
        image.spacing[::-1]
    shape_mm = shape * spacing

    views = [(0, 1)]
    if n_dims == 3:
        views += [(0, 2), (1, 2)]

    # extend of the image
    dpi = 90
    figure_width = sum(shape_mm[x] for y, x in views)
    figure_height = max(shape_mm[y] for y, x in views)

    if parts is not None:
        colors = distinct_colors(len(parts))
        true_pos = Criterion.true_pos(image, parts, n_dims)

    # if we want to visualize a potential, we compute the energies for it
    if potential is not None:
        figure_width = figure_width * 2 + shape_mm[views[0][1]] / 2

        # prepare positions to compute the energies
        if pot_pos is None:
            pot_pos = [image.parts[part].position[::-1]
                       if part in image.parts else None
                       for part in potential.parts]
        all_image_pos = np.transpose(np.unravel_index(
            range(np.prod(shape)), shape))
        positions = [np.resize(pos, (np.prod(shape), n_dims))
                     for pos in pot_pos[:-1]] + [all_image_pos]

        # finally compute the energy tensor
        energies = potential.compute(image, positions=positions)
        energies.shape = shape

        # use an appropriate normalization
        valid_selector = np.isfinite(energies)
        valid_energies = energies[valid_selector]
        if valid_energies.size == 0:
            norm = Normalize(vmin=0, vmax=1)
        elif valid_energies.min() >= 0:
            if valid_energies.min() == 0:
                energies[valid_selector] += 1e-10
                valid_energies += 1e-10
            norm = LogNorm(vmin=valid_energies.min(),
                           vmax=valid_energies.max())
        else:
            vmin, vmax = valid_energies.min(), valid_energies.max()
            if abs(vmin - vmax) < 1e-3:
                vmax += .1
            norm = Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.cm.viridis_r

    fig = plt.figure(figsize=(figure_width / dpi, figure_height / dpi))

    # black background
    fig.add_axes([0, 0, 1, 1], label='bg')
    plt.gca().patch.set_facecolor('black')

    fig_pos_x = 0

    for view_idx, (y_axis, x_axis) in enumerate(views):
        axes = [x_axis, y_axis]
        fixed_axis = next(iter({0, 1, 2} - {y_axis, x_axis}))
        fixed_idx = None
        sub_width = shape_mm[x_axis] / figure_width
        sub_height = shape_mm[y_axis] / figure_height
        sub_y = (1 - sub_height) / 2

        extent = (-spacing[x_axis] / 2,
                  shape_mm[x_axis] - spacing[x_axis] / 2,
                  shape_mm[y_axis] - spacing[y_axis] / 2,
                  -spacing[y_axis] / 2)

        if n_dims == 3:
            if potential is None:
                valid_coord = true_pos[~np.isnan(true_pos[:, 0]), fixed_axis]
                if valid_coord.size == 0 and pred_pos is not None:
                    valid_coord = pred_pos[~np.isnan(pred_pos[:, 0]),
                                           fixed_axis]
                if valid_coord.size == 0:
                    fixed_idx = shape[fixed_axis] // 2
                else:
                    _, fixed_idx = max(
                        [((np.abs(valid_coord - idx)
                           * spacing[fixed_axis] < max_dist).sum(), idx)
                         for idx in range(0, shape[fixed_axis])])
            else:
                fixed_idx = min(max(round(pot_pos[-1][fixed_axis]), 0),
                                shape[fixed_axis] - 1)

        def extract_view(arr):
            if n_dims == 3:
                arr = np.take(arr, fixed_idx, axis=fixed_axis)
                # if we started with a 3D multi channel image, we only
                # visualize the first one
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
            return arr

        def to_xy(pos):
            pos = pos * spacing
            return pos[:, axes] if pos.ndim == 2 else pos[axes]

        def slice_dist(pos):
            return abs(pos[fixed_axis] - fixed_idx) * spacing[fixed_axis]

        def plot_cands(part, color):
            if n_dims == 2:
                plt.plot(*to_xy(candidates[part]).T, '.', ms=1,
                         color=color)
            else:
                for cand in candidates[part]:
                    p = to_xy(cand)
                    hl = max_dist / 2
                    if n_dims == 3:
                        hl = (max_dist - slice_dist(cand)) / 2
                        if hl < .5:
                            continue
                    plt.plot([p[0] - hl, p[0] + hl], [p[1], p[1]],
                             '-', lw=.25, color=color, alpha=.5)
                    plt.plot([p[0], p[0]], [p[1] - hl, p[1] + hl],
                             '-', lw=.25, color=color, alpha=.5)

        def plot_true_pos(pos, color, incorrect=False):
            p = to_xy(pos)
            size = max_dist
            if n_dims == 3:
                size -= slice_dist(pos)
            if size >= 1:
                ellipse = Ellipse(xy=p, width=size, height=size, ec=color,
                                  alpha=1 if incorrect else .5,
                                  lw=1 if incorrect else .5, fc='none',
                                  ls='--' if incorrect else '-')
                plt.gca().add_patch(ellipse)
                plt.plot(*p, ',', ms=.5, alpha=1 if incorrect else .5,
                         color=color)

        def plot_pred_pos(pos, color, incorrect=False):
            p = to_xy(pos)
            hl = max_dist / 2
            if n_dims == 3:
                hl = (max_dist - slice_dist(pos)) / 2
                if hl < .5:
                    return
            plt.plot([p[0] - hl, p[0] + hl], [p[1], p[1]], '-',
                     lw=1 if incorrect else .5, color=color)
            plt.plot([p[0], p[0]], [p[1] - hl, p[1] + hl], '-',
                     lw=1 if incorrect else .5, color=color)

        def plot_annotations():
            if potential is None:
                for part_idx, (color, part) in enumerate(zip(colors, parts)):
                    tp = None if true_pos is None else true_pos[part_idx]
                    pp = None if pred_pos is None else pred_pos[part_idx]
                    corr = None if correct is None else bool(correct[part_idx])

                    if tp is not None and not np.isnan(tp[0]):
                        plot_true_pos(tp, color, incorrect=corr is False)

                        if corr is False:
                            plot_cands(part, color)

                        if (pp is not None and not np.isnan(pp[0])
                            and (n_dims == 2
                                 or (slice_dist(tp) <= max_dist - 1
                                     and slice_dist(pp) <= max_dist - 1))):
                            plt.plot(*np.transpose([to_xy(pp), to_xy(tp)]),
                                     '-', lw=.5, alpha=.7, color=color)

                    if pp is not None and not np.isnan(pp[0]):
                        plot_pred_pos(pp, color, incorrect=corr is False)
            else:
                for i, pos in enumerate(pot_pos):
                    color = 'r' if i + 1 == potential.arity() else 'g'
                    plot_true_pos(pos, color)
                    plt.annotate(str(i + 1), xy=to_xy(pos) + [5, 5], ha='left',
                                 va='top', color=color, fontsize='small')
                    if candidates is not None and i + 1 == potential.arity():
                        plot_cands(potential.parts[i], 'r')

        # image view
        fig.add_axes([fig_pos_x, sub_y,
                      sub_width, sub_height]).set_axis_off()
        fig_pos_x += sub_width
        plt.imshow(extract_view(image.data), extent=extent,
                   cmap=plt.cm.gray, interpolation='none',
                   aspect='equal')

        plot_annotations()

        if view_idx == 0 and pred_pos is not None and correct is not None:
            tm, pm = np.isnan(true_pos[:, 0]), np.isnan(pred_pos[:, 0])
            mislocalized = (~correct & ~tm & ~pm).sum()
            fp = (~correct & tm & ~pm).sum()
            fn = (~correct & ~tm & pm).sum()
            texts = []
            if mislocalized:
                texts.append(f'{mislocalized} mis-localized')
            if fp:
                texts.append(f'{fp} false positive (FP)')
            if fn:
                texts.append(f'{fn} false negative (FN)')
            if texts:
                plt.annotate('\n'.join(texts), xy=(0, 1),
                             xycoords='axes fraction', ha='left', va='top',
                             color='white', fontsize='small')

        # if view_idx == 0:
        #     plt.annotate(' '.join(f'{i + 1}:{p}'
        #                           for i, p in enumerate(self.parts)),
        #                  xy=(0, 0), xycoords='axes fraction',
        #                  xytext=(.5, .5), textcoords='offset points',
        #                  ha='left', va='bottom', color='white',
        #                  fontsize='small')

        # heatmap view
        if potential is not None:
            fig.add_axes([fig_pos_x, sub_y,
                          sub_width, sub_height]).set_axis_off()
            fig_pos_x += sub_width
            extracted_energies = extract_view(energies)
            plt.imshow(extracted_energies, extent=extent, cmap=cmap, norm=norm,
                       interpolation='none', aspect='equal')
            # if candidates is not None:
            #     plot_cands(self.parts[-1])
            # plot unknown energy contour lines
            if potential.unknown_energy is not None:
                valid_energies = extracted_energies[
                    np.isfinite(extracted_energies)]
                vmin, vmax = valid_energies.min(), valid_energies.max()
                if vmin <= potential.unknown_energy <= vmax:
                    x, y = np.meshgrid(np.arange(extracted_energies.shape[1]),
                                       np.arange(extracted_energies.shape[0]))
                    plt.contour(x * spacing[x_axis], y * spacing[y_axis],
                                extracted_energies,
                                levels=[potential.unknown_energy],
                                colors=['orange'], linewidths=[.5],
                                antialiased=False)
            plot_annotations()
            # ellipse = Ellipse(xy=(all_pos[-1] * spacing)[[x_axis, y_axis]],
            #                   width=max_dist, height=max_dist, ec='r', lw=.5,
            #                   fc=(0, 0, 0, 0))
            # plt.gca().add_patch(ellipse)

    # colorbar if we plotted energies as well
    if potential is not None:
        fig.add_axes([fig_pos_x + (1 - fig_pos_x) * .8 / 2,
                      0.05, (1 - fig_pos_x) * 0.2, 0.9])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm, cmap),
                          cax=plt.gca(), format='%.2f')
        cb.outline.set_edgecolor('w')
        cb.set_ticks(norm.inverse(np.linspace(0, 1, 10)).filled().tolist())
        cb.ax.tick_params(length=0, pad=0)
        cb.update_ticks()
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='r',
                 fontsize='x-small', ha='center', x=0.5)

    plt.savefig(str(output_path), dpi=dpi)
    plt.close()

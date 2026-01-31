"""Functions for **learning potential weights, energies and parameters**.

# Basic functionality

Each learner is driven by a `Criterion`, such as the simple and most commonly
used one `MaxDistance`, which generate you a `Result` that indicates the
current performance of the graph applied to a `Sample`.

In order to perform the optimization, we have classes that inherit from
`IterativeLearning`. They try to improve various `Metric`s, generating a
`LearningResult` in each iteration, and can be stopped raising the
`StopOptimization` exception.

# Optimizing a max-margin hinge loss using SGD to estimate weights and energies

The very first optimizer is `SgdMaxMarginLearning`. It should work with all
kinds of potentials and can optimize the weights as well as the energies for
the unknown label.

# Optimizing a max-margin hinge loss using SGD to estimate weights, energies
# **and parameters**

The second optimizer is `FullSgdMaxMarginLearning`. It should provide the
same functionality as the previous one, but is additionally able to learn
potential parameters of potentials that implement the `LearnablePotential`
interface.

# Using constraint generation to optimize potential weights

The third optimizer has a different apporach. It implemets the constraint
generation approach as given in probablisitic graphical models by moller and
friedman with different adaptations and add-ons. To find the weights themselves
the gekko optimization suit is used.
"""


import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import tensorflow as tf
import threading
import time

from abc import abstractmethod
from collections import OrderedDict, defaultdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from gekko import GEKKO
from hiwi import Image, batchize
from humanfriendly import format_timespan
from multiprocessing import Pool
from natsort import natsorted
from numpy.linalg import norm
from pathlib import Path
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple
from typing import Callable, Union

from .graph import Graph
from .potentials import Potential, LearnablePotentialMixin
from .inference import EnergyGraph, EnergyPotential
from .inference import ExactTreeSolver, solve_astar, solve_gibbs
from .inference import LazyFlipperSolver
from .evaluation import Criterion, Metric, Metrics
from .evaluation import LocResult, SetLocResult, plot_results, ImageResult
from .utils import working_dir, complement_color, format_time


log = logging.getLogger(__name__)


class Sample:
    """A sample (derived from an `Image`) we can perform inference on.

    Args:
        image: The image from which this sample is derived.
        graph: The graph we are operating on.
        criterion: The optimization criterion.
    """
    def __init__(self, image: Image, graph: Graph,
                 criterion: Criterion) -> None:
        self.image = image
        """The image from which this sample was derived."""

        self.graph = graph
        """The graph we are trying to optimize."""

        self.criterion = criterion
        """The optimization criterion we are working with."""

        self.true_pos = Criterion.true_pos(image, graph.parts, graph.n_dims)
        """Array for all true positions of all target parts, in order. Missing
        parts are indicated by NaN values."""

        # sanity check
        no_unknown_support = [part for i, part in enumerate(graph.parts)
                              if np.isnan(self.true_pos[i, 0])
                              and part not in graph.support_unknown]
        if no_unknown_support:
            log.error('The image %s has missing parts that the specified '
                      'graph does not support unknown label for: %s',
                      image.name, ', '.join(no_unknown_support))

    def best_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Best achievable positions enforced by the limited set of
        candidates.
        """
        candidates = self.collect_candidates()

        best_cands, best_idx = self.criterion.best_configuration(
            self.image, self.graph.parts, candidates, ret_indices=True)

        # just a sanity check that the model is configured to match the data
        bad_missing_parts = []
        for i, part in enumerate(self.graph.parts):
            if best_idx[i] == -1 and part not in self.graph.support_unknown:
                bad_missing_parts.append(part)

        if bad_missing_parts:
            log.error('%s: Parts %s are missing but the model is not '
                      'configured to support missing for these parts',
                      self.image.name, ', '.join(bad_missing_parts))

        return best_cands, best_idx

    def first_candidates(self) -> np.ndarray:
        """The first best candidate for each part, which corresponds to the
        maximum of the localizer, i.e., just applying a localizer.
        """
        candidates = self.collect_candidates()
        return np.array([candidates[part][0] for part in self.graph.parts])

    def infer(self, n_best: int = 1, approx: bool = False) -> np.ndarray:
        """Finds the best localization results.

        Args:
            n_best: Number of best results to return.
            approx: Whether to use approximate inference in form of the
                LazyFlipper. Is only used for `n_best` == 1 and if there
                is no polynomial-time exact inference algorithm.

        Return:
            An array of localization results, where `NaN` values indicate
            the unknown label.
        """
        candidates, energy_graph = self.compute()

        # maximal inference time, hardcoded for now
        MAX_INF_TIME = 10. if n_best > 1 else threading.TIMEOUT_MAX

        states = None
        global_optimum = None

        start_time = time.time()

        if energy_graph.is_forest:
            solver = ExactTreeSolver()
            global_optimum, optimal_energy = solver.infer(energy_graph)

            if n_best == 1:
                states = global_optimum
        elif approx and n_best == 1:
            solver = LazyFlipperSolver(max_subgraph_size=3)
            states, energy = solver.infer(energy_graph)

        # the default approach: run astar and gibbs in parallel, if astar does
        # not complete within a certain amount of time, we use the best gibbs
        # results found so far
        if states is None:
            pool = Pool()

            astar_args = (energy_graph, n_best)
            if n_best > 1:
                astar_args += (3_000_000,)

            res_astar = pool.apply_async(solve_astar, astar_args)

            res_gibbs_first = None
            res_gibbs_random = None
            res_gibbs_opt = None

            if n_best > 1:
                # TODO: use RandomState rather than non-deterministic random
                start_random = np.array([np.random.randint(l)
                                         for l in energy_graph.n_labels])
                start_first = np.array([1 if part in self.graph.support_unknown
                                        else 0 for part in self.graph.parts])

                # run three different start positions in parallel
                res_gibbs_first = pool.apply_async(
                    solve_gibbs, (energy_graph, n_best, start_first,
                                  MAX_INF_TIME))
                res_gibbs_random = pool.apply_async(
                    solve_gibbs, (energy_graph, n_best, start_random,
                                  MAX_INF_TIME))
                if global_optimum is not None:
                    res_gibbs_opt = pool.apply_async(
                        solve_gibbs, (energy_graph, n_best, global_optimum,
                                      MAX_INF_TIME))

            try:
                states = res_astar.get(MAX_INF_TIME)
            except multiprocessing.TimeoutError:
                log.warning('Did not solve graph using a-star in %f secs, '
                            'falling back to Gibbs chain results',
                            MAX_INF_TIME)

                # join the different results based on the energies
                all_states, all_energies = [], []
                for res in [res_gibbs_first, res_gibbs_random, res_gibbs_opt]:
                    if res is not None:
                        states, energies = res.get()
                        all_states.extend(states)
                        all_energies.extend(energies)

                states, idx = np.unique(all_states, axis=0, return_index=True)
                states = states[np.argsort(np.array(all_energies)[idx])]

                if len(states) < n_best:
                    log.warning('Found only %i states instead of %i using '
                                'multiple Gibbs chains', len(states), n_best)
            except KeyboardInterrupt as e:
                pool.terminate()
                pool.join()
                del pool

                raise e

            pool.terminate()
            pool.join()
            del pool

        time_delta = time.time() - start_time
        if time_delta > 5:
            log.warning('Inference for sample %s took more than 5 '
                        'seconds: %s', self.image.name,
                        format_timespan(time_delta))

        if states.ndim == 1:
            states.shape = (1, states.size)

        if len(states) < n_best:
            log.error('Found fewer best states (%i) than specified (%i)',
                      len(states), n_best)

        pos = np.full((len(states), self.graph.n_parts, self.graph.n_dims),
                      np.nan)

        for i, state in enumerate(states):
            for j, (part, label) in enumerate(zip(self.graph.parts, state)):
                if part in self.graph.support_unknown:
                    label -= 1

                if label >= 0:
                    pos[i, j] = candidates[part][label]

        if n_best == 1 and len(pos) > 0:
            pos = pos[0]

        return pos

    def compute_energies(self, positions: np.ndarray,
                         use_unknown_energies: bool = True,
                         use_weights: bool = True) -> np.ndarray:
        """Computes potential energies.

        Args:
            positions: One or multiple sets of localization predictions.
            use_unknown_energies: Whether to use the real unknown energies
                or use `NaN`s instead.
            use_weights: Whether to apply the energy weighting.

        Return: One or multiple sets of energy values (|positions| x |pots|).
        """
        assert positions.ndim in (2, 3)

        is_one = positions.ndim == 2
        if is_one:
            positions = positions.reshape((1,) + positions.shape)

        n_states = len(positions)

        positions = np.transpose(positions, [1, 0, 2])
        # reorient to part,state,coord

        energies = np.empty((self.graph.n_potentials, n_states))

        for i, pot in enumerate(self.graph.potentials):
            var_indices = [self.graph.part_idx[part] for part in pot.parts]

            energies[i] = pot.unknown_energy if use_unknown_energies \
                else np.nan

            pot_pos = positions[var_indices]
            valid_pos = ~np.isnan(pot_pos).any(axis=(0, 2))

            energies[i, valid_pos] = pot.compute(self.image,
                                                 pot_pos[:, valid_pos])

            if use_weights:
                energies[i] *= pot.weight

        energies = energies.T

        if is_one:
            energies = energies[0]

        return energies

    def compute(self) -> Tuple[Dict[str, np.ndarray], EnergyGraph]:
        """Computes an energy graph.

        First, it uses `collect_candidates()` to find the important candidates,
        for which it then computes the enery graph.

        Return: (`candidates`, `energy_graph`)
        """
        candidates = self.collect_candidates()

        energy_graph = self.graph.compute_energies(
            self.image, candidates,
            use_weights=True,
            support_unknown=self.graph.support_unknown)

        return candidates, energy_graph

    def collect_candidates(self) -> Dict[str, np.ndarray]:
        """Collects potential candidate positions."""
        return self.graph.collect_candidates(self.image)

    def eligible_for_training(self) -> Union[bool, str]:
        """Determines whether this sample is eligible for training.

        Return:
            In case it should not be used in training, it can either return
            `False` or a reason (`str`) why it should not be included.
        """
        best_result = self.evaluate(self.best_candidates()[0])

        if not best_result.correct.all():
            parts = [p for p, c in zip(self.graph.parts, best_result.correct)
                     if not c]
            return (f'no correct candidate for {len(parts)} parts '
                    + ', '.join(parts))

        return True

    def evaluate(self, pred_pos: np.ndarray) -> LocResult:
        """Evaluates a localization result for this sample.

        Args:
            pred_pos: The predicted positions.

        Returns: A `LocResult` for the given predicted positions.
        """
        return self.criterion.evaluate(self.image, self.graph.parts,
                                       pred_pos, self.true_pos)


class FixedGraphSample(Sample):
    """A sample that assumes a fixed graph.

    This sample assumes that the potentials of the graph DO NOT change over
    time. Thus, we can precompute candidates and unweighted energies,
    and re-use those values.

    Furthermore, this also might render a sample not `solvable()`, due to
    the fixed set of candidates.
    """
    def __init__(self, image: Image, graph: Graph, criterion: Criterion):
        super().__init__(image, graph, criterion)

        @working_dir.cache(f'samples/fixed_sample_{image.name}')
        def compute():
            candidates = graph.collect_candidates(image)

            energy_graph = graph.compute_energies(
                image, candidates,
                use_weights=False,
                support_unknown=graph.support_unknown,
                unknown_energy=np.nan)

            return candidates, energy_graph

        candidates, energy_graph = compute()

        self.candidates = candidates
        """Our fixed set of candidates."""

        self.energy_graph = energy_graph
        """Unweighted and nan unknown energy using precomputed energy graph."""

    @lru_cache(1)
    def best_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        return super().best_candidates()

    @lru_cache(1)
    def first_candidates(self) -> np.ndarray:
        return super().first_candidates()

    def collect_candidates(self) -> Dict[str, np.ndarray]:
        return self.candidates

    def compute_energies(self, positions: np.ndarray,
                         use_unknown_energies: bool = True,
                         use_weights: bool = True) -> np.ndarray:
        is_one = positions.ndim == 2
        if is_one:
            positions = positions.reshape((1,) + positions.shape)

        n_states = len(positions)

        positions = np.transpose(positions, [1, 0, 2])
        # reorient to part,state,coord

        # find the candidate indices from the given positions
        indices = np.full((self.graph.n_parts, n_states), -1)
        for part_idx, states in enumerate(positions):
            part = self.graph.parts[part_idx]
            cands = self.candidates[part]

            start_idx = 1 if part in self.graph.support_unknown else 0

            for state_idx, pos in enumerate(states):
                if np.isnan(pos[0]):
                    continue

                idx = np.where((cands == pos).all(axis=1))[0]
                assert len(idx) > 0

                indices[part_idx, state_idx] = start_idx + idx[0]

        # collect energies from the precomputed graph
        energies = np.empty((len(self.graph.potentials), n_states))
        for i, pot in enumerate(self.energy_graph.potentials):
            energies[i] = self.graph.potentials[i].unknown_energy \
                if use_unknown_energies else np.nan

            pot_indices = indices[pot.variables]
            valid_states = (pot_indices >= 0).all(axis=0)

            energies[i, valid_states] = pot.values[
                tuple(pot_indices[:, valid_states])]

            if use_weights:
                energies[i] *= self.graph.potentials[i].weight

        energies = energies.T
        if is_one:
            energies = energies[0]

        return energies

    def compute(self) -> Tuple[Dict[str, np.ndarray], EnergyGraph]:
        potentials = []

        for pot, en_pot in zip(self.graph.potentials,
                               self.energy_graph.potentials):
            if pot.weight == 0:
                continue

            values = np.copy(en_pot.values)

            if any(part in self.graph.support_unknown
                   for part in pot.parts):
                values[np.isnan(values)] = pot.unknown_energy

            values *= pot.weight

            assert not np.isnan(values).any()

            potentials.append(EnergyPotential(en_pot.variables,
                                              values))

        updated_energy_graph = EnergyGraph(self.energy_graph.n_labels,
                                           potentials)

        return self.candidates, updated_energy_graph


class Rivals(NamedTuple):
    """Keeps the list rival/correct configuration energies and errors."""

    index: List[Any]
    """List of identifiable entries, one describing each rival to identify
    and exclude duplicates.
    """

    corr_energies: List[np.ndarray]
    """List of the potential energies for the correct configurations."""

    incorr_energies: List[np.ndarray]
    """List of the potential energies for the incorrect (rivaling)
    configuration.
    """

    error_reduction: List[float]
    """Reduction in error (as given by the `Criterion`), going from the
    rivaling to the correct configuration.
    """

    @staticmethod
    def empty() -> 'Rivals':
        """Creates a new empty rival set."""
        return Rivals(index=[], corr_energies=[], incorr_energies=[],
                      error_reduction=[])


class LearningResult(NamedTuple):
    """Results of a parameter learning session at a specific time point."""

    iteration: int
    """The index of the iteration this result is from."""

    epoch: float
    """The epoch, given as fraction."""

    time: float
    """Duration in seconds when this result was generated."""

    metrics: Dict[Metric, float]
    """Mapping of the metrics available."""

    def __str__(self):
        prefix = 'Baseline: ' if self.iteration == 0 else \
            f'Iteration {self.iteration}({self.epoch:.2f}): '
        return prefix + ' '.join(f'{m}={m.format(v)}'
                                 for m, v in self.metrics.items())


class StopOptimization(Exception):
    """Exception to signal the end of the optimization."""
    pass


class GraphParams(NamedTuple):
    """Describes the parametric state of a graph w.r.t. learnable ones."""

    weights: np.ndarray
    """The potential weights."""

    unknown_energies: np.ndarray
    """The potential unknown energies"""

    potential_parameters: Dict[LearnablePotentialMixin, Any]
    """Maps potential instances to corresponding parameters."""

    @staticmethod
    def get(graph: Graph) -> 'GraphParams':
        """Extracts the learnable parameters from `graph`."""
        potential_params = {pot: pot.get_parameters()
                            for pot in graph.potentials
                            if isinstance(pot, LearnablePotentialMixin)}
        return GraphParams(weights=graph.weights,
                           unknown_energies=graph.unknown_energies,
                           potential_parameters=potential_params)

    def set(self, graph: Graph) -> None:
        """Sets the learnable `graph` parameters to this state. """
        graph.weights = self.weights
        graph.unknown_energies = self.unknown_energies
        for pot, params in self.potential_parameters.items():
            assert pot in graph.potentials
            pot.set_parameters(params)


class OptBatch(NamedTuple):
    """An optimization batch send to an optimizer."""

    samples: List[Sample]
    """The samples to use for optimization."""

    prev_result: LearningResult
    """Result of the previous iteration."""

    best_result: LearningResult
    """Result of the best iteration so far."""

    best_params: GraphParams
    """The best achieved parameters so far."""


class IterativeLearning:
    """Abstract base class for iterative learners.

    This basically provides the basic stuff to compute common error metrics,
    defining stopping criteria and doing all the pre-processing.

    Args:
        sample_cls: Type of the `Sample` class to use for inference etc.
        criterion: Our evaluation criterion.
        n_rivals: Number of rivals to generate for each sample.
        allow_correct_rivals: Also allows rival (incorrect) configurations that
            are actually correct, but have a higher error than the correct
            configuration.
        approx_test_inference: Whether to use approximate inference to generate
            metric values during training.
    """
    def __init__(self, sample_cls: type, criterion: Criterion,
                 n_rivals: int = 1,
                 allow_correct_rivals: bool = False,
                 approx_test_inference: bool = False) -> None:
        self.criterion = criterion
        """The criterion to classify predictions as correct or incorrect."""

        self.sample_cls = sample_cls
        """Type of the `Sample` class to use."""

        self.n_rivals = n_rivals
        """Number of rivals to use for each sample."""

        self.allow_correct_rivals = allow_correct_rivals
        """Whether to allow also fully correct, but erroneous rivals."""

        self.approx_test_inference = approx_test_inference
        """Whether to use approximate inference during metric computation."""

    def optimize(self, graph: Graph,
                 train_images: List[Image],
                 val_images: Optional[List[Image]] = None,
                 opt_metrics: Optional[List[Metric]] = None,
                 batch_size: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 max_iterations: Optional[int] = None,
                 max_stagnation: Optional[int] = None,
                 max_time: Optional[float] = None,
                 log_metrics_every: int = 1) -> LearningResult:
        """Performs the weight optimization.

        After the optimization has been finished, the learned parameters
        with the best `opt_metrics` values are used to update the graph.

        Args:
            graph: The graph we want to optimize.
            train_images: List of training images to perform the optimization
                on.
            val_images: Optional list of val images to compute the metrics for
                as well. It can also be used to perform the parameter selection
                depending on `minimize_metrics`.
            opt_metrics: List of metrics to optimize, order defines the
                priority. If `None`, it uses a heuristic to select the best
                metrics. E.g., if `val_images` is given, it uses
                `['val_error', 'error', 'loss']`, meaning it minimizes error
                first and if error is equal it uses the loss.
            batch_size: An optional mini batch size use to perform stochastic
                gradient descent instead of batch gradient descent.
            max_epochs: Maximal number of epochs to run, i.e., number of
                passes through the whole dataset.
            max_iterations: Maximal number of descent steps.
            max_stagnation: Maximal number of iterations without an
                improvement before stopping the optimization.
            max_time: Maximal time in seconds this optimization should last,
                beware that the optimization will probably be a bit longer
                than specified here.
            log_metrics_every: Log metrics every x iterations.

        Returns:
            A `LearningResult` which contains the results for the set of
            parameters that were finally used.
        """
        assert len(train_images) > 0
        assert val_images is None or len(val_images) > 0
        assert batch_size is None or batch_size > 0
        assert max_epochs is None or max_epochs > 0
        assert max_iterations is None or max_iterations > 0
        assert max_time is None or max_time > 0
        assert not all(v is None for v in [max_epochs, max_iterations,
                                           max_time])

        max_stagnation = np.inf if max_stagnation is None else max_stagnation

        log.debug('Deriving training samples from %i images',
                  len(train_images))

        # compute the samples
        log.info('Create training samples')
        train_samples = self.create_samples(graph, train_images)

        if len(train_samples) == 0:
            log.fatal('We do not have any training samples!')

        val_samples = None
        if val_images:
            log.info('Create validation samples')
            val_samples = self.create_samples(graph, val_images or [],
                                              is_val=True)

        # adjust mini batch size
        if batch_size is None:
            batch_size = len(train_samples)
        elif batch_size > len(train_samples):
            log.warning('Specified batch_size of %i is larger than number of '
                        'training samples %i, using the latter instead',
                        batch_size, len(train_samples))
            batch_size = len(train_samples)

        # select the best minimization metrics
        if opt_metrics is None:
            opt_metrics = [Metrics.ERROR]
            if val_samples:
                opt_metrics.insert(0, Metrics.ERROR.val)

        log.info('Going to optimize the metrics %s with %i training samples '
                 'per iteration', ', '.join(map(str, opt_metrics)),
                 batch_size)

        # data generator
        batches = enumerate(batchize(train_samples, batch_size,
                                     max_epochs=max_epochs,
                                     max_iterations=max_iterations))

        start_time = time.time()

        # create the optimizer, and run the initialization code
        optimizer = self.optimizer(graph)
        optimizer.__next__()  # that actually starts the coroutine

        prev_weights = None
        prev_unknown_energies = None

        def build_result(iteration, init_metrics=None, compute=True):
            nonlocal prev_weights, prev_unknown_energies

            # compute and assemble metrics
            metrics = OrderedDict()

            if init_metrics:
                for metric in natsorted(init_metrics.keys(),
                                        key=lambda m: m.name):
                    metrics[metric] = init_metrics[metric]

            if compute:
                log.debug('Computing metrics over all training%s samples',
                          ' and validation' if val_samples else '')
                metrics.update(self.compute_metrics(train_samples))
                if val_samples:
                    metrics.update(self.compute_metrics(val_samples,
                                                        is_val=True))

            metrics[Metrics.USED_POTENTIALS] = np.sum(graph.weights > 0)

            # relative weight updates
            weights = graph.weights
            if prev_weights is not None:
                metrics[Metrics.REL_WEIGHTS_UPDATE] = \
                    norm(weights - prev_weights) \
                    / (norm(weights) + norm(prev_weights)) * 2 * 100
            prev_weights = weights

            # relative unknown energies update
            if graph.unknown_potentials:
                unknown_energies = graph.unknown_energies[
                    np.isfinite(graph.unknown_energies)]
                if prev_unknown_energies is not None:
                    metrics[Metrics.REL_UNKNOWN_ENERGIES_UPDATE] = \
                        norm(unknown_energies - prev_unknown_energies) \
                        / (norm(unknown_energies)
                           + norm(prev_unknown_energies)) * 2 * 100
                prev_unknown_energies = unknown_energies

            return LearningResult(
                iteration=iteration,
                epoch=iteration * batch_size / len(train_samples),
                time=time.time() - start_time,
                metrics=metrics)

        # baseline results
        result = build_result(0)

        all_results = [result]
        best_result = result
        best_params = GraphParams.get(graph)
        log.info('%s', result)

        try:
            for iteration, (epoch, batch) in batches:

                # here happens the actual optimization step
                metrics = optimizer.send(OptBatch(samples=batch,
                                                  prev_result=result,
                                                  best_result=best_result,
                                                  best_params=best_params))
                optimizer.__next__()  # advance from yield results to (yield)

                log_metrics = (iteration + 1) % log_metrics_every == 0

                result = build_result(iteration + 1, metrics,
                                      compute=log_metrics)
                all_results.append(result)
                log.info('%s', result)

                if log_metrics:
                    # check for new best result
                    changed_values = []
                    for metric in opt_metrics:
                        if metric not in result.metrics:
                            break

                        cur_val = result.metrics[metric]
                        try:
                            best_val = best_result.metrics[metric]
                        except KeyError:
                            if best_result.iteration > 0:
                                raise
                            # compensate for metrics that were not computed for
                            # baseline results, i.e., never hit optimization
                            best_val = cur_val

                        if metric.is_better(best_val, cur_val):
                            changed_values.append('{}={}{}{}'.format(
                                metric, metric.format(best_val),
                                '↘' if metric.minimize else '↗',
                                metric.format(cur_val)))

                            best_result = result
                            best_params = GraphParams.get(graph)

                            log.info('💡 This is a new best: %s',
                                     ' '.join(changed_values))
                        elif metric.is_equal(best_val, cur_val):
                            changed_values.append('{}={}'.format(
                                metric, metric.format(cur_val)))
                            continue

                        break

                if iteration - best_result.iteration >= max_stagnation:
                    raise StopOptimization(
                        'no improvement for {} iterations'.format(
                            iteration - best_result.iteration))

                if max_time is not None:
                    time_passed = time.time() - start_time
                    if time_passed > max_time:
                        raise StopOptimization(
                            'did optimize for {} (> {})'.format(
                                format_timespan(time_passed),
                                format_timespan(max_time)))

            stop_reason = ''  # silence qa

        except KeyboardInterrupt:
            stop_reason = 'caught keyboard interrupt'
        except StopOptimization as e:
            stop_reason = str(e)
        # FIXME: this does not work good with testing, we do not see the error
        except Exception as e:
            log.critical('Caught an unhandled exception in %s optimizer, '
                         'failing gracefully be storing latest best result',
                         self.__class__.__name__, exc_info=True)
            stop_reason = f'caught unhandled exception ({str(e)})'

        log.info('☠ Stopping optimization: %s', stop_reason)

        optimizer.close()

        if working_dir:
            log.debug('Plotting metric graphs')
            self.plot_results(all_results, best_result, len(train_samples),
                              batch_size, len(val_samples or []),
                              working_dir / 'metrics.pdf')

        best_params.set(graph)

        log.info('Using parameters of %s', best_result)
        log.debug('- Weights: %s', graph.weights)
        if graph.unknown_potentials:
            log.debug('- Unknown energies: %s', ' '.join(
                '{}:{}'.format(i, p.unknown_energy)
                for p, i in graph.unknown_potentials.items()))

        self.plot_sample_results('training', train_samples)
        if val_samples:
            self.plot_sample_results('validation', val_samples)

        return best_result

    @abstractmethod
    def optimizer(self, graph: Graph) -> Generator[Dict[Metric, float],
                                                   OptBatch,
                                                   None]:
        """An optimization coroutine.

        Here happens the actual optimization.

        The coroutine is supposed to wait for (`(yield)`) a batch and
        return a dictionary of metrics.

        Args:
            graph: The graph we want to optimize.
        """
        raise NotImplementedError

    def infer_rival_energies(self, graph: Graph, samples: List[Sample],
                             use_unknown_energies: bool = False,
                             rivals: Optional[Rivals] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rivals is None:
            rivals = Rivals.empty()

        n_rivals = 0

        def is_correct(pred_result, corr_result):
            if self.allow_correct_rivals:
                return (pred_result.error - corr_result.error).sum() < 1e-2
            else:
                return (pred_result.correct == corr_result.correct).all()

        def add_rival(sample, pred_pos, corr_pos, pred_result, corr_result):
            nonlocal n_rivals

            idx = (sample.image.name,) + tuple(np.around(pred_pos,
                                                         decimals=3).flat)
            if idx in rivals.index:
                log.debug('Ignored rival cause it already exists in rival set')
                return

            corr_energy, incorr_energy = \
                sample.compute_energies(
                    np.array([corr_pos, pred_pos]),
                    use_unknown_energies=use_unknown_energies,
                    use_weights=False)

            # safety check to find outlier images and bad potentials
            def find_suspicious_energies(positions, energies, name):
                for pot_idx, energy in enumerate(energies):
                    if energy == np.inf:
                        pot = graph.potentials[pot_idx]
                        pos = np.array([positions[graph.part_idx[p]]
                                        for p in pot.parts])

                        output_dir = working_dir / ('learning_suspicious_'
                                                    + name + '_energies')
                        output_dir.mkdir(exist_ok=True, parents=True)
                        output_path = output_dir / '{}-{}_{}.pdf'.format(
                            pot_idx, pot, sample.image.name)

                        if output_path.exists():
                            continue

                        log.error('Potential %i %s produced an infinity energy'
                                  ' for the %s configuration %s in image %s',
                                  pot_idx, pot, name, pos, sample.image.name)

                        plot_results(sample.image, output_path,
                                     potential=pot, pot_pos=pos)

            find_suspicious_energies(corr_pos, corr_energy, 'correct')
            find_suspicious_energies(pred_pos, incorr_energy, 'incorrect')

            # how did we get a negative loss?!?!?!
            # maybe the error reduction was negative?

            rivals.index.append(idx)
            rivals.corr_energies.append(corr_energy)
            rivals.incorr_energies.append(incorr_energy)
            rivals.error_reduction.append(np.sum(pred_result.error
                                                 - corr_result.error))
            n_rivals += 1

        for i, sample in enumerate(samples):
            corr_pos = self.correct_position(sample)
            corr_result = sample.evaluate(corr_pos)

            n_sample_rivals = 0

            # this is a special case that we exploit for better runtime
            if self.n_rivals == 1 and graph.is_forest:
                pred_pos = sample.infer()
                pred_result = sample.evaluate(pred_pos)
                assert (pred_result.error >= corr_result.error).all()

                if not is_correct(pred_result, corr_result):
                    add_rival(sample, pred_pos, corr_pos, pred_result,
                              corr_result)
                    continue

            for pred_pos in sample.infer(n_best=self.n_rivals + 20):
                pred_result = sample.evaluate(pred_pos)
                assert (pred_result.error >= corr_result.error).all()

                # ignore correct states
                if is_correct(pred_result, corr_result):
                    continue

                add_rival(sample, pred_pos, corr_pos, pred_result, corr_result)

                n_sample_rivals += 1
                if n_sample_rivals == self.n_rivals:
                    break

        max_n_rivals = self.n_rivals * len(samples)
        if n_rivals != max_n_rivals:
            log.warning('Collected only %i instead of %i rivals * %i samples '
                        '= %i total rivals', n_rivals, self.n_rivals,
                        len(samples), max_n_rivals)
        assert n_rivals > 0 or rivals is not None

        if n_rivals != len(rivals.corr_energies):
            log.debug('Using now totally %i rivals (%i found in this '
                      'iteration)', len(rivals.corr_energies), n_rivals)

        corr_energies = np.array(rivals.corr_energies)
        incorr_energies = np.array(rivals.incorr_energies)
        error_reduction = np.array(rivals.error_reduction)

        # ensure we work with valid values
        with np.errstate(invalid='ignore'):
            dropped_terms = graph.weights == 0
            assert (np.isnan(corr_energies) | dropped_terms
                    | (corr_energies > -np.inf)).all()
            assert (np.isnan(incorr_energies) | dropped_terms
                    | (incorr_energies > -np.inf)).all()

        return corr_energies, incorr_energies, error_reduction

    def correct_position(self, sample: Sample) -> np.ndarray:
        """Defines the correct position for the given `sample`.

        This correct position is used in the rival expression to form the
        energy margin between various incorrect positions.
        """
        return sample.best_candidates()[0]

    def compute_metrics(self, samples: List[Sample],
                        is_val: bool = False,
                        result_cb: Optional[Callable[[Sample, ImageResult],
                                                     None]] = None) -> Metrics:
        """Computes sample metrics for a list of samples.

        Args:
            samples: The list of samples to compute the metrics for.
            is_val: Whether these are validation or training images.
            result_cb: A function that is called (if given) with a sample
                and the respective localiztion results.

        Returns: A new dictionary mapping metric name to value.
        """
        image_results = []

        for sample in samples:
            best_cands_pos, best_cands_idx = sample.best_candidates()
            result = sample.evaluate(sample.infer(
                approx=self.approx_test_inference))
            result_localizer = sample.evaluate(sample.first_candidates())
            result_best = sample.evaluate(best_cands_pos)
            best_cands = best_cands_idx[result_best.correct
                                        & (best_cands_idx >= 0)] + 1

            image_result = ImageResult(predicted=result,
                                       localizer=result_localizer,
                                       best=result_best,
                                       best_cands=best_cands,
                                       candidates=sample.collect_candidates())

            if result_cb is not None:
                result_cb(sample, image_result)

            image_results.append(image_result)

        vals = Metrics.compute(image_results)

        if is_val:
            vals = Metrics([(k.val, v) for k, v in vals.items()])

        return vals

    def create_samples(self, graph: Graph, images: List[Image],
                       is_val: bool = False) -> List[Sample]:
        """Turns images into samples.

        Args:
            graph: Tests the images against this graph in case of
                training images.
            images: The list of images we want to turn into samples.
            is_val: Whether these are validation or training images.

        Returns: A new list of `Sample`s.
        """

        log.info('Creating samples for %i %s images', len(images),
                 'validation' if is_val else 'training')

        samples = []

        for image in images:
            sample = self.sample_cls(image, graph, self.criterion)

            if not is_val:
                eligible = sample.eligible_for_training()
                if eligible is not True:
                    log.warning('Ignoring training image %s%s', image.path,
                                ': ' + eligible if isinstance(eligible, str)
                                else '')
                    continue

            samples.append(sample)

        if len(samples) != len(images):
            log.warning('Derived only %i samples from %i images',
                        len(samples), len(images))

        return samples

    def plot_sample_results(self, category: str,
                            samples: List[Sample]) -> None:
        """Computes and plots the final localization results."""
        log.info('Computing and plotting final %s results:', category)

        name_length = max(len(s.image.name) for s in samples) + 1
        results = []

        output_dir = working_dir / f'results_{category}'
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        chain = samples[0].graph.is_chain
        if chain is not None:
            log.info('Parts are illustrated as chain from %s to %s',
                     chain[0], chain[-1])
            chain = [samples[0].graph.part_idx[c] for c in chain]

        def process(sample: Sample, result: ImageResult):
            image = sample.image
            pred_result = result.predicted
            results.append(pred_result)

            if chain is not None:
                pred_result.order = chain
            log.info(' - %-*s  %s', name_length, image.name + ':', pred_result)

            if output_dir:
                n_errs = pred_result.correct.size - pred_result.correct.sum()
                output_path = output_dir / f'{n_errs}_{image.name}.pdf'
                plot_results(image, output_path, parts=sample.graph.parts,
                             pred_pos=pred_result.pred_pos,
                             correct=pred_result.correct,
                             candidates=result.candidates)

        metrics = self.compute_metrics(samples, result_cb=process)

        log.info('Final %s results: %s', category, SetLocResult(results))
        for m, v in metrics.items():
            log.info(' - %s = %s', m, m.format(v))

    @staticmethod
    def plot_results(results: List[LearningResult],
                     best_result: LearningResult, n_samples: int,
                     mini_batch: int, n_val_samples: int,
                     output_file: Path) -> None:
        """Plots the results."""

        # get all metric values
        min_iter = np.inf
        max_iter = 0
        deltas = dict()
        values = defaultdict(lambda: ([], []))
        for r in results:
            deltas[r.iteration] = r.time
            min_iter = min(min_iter, r.iteration)
            max_iter = max(max_iter, r.iteration)
            for name, value in r.metrics.items():
                values[name][0].append(r.iteration)
                values[name][1].append(value)

        # build metric groups
        available_metrics = natsorted(values,
                                      lambda x: -len(x.grouped_metrics))
        used_metrics = set()
        groups = []
        for metric in available_metrics:
            if metric not in used_metrics and not metric.is_val:
                group = [metric]
                used_metrics.add(metric)
                for grouped_metric in metric.grouped_metrics:
                    if grouped_metric not in used_metrics:
                        used_metrics.add(grouped_metric)
                        group.append(grouped_metric)

                group = natsorted(group, lambda m: m.name)

                if len(group) <= 3:
                    group.extend([m.val for m in group])
                else:
                    groups.append([m.val for m in group])

                groups.append(natsorted(group, lambda m: m.name))

        fig_height = 4 * len(groups) + .8
        plt.figure(figsize=(7, fig_height))

        plots = {}

        for i, metrics in enumerate(groups):
            plt.subplot(len(groups), 1, i + 1)

            xlim_delta = (max_iter - min_iter) * .05
            plt.xlim((min_iter - xlim_delta, max_iter + xlim_delta))

            ax1 = plt.gca()
            ax2 = ax1.twiny()
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels([format_time(deltas[i]) if i in deltas else ''
                                 for i in ax2.get_xticks()])
            ax2.tick_params(axis='x', labelsize='small')
            ax2.set_xlabel('Time [H:MM:SS]')

            plt.sca(ax1)
            plt.gca().set_xticklabels(['{:.0f} ({:.1f})'.format(
                                       i, i * mini_batch / n_samples)
                                       for i in plt.gca().get_xticks()])
            plt.gca().tick_params(axis='x', labelsize='small')
            plt.xlabel('Iteration (Epoch)')
            plt.grid()

            units = []
            ylim = None

            for metric in metrics:
                if metric in values:
                    plot_kwargs = dict(metric.plot_kwargs)

                    # complementary colors for validation plots
                    if metric.is_val and metric.org in plots:
                        prev_plot = plots[metric.org][0]
                        color = complement_color(prev_plot.get_color())

                        plot_kwargs['c'] = color
                        plot_kwargs.pop('color', None)

                    plots[metric] = plt.plot(*values[metric], label=metric,
                                             **plot_kwargs)

                    if metric.unit is not None and metric.unit not in units:
                        units.append(metric.unit)

                    if metric.ylim is not None:
                        if ylim is None:
                            ylim = metric.ylim
                        else:
                            ylim = [min(metric.ylim[0], ylim[0]),
                                    max(metric.ylim[1], ylim[1])]

            if units:
                plt.ylabel(' / '.join(reversed(units)))
            if ylim is not None:
                plt.ylim(ylim)

            # mark best result
            plt.plot([best_result.iteration] * 2, plt.ylim(), 'r--')
            plt.legend(loc='best', fancybox='false')

        plt.suptitle(
            'n_samples = {}   -   mini_batch = {}   -   n_val_samples = {}'
            .format(n_samples, mini_batch, n_val_samples))

        plt.tight_layout(rect=(0, 0, 1, (fig_height - .8) / fig_height))
        plt.savefig(str(output_file), bbox_inches='tight')
        plt.close()


class Loss(str, Enum):
    """Different loss functions."""

    PERCEPTRON = 'perceptron'
    """Basic perceptron loss, i.e., just the margin."""

    HINGE = 'hinge'
    """Only cases still not saturating a margin contribute."""

    LOG = 'log'
    """Smooth version of the hinge loss."""

    MCE = 'mce'
    """Smooth version of the minimal classification error loss using a
    sigmoid.
    """


class Regularization(str, Enum):
    """Different types of regularization."""

    L1 = 'l1'
    """Normal L1 regularization reducing the abs sum of the weights."""

    L2 = 'l2'
    """Normal L2 regularization reducing the squared sum of the weights."""


class SgdMetrics:
    """Metrics specific to `SgdMaxMarginLearning`."""

    TOTAL_RIVALS = Metric('total_rivals')
    """Number of total rivals current in the set, only available if
    `keep_rivals` is `True`.
    `"""

    SATISFIED_RIVALS = Metric('satisfied_rivals', plot_with=[TOTAL_RIVALS])
    """Number of rivals that are currently satisfied, i.e., the correct
    configuration has a lower energy than the rival.
    `"""

    SATISFIED_MARGIN_RIVALS = Metric('satisfied_margin_rivals',
                                     plot_with=[TOTAL_RIVALS,
                                                SATISFIED_RIVALS])
    """Number of rivals that have a higher energy, minus the margin,
    than the correct configuration.
    """


class SgdMaxMarginLearning(IterativeLearning):
    """Optimization of a max-margin-based loss function using SGD.

    Args:
        loss: The loss function that is used, see :class:`Loss` for
            available types.
        margin: Some loss functions require a margin parameter.
        unknown_energies_first: Whether to optimize the loss w.r.t. the
            unknown energies independently prior to doing the joint
            optimization.
        learning_rate: The learning rate used for the Adam optimizer.
        regularization: An optional type of regularization to apply.
        regularization_factor: The amount of regularization that is applied
            in addition to the data term.
        keep_rivals: This basically creates a growing list of rivals where
            the list is re-used in each iteration.
        min_weight: A hard min weight that is re-set after each iteration.
        nonzero_weight: Ensures using `log(1 + exp(w))` that weights never
            get zero.
        error_weighting: Whether to weight the energy difference with the
            reduction in soft error. If it is a `float`, it is used as
            normalization constant.
    """

    def __init__(self, criterion: Criterion,
                 loss: Loss = Loss.HINGE, margin: float = 0.1,
                 unknown_energies_first: bool = False,
                 learning_rate: float = 0.01,
                 regularization: Optional[Regularization] = None,
                 regularization_factor: float = 0.1,
                 keep_rivals: bool = False,
                 min_weight: Optional[float] = None,
                 nonzero_weight: bool = False,
                 error_weighting: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(FixedGraphSample, criterion, **kwargs)

        assert min_weight is None or min_weight >= 0
        assert min_weight is None or not nonzero_weight

        self.loss = loss
        """The loss function to use."""

        self.margin = float(margin)
        """Certain loss functions try to saturate a certain margin."""

        self.unknown_energies_first = unknown_energies_first
        """Whether to independently optimize the loss w.r.t. the unknown
        energies first before running the joint optimization.
        """

        self.learning_rate = learning_rate
        """The main learning rate used by the optimizer."""

        self.regularization = regularization
        """An optional regularization to apply."""

        self.regularization_factor = float(regularization_factor)
        """The amount of regularization to apply."""

        self.keep_rivals = keep_rivals
        """Whether to keep previously found rivals and re-use them in each
        iteration.
        """

        self.min_weight = 0. if min_weight is None else min_weight
        """Ensures that the weight does not go below this value."""

        self.nonzero_weight = nonzero_weight
        """Makes sure that the weight stays positive using log(1 + exp(w))."""

        if error_weighting is not None:
            assert isinstance(error_weighting, (int, float))
            error_normalization = error_weighting
            error_weighting = True
        else:
            error_weighting = False
            error_normalization = 1.

        #: Whether to weight the energy margin with the error.
        self.error_weighting = error_weighting

        #: A normalization constant to apply to the soft error.
        self.error_normalization = error_normalization

    def optimizer(self, graph: Graph) -> Generator[Dict[Metric, float],
                                                   OptBatch,
                                                   None]:
        log.debug('Building TensorFlow compute graph for the loss function')

        weights = graph.weights
        unknown_energies = graph.unknown_energies

        n_terms = weights.size

        graph_tf = tf.Graph()

        with graph_tf.as_default():
            # the variables we want to optimize
            weights_tf = tf.Variable(weights.astype(np.float64))
            unknown_energies_tf = tf.Variable(
                unknown_energies.astype(np.float64))

            if self.nonzero_weight:
                weights_tf = tf.log1p(tf.exp(weights_tf))

            # the individual term values
            corr_terms_tf = tf.placeholder(tf.float64, shape=(None, n_terms))
            incorr_terms_tf = tf.placeholder(tf.float64, shape=(None, n_terms))

            error_factors_tf = tf.placeholder(tf.float64, shape=(None, 1))

            # select either the energy terms or the unknown energy placeholder
            tiled_unknown_energies_tf = tf.broadcast_to(
                unknown_energies_tf, [tf.shape(corr_terms_tf)[0], n_terms])
            corr_selection_tf = tf.where(tf.is_nan(corr_terms_tf),
                                         tiled_unknown_energies_tf,
                                         corr_terms_tf)
            incorr_selection_tf = tf.where(tf.is_nan(incorr_terms_tf),
                                           tiled_unknown_energies_tf,
                                           incorr_terms_tf)

            # compute the actual state energy sums
            weights_mat_tf = tf.reshape(weights_tf, (n_terms, 1))
            corr_energies_tf = tf.matmul(corr_selection_tf, weights_mat_tf)
            incorr_energies_tf = tf.matmul(incorr_selection_tf, weights_mat_tf)

            margin_tf = corr_energies_tf - incorr_energies_tf

            if self.loss == Loss.PERCEPTRON:
                loss_tf = margin_tf
            elif self.loss == Loss.HINGE:
                loss_tf = tf.maximum(np.float64(0),
                                     self.margin * error_factors_tf
                                     + margin_tf)
            elif self.loss == Loss.MCE:
                loss_tf = 1.0 / (1.0 + tf.exp(margin_tf))
            elif self.loss == Loss.LOG:
                loss_tf = tf.log(1.0 + tf.exp(margin_tf))
            else:
                raise RuntimeError('Unknown loss function %s' % self.loss)

            if self.error_weighting:
                loss_tf = error_factors_tf * loss_tf

            loss_tf = tf.reduce_mean(loss_tf)

            # add regularization if specified
            if self.regularization is not None:
                if self.regularization == Regularization.L1:
                    regularization_tf = tf.reduce_sum(tf.abs(weights_tf))
                elif self.regularization == Regularization.L2:
                    regularization_tf = tf.reduce_sum(weights_tf**2)

                loss_tf += self.regularization_factor * regularization_tf

            # various optimization setups
            optimize_all_tf = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(loss_tf)
            optimize_unknown_energies_tf = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(
                loss_tf, var_list=[unknown_energies_tf])

            if not self.nonzero_weight:
                constrain_weights_tf = weights_tf.assign(
                    tf.maximum(weights_tf, tf.constant(self.min_weight,
                                                       tf.float64)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        optimize_tf = optimize_unknown_energies_tf \
            if self.unknown_energies_first else optimize_all_tf

        rivals = Rivals.empty() if self.keep_rivals else None

        with tf.Session(config=config, graph=graph_tf) as session:
            session.run(tf.global_variables_initializer())

            log.debug('Starting optimization')

            while True:
                # get the batch from the outside
                batch = (yield)

                # find the rivals for the current set of weights
                corr_energies, incorr_energies, error_reduction = \
                    self.infer_rival_energies(graph, batch.samples,
                                              rivals=rivals)

                replace_inf(corr_energies)
                replace_inf(incorr_energies)

                if optimize_tf is optimize_unknown_energies_tf \
                        and (batch.prev_result.iteration
                             - batch.best_result.iteration) > 100:
                    log.info('Did optimize unknown energies for 100 '
                             'iterations without improvement, '
                             'starting joint optimization')
                    optimize_tf = optimize_all_tf
                    unknown_energies_tf.load(
                        batch.best_params.unknown_energies, session)

                _, loss = session.run([optimize_tf, loss_tf], feed_dict={
                    corr_terms_tf: corr_energies,
                    incorr_terms_tf: incorr_energies,
                    error_factors_tf: error_reduction.reshape(
                        error_reduction.size, 1)
                })

                if not self.nonzero_weight:
                    session.run(constrain_weights_tf)

                weights, unknown_energies = session.run([weights_tf,
                                                         unknown_energies_tf])

                graph.weights = weights.flat
                graph.unknown_energies = unknown_energies.flat

                metrics = {Metrics.LOSS: loss,
                           SgdMetrics.TOTAL_RIVALS: corr_energies.shape[0]}

                for idx, pot in enumerate(graph.potentials):
                    if pot.unknown_energy is not None:
                        corr_energies[np.isnan(corr_energies[:, idx]),
                                      idx] = pot.unknown_energy
                        incorr_energies[np.isnan(incorr_energies[:, idx]),
                                        idx] = pot.unknown_energy

                corr_energies = (corr_energies @ weights).flatten()
                incorr_energies = (incorr_energies @ weights).flatten()

                metrics[SgdMetrics.SATISFIED_RIVALS] = \
                    (incorr_energies > corr_energies).sum()
                metrics[SgdMetrics.SATISFIED_MARGIN_RIVALS] = \
                    (incorr_energies
                     > corr_energies + self.margin * error_reduction).sum()

                yield metrics


def replace_inf(arr: np.ndarray, replacement: float = 1.0e10) -> None:
    """Replaces inf with a large energy substitute."""
    arr[arr == np.infty] = replacement

    max_energy = arr.max()
    if max_energy > replacement:
        log.error('The infinity replacement value %f is to low, the '
                  'largest energy value found was %f', replacement,
                  max_energy)


class CgMetrics:
    """Metrics generated by the `CgMaxMarginLearning` approach."""

    LOSS_FROM_SLACK = Metric('loss from slack', plot_with=[Metrics.LOSS])
    """The amount of loss that came from the slack variables."""

    NUMBER_OF_CONSTRAINTS_USED = Metric('number of constraints used')
    """The number of constraints used in this iteration step."""

    NUMBER_OF_CONSTRAINTS = Metric('number of constraints',
                                   plot_with=[NUMBER_OF_CONSTRAINTS_USED])
    """The number of constraints available."""

    NUMBER_OF_EASILY_FULFILLED_CONSTRAINTS = \
        Metric('number of constraints with 0 slack',
               plot_with=[NUMBER_OF_CONSTRAINTS_USED])
    """Number of constraints that didn't need any slack to be fulfilled."""

    DURATION_STEP = Metric('time for complete step')
    """The time the current step took."""

    DURATION_OPTIMIZATION = Metric('time for my code',
                                   plot_with=[DURATION_STEP])
    """The time my code took in this step."""

    DURATION_OPTIMIZATION_SUM = Metric('summed time my code ran')
    """The summed time my code ran overall. This smoothes over some
    performance anomalies.
    """


class CgMaxMarginLearning(IterativeLearning):
    """implements constraint generation to find optimal weights.
    Args:
        slack_factor:
            This constant weights the impact of the slack variables on the
            optimization goal. A high slack_factor leads to small slack values
            and mostly a better optimization.
        loss_func:
            This switches to the desired loss function. See below for
            an explanation of the different loss functions.
        margin:
            If dynamic_margin is false this constant gives the margin by which
            all constraints have to be fullfilled.
        opt_acc:
            This is the optimization accuracy for both constraints and loss
            function. The optimizer can violate Constraints up to this
            constant, which is why this is used in constraints.
        inf_rep:
            Full name is infinity replacement. I can get infinity values from
            outside and this is used to deal with them.
        aac:
            appen_all_constraints. If this is true all constraints are always
            appended. It might be reasonable to not do this, but True is a
            good default value.
        cnab:
            constraint not append barrier. If not all constraints are appended
            it is still unreasonable to leave out all that are fullfilled and
            this gives the margin by which they have to be fullfilled to be
            left out.
        allow_duplicates:
            If this is False I check for duplicate constraints. This should
            always be True except if there are very few contraints total and a
            high percentage of all constraints are actually added to the
            solver.
        solver:
            The solver the gekko suit uses. There are 3 solvers 1-3 and 0 uses
            all solver. There are 2 more solver which require a licence. Only
            1 works but it seems to be the best for us anyways.
        remote:
            gekko solves online by default, but I don't want
            to use that so default is False.
        d_margin:
            dynamic margin. If this is true I use the error reduction from
            outside to have a different margin depending on the error of
            given configuration.
        starting_values:
            -
        d_starting_values:
            -
        loss_multiplier:
            This constant is multiplied to the loss function. This is done so
            constraints and loss function are in the same general area which
            hopefully improves accuracy and convergence
        del_constraints:
            If this is True we completly delete fulfilled constraints. This
            should be used with great care.
        con_del_barrier:
            constraint delete barrier. See cnab.
        con_del_max:
            not implemented yet.
        error_red_func:
            There are different ways to use the error reduction term to get
            dynamic margins. Here the way is chosen. Still experimental.
    """

    def __init__(self, criterion: Criterion, slack_factor=1000, loss_func=1,
                 margin=1, opt_acc=1e-8, inf_rep=1e6, aac=True, cnab=1,
                 allow_duplicates=True, solver=1, remote=False, d_margin=False,
                 starting_value=0.5, d_starting_value=False,
                 loss_multiplier=1/100, del_con=False, con_del_barrier=0,
                 con_del_max=-1, error_red_func=1, debug_run=False,
                 **kwargs):

        # FIXME: beta j unabhängig von lambda j argumentieren. Travis
        if not debug_run:
            super().__init__(sample_cls=FixedGraphSample,
                             criterion=criterion,
                             **kwargs)
        self.slack_factor = slack_factor
        self.loss_function = loss_func
        self.margin = margin
        self.optimization_accuracy = opt_acc
        self.infinity_replacement = inf_rep
        self.append_all_constraints = aac
        self.constraint_not_append_barrier = cnab
        self.allow_duplicates = allow_duplicates
        self.solver = solver
        self.remote = remote
        self.dynamic_margin = d_margin
        self.starting_value = starting_value
        self.dynamic_starting_value = d_starting_value

        self.loss_multiplier = loss_multiplier
        self.delete_constraints = del_con
        self.constraint_delete_barrier = con_del_barrier
        self.constraint_delete_maximum = con_del_max
        self.error_reduction_f = error_red_func

        self.parameter_string = "Parameter-log: slack_factor: ", \
                                self.slack_factor, \
                                " \tloss_function:", self.loss_function, \
                                " \tmargin:", self.margin, \
                                " \toptimization_accuracy:",\
                                self.optimization_accuracy, \
                                " \tinfinity_replacement:",\
                                self.infinity_replacement,\
                                " \tappend_all_constraints:",\
                                self.append_all_constraints, \
                                " \tconstraint_not_append_barrier:",\
                                self.constraint_not_append_barrier,\
                                " \tallow_duplicates:", self.allow_duplicates,\
                                " \tsolver:", self.solver, " \tremote:",\
                                self.remote, " \tdynamic_margin:",\
                                self.dynamic_margin, " \tstarting_value:",\
                                self.starting_value,\
                                " \tdynamic_starting_value:",\
                                self.dynamic_starting_value,\
                                " \tloss_multiplier:", self.loss_multiplier,\
                                " \tdelete_constraints:",\
                                self.delete_constraints, \
                                " \tconstraint_delete_barrier:",\
                                self.constraint_delete_barrier,\
                                " \tconstraint_delete_maximum:",\
                                self.constraint_delete_maximum, \
                                " \terror_reduction_f:",\
                                self.error_reduction_f
        self.parameter_string = ''.join(str(s) for s in self.parameter_string)
        self.parameter_string += "\n"

        self.log_string = ""

        """
        In debug-runs parts of the usual output are abused for debugging.
        """
        self.debug_run = debug_run

        # searching for some obvious bugs
        if self.infinity_replacement * 10 > 1 / self.optimization_accuracy:
            log.error('optimization_accuracy is not small enough compared to '
                      'infinity_replacement')

        assert self.append_all_constraints or \
            self.constraint_not_append_barrier >= 0
        assert self.dynamic_margin or self.margin >= 0
        assert not self.delete_constraints or \
            self.constraint_delete_barrier >= 0

    # main method
    def optimizer(self, graph: Graph):
        # printing parameters to log
        log.info(self.parameter_string)

        # setting some initial variables
        # unknown_energies = graph.unknown_energies
        weights = graph.weights
        n_weights = weights.size  # number of weights/potential functions

        count = 1  # number of iteration

        # the maximum of all energies/corr_energies over all time.
        # This is compared with infinity_replacement.
        p_max = 0

        e_max = 0
        e_min = 10e10

        # If correct_energies have infinity in a potential then the
        # corresponding weight has to be zero to minimize the objective. I
        # implement this by setting the upper bound of that weight to 0. Gekoo
        # automatically removes is later.
        weight_upper_bound = np.full_like(np.ones(n_weights), np.infty)

        # some kind of infinity_replacement used
        some_kind_of_infinity_replacement_used = False

        #
        accumulated_time = 0

        # "fixes" warning
        energies = np.empty(1)
        margin_vector = np.empty(1)
        slack = np.empty(1)

        # t1 has to be initialized once and here is a reasonable place
        t1 = time.clock()

        # initial_unknown_energies = graph.unknown_energies

        while True:
            # Here a python generator is used. Effectively this loop is
            # repeated indefinitely while other code is executed after a loop.

            # get the batch from the outside
            batch = (yield)
            corr_energies, incorr_energies, error_reduction = \
                self.infer_rival_energies(graph, batch.samples,
                                          use_unknown_energies=True)

            # if use_unknown_energies == False, corr_energies and
            # incorr_energies will contain NaN values in places where the
            # beta/unknown_energies should be used you have to set
            # graph.unknown_energies before the yield

            t2 = time.clock()

            batch_size = corr_energies.shape[0]

            p_max = \
                max(p_max, np.amax(np.absolute(corr_energies
                                               [np.isfinite(corr_energies)])),
                    np.amax(np.absolute(
                        incorr_energies[np.isfinite(incorr_energies)])))

            e_max = max(e_max, np.amax(error_reduction))
            e_min = min(e_min, np.amin(error_reduction))

            some_kind_of_infinity_replacement_used = max(
                some_kind_of_infinity_replacement_used,
                np.amax(np.isinf(corr_energies)),
                np.amax(np.isinf(incorr_energies)))

            # energies contains every constraint ever added to the program.
            # Here this is ascertained
            if count > 1:
                energies = np.concatenate((energies, -corr_energies +
                                           incorr_energies), axis=0)
                if not self.allow_duplicates:
                    energies = np.unique(energies, axis=0)
            else:
                energies = -corr_energies + incorr_energies

            # If a potential function has infinite potential for a correct
            # configuration set the maximum corresponding weight to 0
            for k in range(n_weights):
                if not np.min(np.isfinite(
                        np.where(energies > 0, 0, energies)), axis=0)[k]:
                    some_kind_of_infinity_replacement_used = True
                    weight_upper_bound[k] = 0

            # sets +/-infty in energies to +/- infty_replacement
            energies[np.isinf(np.where(energies < 0, 0, energies))] = \
                self.infinity_replacement
            energies[np.isinf(np.where(energies > 0, 0, energies))] = \
                - self.infinity_replacement

            # The next 2 loops deal with the margin used for the constraints.
            # The idea is that margin vector contains the correct margin for
            # the corresponding potential differences stored in energies. I use
            # this even if I use a constant margin since this improves
            # readability of the code later on.
            if not self.dynamic_margin:
                error_reduction = np.full(batch_size, self.margin)
            else:
                error_reduction = \
                    self.error_reduction_function(error_reduction)

            if count == 1:
                margin_vector = error_reduction
            else:
                margin_vector = np.concatenate(
                    (margin_vector, error_reduction), axis=0)

            # some code that deletes margins
            if self.delete_constraints:
                delete_count = 0
                for k in range(energies.shape[0]):
                    if k >= energies.shape[0]:
                        continue

                    intermediate_value = self.constraint(
                        weights @ energies[k, :], 0, margin_vector[k])

                    if (intermediate_value >= self.constraint_delete_barrier
                        and (self.constraint_delete_maximum == -1
                             or delete_count <=
                             self.constraint_delete_maximum)):
                        energies = np.delete(energies, k, axis=0)
                        margin_vector = np.delete(margin_vector, k, axis=0)
                        k -= 1
                        delete_count += 1

            n_constr = energies.shape[0]

            # here gekko config starts. I initialize a new gekko instance every
            # iteration since I have trouble otherwise.
            # https://gekko.readthedocs.io/en/latest/global.html
            x = []

            # Initialize gekko. Per gekko default the problem is solved online.
            # Some options only work online apparently.
            gekko = GEKKO(remote=self.remote)
            # if I want to force online solving
            # m = GEKKO(remote=True)

            # Initialize variables
            for k in range(n_weights):
                if weight_upper_bound[k] == np.infty:
                    x.append(gekko.Var(lb=0, ub=1.0e20, value=weights[k]))
                else:
                    x.append(gekko.Var(lb=0, ub=0, value=weights[k]))

            # https://gekko.readthedocs.io/en/latest/global.html#imode
            gekko.options.IMODE = 3

            """
            "Solver options: 0 = Benchmark All Solvers, 1-5 = Available Solvers
            Depending on License. Solver 1-3 are free."
            The first is the only one working offline
            """
            gekko.options.SOLVER = self.solver

            # objective tolerance, default 1.0e-6
            gekko.options.OTOL = self.optimization_accuracy
            # restraint tolerance default 1.0e-6
            gekko.options.RTOL = self.optimization_accuracy

            # m.options.DIAGLEVEL = 6  # Input 0-10, default 0. Slows down the
            # program but gives more detailed output to find bugs.
            # m.options.MAX_ITER = 10000  # Default 100
            # m.options.MAX_TIME = 1.0e20  # Default 1.0e20
            # m.options.REDUCE = 0  # tries to reduce the complexity of the
            # problem by analysing data before sending it to the solver.
            # Doesn't work here. Default 0
            # m.options.WEB = 0 # disables creation of web interfaces

            n_gekko_equations = 0
            # adding constraints to gekko
            for k in range(n_constr):
                if self.append_all_constraints:
                    if k < n_constr - batch_size:
                        # if possible initialize slack with the result of the
                        # last iteration.
                        x.append(gekko.Var(lb=0, ub=1.0e20, value=slack[k]))
                    else:
                        x.append(gekko.Var(lb=0, ub=1.0e20,
                                           value=self.starting_value))

                    weights_times_energies = np.dot(
                        np.transpose(np.asarray(x[:n_weights])),
                        energies[k, :])

                    slack_variable = np.transpose(
                        np.asarray(x[n_weights + n_gekko_equations]))

                    gekko.Equation(self.constraint(weights_times_energies,
                                                   slack_variable,
                                                   margin_vector[k]) >= 0)

                    n_gekko_equations += 1
                    # in one line:
                    # m.Equation(self.constraint(np.dot(np.transpose(
                    # np.asarray(x[:n_weights])), energies[k, :]),
                    # np.transpose(np.asarray(x[n_weights +
                    # m._equations.__len__()])), margin_vector[k]) >= 0)

                # only add constraints that are not already fulfilled by cnab
                elif (self.constraint(weights @ energies[k, :], 0,
                                      margin_vector[k]) <
                      self.constraint_not_append_barrier):
                    # Note: dynamic starting value only implemented for aac
                    x.append(gekko.Var(lb=0, ub=1.0e20,
                                       value=self.starting_value))

                    weights_times_energies = np.dot(
                        np.transpose(np.asarray(x[:n_weights])),
                        energies[k, :])

                    slack_variable = np.transpose(
                        np.asarray(x[n_weights + n_gekko_equations]))

                    gekko.Equation(self.constraint(weights_times_energies,
                                                   slack_variable,
                                                   margin_vector[k]) >= 0)

            n_used_constr = n_gekko_equations
            if n_used_constr == 0:
                print("There are no constraints used, optimization stops")
                # break
            gekko.Obj(self.get_loss_func(np.asarray(x[:n_weights]),
                      np.transpose(np.asarray(x[n_weights:])),
                      n_used_constr))

            # Note: Gekko can crash and probably also throw exceptions.
            #  I should look into that
            # the actual solving with gekko
            gekko.solve(disp=False)  # Solve

            weights = np.transpose(np.asarray(x[:n_weights]))[0]
            slack = np.transpose(np.asarray(x[n_weights:]))[0]
            loss = self.get_loss_func(weights, slack, n_used_constr)

            self.log_string = "loss: ", self.d_as_r_s(loss, 10),\
                              "\t\tloss from slack: ",\
                              self.d_as_r_s(
                                  self.get_loss_func(np.zeros(1), slack,
                                                     n_used_constr), 10),\
                              "\nweights: ",\
                              np.array2string(weights, precision=3,
                                              separator=',',
                                              suppress_small=True)
            self.log_string = ''.join(str(s) for s in self.log_string)
            self.log_string += "\n"
            log.debug(self.log_string)
            # log.debug("slack: %s", slack)
            if (p_max > self.infinity_replacement / 100
                    and some_kind_of_infinity_replacement_used is True):
                log.warning("Infty replacement too small compared to %f",
                            p_max)

            if count % 5 == 0:
                log.debug("**************************************************")
                self.log_string = "Count: ", count, "\tp_max = ", p_max,\
                                  "\te_max = ", e_max,\
                                  "\te_min = ", e_min, ",\tslack_max = ",\
                                  max(slack), ", \taverage_slack_value =  ",\
                                  np.average(slack),\
                                  ",\tnumber_of_zero_constraints = ",\
                                  sum(np.where(slack == 0, 1, 0)),\
                                  "\t number_of_constraints = ",\
                                  energies.shape[0]
                self.log_string = ''.join(str(s) for s in self.log_string)
                log.debug(self.log_string)
                if np.min(weight_upper_bound) == 0:
                    log.debug("There are correct positions with infinite"
                              " potential, so these weights are set to 0:")
                    log.debug(np.where(weight_upper_bound == 0)[0])
                log.debug("**************************************************")
            count += 1

            graph.weights = weights

            # graph.unknown_energies = ....

            duration_step = time.clock() - t1
            duration_opt = time.clock() - t2
            accumulated_time += duration_opt
            t1 = time.clock()

            # this line exists since at home I use this to return the slack
            # variables without changing the syntax.
            if self.debug_run:
                graph.debug_slack = slack

            yield {Metrics.LOSS: loss,
                   CgMetrics.LOSS_FROM_SLACK:
                       self.get_loss_func(np.zeros(1), slack, n_used_constr),
                   # not used atm since I always append all constraints
                   # Metrics.NUMBER_OF_CONSTRAINTS: n_constr,
                   CgMetrics.NUMBER_OF_CONSTRAINTS_USED: n_used_constr,
                   CgMetrics.NUMBER_OF_EASILY_FULFILLED_CONSTRAINTS:
                       sum(np.where(slack < self.optimization_accuracy, 1, 0)),
                   CgMetrics.DURATION_STEP: duration_step,
                   CgMetrics.DURATION_OPTIMIZATION: duration_opt,
                   CgMetrics.DURATION_OPTIMIZATION_SUM: accumulated_time}

    def constraint(self, weights_times_energies, slack_variable, margin):
        """If slack-variable is less than margin this assures that the correct
        configuration has lower energy than the best rival configuration. I
        subtract self.optimization_accuracy to make sure that the constraints
        are really fulfilled and not only up to accuracy.
        """
        return (weights_times_energies + slack_variable -
                margin - self.optimization_accuracy)  # >= 0

    def loss_1(self, weights_sum: float, slack_sum, slack_count):
        """This corresponds to minimizing 1-norm of the weights + sum of
        slack. There are some weighting factors involved.
        """
        return (weights_sum + self.slack_factor * slack_sum / slack_count) * \
            self.loss_multiplier

    def loss_2(self, weights_squared_sum, slack_sum, slack_count):
        """This corresponds to minimizing 2-norm of the weights + sum of slack.
        There are some weighting factors involved.
        """
        return (weights_squared_sum + self.slack_factor * slack_sum /
                slack_count) * self.loss_multiplier

    def get_loss_func(self, weights, slack, slack_count):
        """Gets the relevant loss function."""
        if self.loss_function == 1:
            return self.loss_1(weights.sum(), slack.sum(), slack_count)
        elif self.loss_function == 2:
            return self.loss_2(weights @ weights, slack.sum(), slack_count)

    def error_reduction_function(self, x):
        """An error reduction function that makes sense to me.
        For x in [0,5000] the output is 0-6
        In a concrete test values ranged from 0.6 to 8.8.
        """
        if self.error_reduction_f == 1:
            return np.log(x / 10 + 1)
        raise RuntimeError('I guess this should never happen?')

    @staticmethod
    def d_as_r_s(x, space):
        """Returns double as rounded string with spaces to the right for better
        alignment.
        """
        return repr(int(10000 * x) / 10000).rjust(space)


class LearnableGraphSample(Sample):
    """Sample class used in `FullSgdMaxMarginLearning`.

    Provides a special candidate collection override to catch cases where
    no candidates were generated for some parts. This can be caused if the
    potential that generates the candidates is learned.
    """

    MIN_N_CANDIDATES: int = 5
    """Minimal number of candidates. If fewer than this, we randomly generate
    additional ones.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rng = np.random.RandomState(42)
        """PRNG used to generate random candidate positions, if needed."""

    def collect_candidates(self) -> Dict[str, np.ndarray]:
        candidates = super().collect_candidates()
        shape = self.image.data.shape[:self.graph.n_dims]

        for part, cands in list(candidates.items()):
            n_missing = self.MIN_N_CANDIDATES - len(cands)
            if n_missing > 0:
                log.warning('Did collect only %i of %i required candidates '
                            'for part %s in image %s, adding %i random '
                            'candidates', len(cands), self.MIN_N_CANDIDATES,
                            part, self.image.name, n_missing)
                cords = np.transpose([self.rng.randint(s, size=n_missing)
                                      for s in shape])
                candidates[part] = np.vstack((cands, cords))

        return candidates

    def infer(self, n_best: int = 1, *args, **kwargs) -> np.ndarray:
        # special case to train only unary localizer potentials, which do
        # benefit from creating states by using the cands independently
        if n_best > 1 and len(self.graph.connections) == 0:
            candidates = self.collect_candidates()
            min_cands = min(len(c) for c in candidates.values())

            pos = np.empty((min_cands, self.graph.n_parts, self.graph.n_dims))

            for part, cands in candidates.items():
                for j in range(min_cands):
                    pos[j, self.graph.part_idx[part]] = cands[j]

            return pos

        return super().infer(n_best=n_best, *args, **kwargs)

    def eligible_for_training(self) -> Union[bool, str]:
        return True


class FullSgdMaxMarginLearning(IterativeLearning):
    """Learns potential weights, unknown energies and potential parameters.

    The parameter optimization is only done for potentials implementing
    the `LearnablePotentialMixin` interface.

    Args:
        criterion: Our localization criterion
        session: The session we are optimizing in, this MUST BE the same
            session/graph you used to create your `LearnablePotential`s
        learning_rate: The learning rate to use for the Adam optimizer.
        margin_factor: The margin factor that is applied to the error which
            is than used as minimal energy difference to be enforced.
        learn_weights: Whether we should also optimize the used weights.
        learn_unknown_energies: Whether we should also optimize the used
            unknown energies.
        n_rivals: Number of rivaling states to optimize against.
    """
    def __init__(self, criterion: Criterion, session: tf.Session,
                 learning_rate: float = 1.0e-4,
                 margin_factor: float = 10.0,
                 learn_weights: bool = True,
                 learn_unknown_energies: bool = True,
                 n_rivals: int = 10,
                 **kwargs) -> None:
        super().__init__(sample_cls=LearnableGraphSample, criterion=criterion,
                         n_rivals=n_rivals, **kwargs)

        self.session = session
        """The TensorFlow session we are operating in."""

        self.learning_rate = learning_rate
        """The learning rate to use for the Adam optimizer."""

        self.margin_factor = margin_factor
        """The margin factor to apply to the error values."""

        self.learn_weights = learn_weights
        """Whether to optimize the weights."""

        self.learn_unknown_energies = learn_unknown_energies
        """Whether to optimize the unknown energies."""

    def correct_position(self, sample: Sample) -> np.ndarray:
        return sample.true_pos

    def optimizer(self, graph: Graph) -> Generator[Dict[Metric, float],
                                                   List[Sample],
                                                   None]:
        # this is statically filled an re-used every iteration, only the
        # numpy arrays below have to be re-filled, not recreated though!
        feed_dict = dict()

        # mapping from part to positions
        positions: Dict[str, np.ndarray] = dict()
        # energies for potentials that are not learnable, maps potential
        # instance to
        pot_energies: Dict[Potential, np.ndarray] = dict()

        # maximum number of states
        max_n_states = 1 + self.n_rivals

        # state specific errors that scale the margin
        errors: np.ndarray = np.zeros((max_n_states, graph.n_parts))

        # get the default weights
        weights = np.array([pot.weight for pot in graph.potentials])

        # select potentials that have an unknown energy associated
        assert all(np.isfinite(pot.unknown_energy)
                   for pot in graph.unknown_potentials.keys()), \
            'specify unknown energies for potentials that have mis. parts'
        unknown_energies = np.array([
            pot.unknown_energy if pot in graph.unknown_potentials else np.nan
            for pot in graph.potentials])

        with self.session.graph.as_default():
            weights_tf = tf.Variable(weights, trainable=self.learn_weights)
            unknown_energies_tf = tf.Variable(
                unknown_energies, trainable=self.learn_unknown_energies)

            # number of effectively used rivals, might differ during runs
            # depending on how the candidates were found and how inference
            # determined the rivals
            used_n_states_tf = tf.placeholder(tf.int32, shape=())

            # prepare the input positions, i.e., mapping from part to pos
            positions_tf = {}
            unknowns_tf = {}
            positions_shape = (max_n_states, graph.n_dims)
            for part in graph.parts:
                positions_tf[part] = tf.placeholder(np.float64,
                                                    positions_shape)
                positions[part] = np.full(positions_shape, np.nan)
                feed_dict[positions_tf[part]] = positions[part]

                positions_tf[part] = positions_tf[part][:used_n_states_tf]
                unknowns_tf[part] = tf.is_nan(positions_tf[part][:, 0])

            # compute the state energy by summing the individual potential
            # energies, or unknown energies
            # energy_sums_tf = tf.constant(0., tf.float64)
            all_energies_tf = []
            for pot_idx, pot in enumerate(graph.potentials):
                pot_positions_tf = [positions_tf[part]
                                    for part in pot.parts]

                # for learnable potentials, we actually build the compute
                # graph
                if isinstance(pot, LearnablePotentialMixin):
                    pot_energies_tf = tf.cast(pot.compute_tf(pot_positions_tf),
                                              tf.float64)
                else:
                    pot_energies_tf = tf.placeholder(tf.float64, max_n_states)
                    pot_energies[pot] = np.full(max_n_states, np.nan)

                    feed_dict[pot_energies_tf] = pot_energies[pot]

                    pot_energies_tf = pot_energies_tf[:used_n_states_tf]

                # select either the learnable unknown energy or the actual
                # potential value
                unknown_selector_tf = tf.reduce_any(
                    tf.stack([unknowns_tf[part] for part in pot.parts]),
                    axis=0)

                selected_pot_energies_tf = tf.where(
                    unknown_selector_tf,
                    tf.broadcast_to(unknown_energies_tf[pot_idx],
                                    (used_n_states_tf,)),
                    pot_energies_tf)

                assert_tf = tf.Assert(
                    tf.reduce_all(tf.is_finite(selected_pot_energies_tf)),
                    [selected_pot_energies_tf])

                selected_pot_energies_tf *= weights_tf[pot_idx]

                with tf.control_dependencies([assert_tf]):
                    all_energies_tf.append(selected_pot_energies_tf)

            # state and part specific errors
            errors_tf = tf.placeholder(tf.float64, errors.shape)
            feed_dict[errors_tf] = errors
            errors_tf = errors_tf[:used_n_states_tf]
            state_errors_tf = tf.reduce_sum(errors_tf, axis=1)

            # ensure that each potential optimizes itself, label vs state
            pot_loss_tf = []
            for pot_idx, (pot, pot_energies_tf) in enumerate(
                    zip(graph.potentials, all_energies_tf)):
                # makes only sense when we can actually change the pot params
                if not isinstance(pot, LearnablePotentialMixin):
                    continue

                pot_margins_tf = pot_energies_tf[0] - pot_energies_tf[1:]

                min_margin_tf = tf.constant(self.margin_factor
                                            / len(all_energies_tf), tf.float64)

                if pot.arity() == 1:
                    min_margin_tf = min_margin_tf * errors_tf[1:, pot_idx]

                pot_loss_tf.append(
                    tf.reduce_sum(tf.maximum(tf.constant(0., tf.float64),
                                             pot_margins_tf + min_margin_tf)))
            pot_loss_tf = sum(pot_loss_tf)

            all_energies_tf = tf.stack(all_energies_tf)
            energy_sums_tf = tf.reduce_sum(all_energies_tf, axis=0)

            # corr energy vs incorr energies
            corr_sum_tf = energy_sums_tf[0]
            incorr_sums_tf = energy_sums_tf[1:]
            margins_tf = corr_sum_tf - incorr_sums_tf

            # the state loss, using a maximum filter here does require using
            # a properly chosen margin, because else we use the correct state
            # energy, which might be very low already (relatively speaking) due
            # to perfectly matching binary values, though this state could
            # never be reached, because the localizer has not been trained
            # and the configuration would never show up in the candidates,
            # and the maximum filter would basically prevent to train it
            margin_factor_tf = tf.constant(self.margin_factor, tf.float64)
            loss_tf = tf.reduce_sum(tf.maximum(
                tf.constant(0., tf.float64),
                margins_tf + margin_factor_tf * state_errors_tf[1:]))

            auxiliary_loss = tf.reduce_sum(
                [tf.cast(pot.auxiliary_loss_tf(), tf.float64)
                 for pot in graph.potentials
                 if isinstance(pot, LearnablePotentialMixin)])

            # final loss is state loss plus label loss plus auxil. loss
            loss_tf = loss_tf + pot_loss_tf + auxiliary_loss

            # trainable tensors and our optimizer
            trainable_vars = tf.trainable_variables()
            with tf.name_scope('adam_optimizer'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)

                # strategy to iteratively compute gradients instead of doing
                # everything in one session.run, which may burst the memory if
                # the batch size is to large
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
                    add_gradients_tf = [agv.assign_add(gv[0]) for agv, gv in
                                        zip(accum_gradients_tf, gradients_tf)]
                update_weights_tf = optimizer.apply_gradients(
                    list(zip(accum_gradients_tf, trainable_vars)))

            if working_dir:
                tf.summary.histogram('state_errors', errors_tf)
                tf.summary.scalar('corr_energy', corr_sum_tf)
                tf.summary.histogram('incorr_energies', incorr_sums_tf)
                tf.summary.histogram('mean_pot_energies',
                                     tf.reduce_mean(all_energies_tf, axis=1))
                tf.summary.scalar('loss', loss_tf)

                summary_dir = f'tensorboard/{datetime.now():%Y%m%d-%H%M%S}'
                summary_writer_tf = tf.summary.FileWriter(
                    (working_dir / summary_dir).resolve(),
                    self.session.graph)
                summary_merge_tf = tf.summary.merge_all()
                summary_idx = 0

        # get all variables that we must initialize, i.e., only optimizer
        # specific variables and not potential variables for instance
        optimizer_variables_tf = [optimizer.get_slot(var, name)
                                  for name in optimizer.get_slot_names()
                                  for var in trainable_vars]
        # hack to get Adam specific variables (beta)
        optimizer_variables_tf.extend(optimizer._non_slot_dict.values())
        optimizer_variables_tf = [var for var in optimizer_variables_tf
                                  if var is not None]

        self.session.run(tf.variables_initializer(
            [weights_tf, unknown_energies_tf] + optimizer_variables_tf))

        # from tensorflow_large_model_support import LMS
        # lms_obj = LMS({'adam_optimizer'})
        # lms_obj.run(graph=self.session.graph)
        # self.session.run(tf.global_variables_initializer())

        # iterations loop
        while True:
            batch = (yield)

            # clear gradients before we do anything
            self.session.run(zero_accum_gradients_tf)

            loss = 0

            # process sample after sample, accumulating the loss
            for sample in batch:
                # we copy the initial feed dict and add sample/potential
                # specific inputs, i.e, the image
                sample_feed_dict = dict(feed_dict)
                for pot in graph.potentials:
                    if isinstance(pot, LearnablePotentialMixin):
                        pot.feed_values(sample_feed_dict, sample.image)

                corr_position = sample.true_pos.reshape(1, graph.n_parts,
                                                        graph.n_dims)
                incorr_positions = sample.infer(n_best=self.n_rivals + 10)
                all_positions = np.concatenate(
                    (corr_position, incorr_positions), axis=0)

                energies = sample.compute_energies(all_positions)
                energies = energies.sum(axis=1)

                # find rivaling positions
                used_pos_idx = 0
                for pos_idx, (pos, energy) in enumerate(zip(all_positions,
                                                            energies)):
                    result = self.criterion.evaluate(
                        sample.image, graph.parts, pos, sample.true_pos)

                    if pos_idx == 0:
                        log.debug('Correct state: energy = %f', energy)
                    else:
                        # FIXME: Better way of selecting rivals

                        log.debug('Incorrect state %i: energy=%f incorrect '
                                  'parts=%i error_sum=%f error_avg=%f',
                                  pos_idx, energy, np.sum(~result.correct),
                                  np.sum(result.error), np.mean(result.error))

                        if result.correct.all():
                            log.warning('Ignoring best state %i, since it is '
                                        'fully correct', pos_idx)
                            continue

                    errors[used_pos_idx] = result.error

                    for part, p in zip(graph.parts, pos):
                        positions[part][used_pos_idx] = p

                    used_pos_idx += 1
                    if used_pos_idx == self.n_rivals + 1:
                        break

                if used_pos_idx == 1:
                    log.error('Found no rivals for sample, ignoring it')
                    continue
                elif used_pos_idx < self.n_rivals + 1:
                    log.warning('Using only %i rivals instead of %i',
                                used_pos_idx - 1, self.n_rivals)

                sample_feed_dict[used_n_states_tf] = used_pos_idx

                # populate non-learnable potential energies
                for pot, values in pot_energies.items():
                    pot_positions = [positions[part][:used_pos_idx]
                                     for part in pot.parts]
                    valid_pos = np.logical_not(np.any(np.isnan(pot_positions),
                                                      axis=(0, 2)))
                    values[valid_pos] = pot.compute(sample.image,
                                                    [pos[valid_pos] for pos
                                                     in pot_positions])

                # compute and store gradients for this sample
                fetches = [add_gradients_tf, loss_tf]
                if working_dir:
                    fetches.append(summary_merge_tf)

                results = self.session.run(fetches, feed_dict=sample_feed_dict)
                log.debug('Sample loss: %f', results[1])
                loss += results[1]

                if working_dir:
                    summary_writer_tf.add_summary(results[2], summary_idx)
                    summary_idx += 1

            # here we do the actual optimization step
            self.session.run(update_weights_tf)

            if self.learn_weights:
                # ensure that potential weights are positive
                weights = self.session.run(weights_tf)
                weights = np.maximum(weights, 0)
                weights_tf.load(np.maximum(weights, 0), self.session)

                graph.weights = weights

            if self.learn_unknown_energies and graph.unknown_potentials:
                unknown_energies = self.session.run(unknown_energies_tf)
                for pot, i in graph.unknown_potentials.items():
                    pot.unknown_energy = unknown_energies[i]
                    assert np.isfinite(pot.unknown_energy)
                log.debug('Unknown energies: %s',
                          ' '.join('%i=%f' % (i, pot.unknown_energy)
                                   for pot, i
                                   in graph.unknown_potentials.items()))

            # clear cached values etc.
            for pot in graph.potentials:
                if isinstance(pot, LearnablePotentialMixin):
                    pot.changed()

            yield {Metrics.LOSS: loss}


# FIXME: use the graph.plot .. method
# def plot_potential_energies(iteration):
#     if not working_dir \
#             or (iteration > 0
#                 and not isinstance(self, FullSgdMaxMarginLearning)):
#         return
#
#     def find_plot_image(samples, pot):
#         try:
#             return max([sample.image for sample in samples
#                         if all(part in sample.image.objects[0].parts
#                                for part in pot.parts)],
#                        key=lambda x: len(x.objects[0].parts))
#         except ValueError:
#             return None
#
#     def plot_pot(idx, pot, image, prefix):
#         pot_dir_name = 'pot{}_{}_{}'.format(
#             idx, pot.__class__.__name__, '-'.join(pot.parts))
#         plot_dir = working_dir / 'potential_energies' / prefix \
#                    / pot_dir_name
#         plot_dir.mkdir(exist_ok=True, parents=True)
#         pot.plot_energies(image, plot_dir, iteration)
#
#     for i, pot in enumerate(graph.potentials):
#         # non-learnable potentials do not change their energy over time
#         if iteration > 0 and not isinstance(pot,
#                                             LearnablePotentialMixin):
#             continue
#
#         train_plot_image = find_plot_image(train_samples, pot)
#         val_plot_image = find_plot_image(val_samples or [], pot)
#
#         if train_plot_image:
#             plot_pot(i, pot, train_plot_image, 'train')
#         if val_plot_image:
#             plot_pot(i, pot, val_plot_image, 'val')


def create_tf_session() -> tf.Session:
    """Creates a TensorFlow session that uses fewer memory.

    This is achieved by disabling various optimizations, that create duplicated
    graphs and thus block some GPU memory.

    Return: A new TensorFlow `Session`.
    """
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)

    rewrite_options = RewriterConfig(disable_model_pruning=True,
                                     constant_folding=RewriterConfig.OFF,
                                     shape_optimization=RewriterConfig.OFF,
                                     remapping=RewriterConfig.OFF,
                                     memory_optimization=RewriterConfig.OFF)

    graph_options = tf.GraphOptions(optimizer_options=optimizer_options,
                                    rewrite_options=rewrite_options)

    config = tf.ConfigProto(graph_options=graph_options)

    return tf.Session(config=config)

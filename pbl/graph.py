"""Provides a **coordinate-based** Markov network abstraction.

Everything is driven by the `Graph` class, which is basically a collection of
`Potential`s. You can use it to build your model and perform inference on it.

For concrete `Potential`s (given that this is just an abstract representation)
check the `pbl.potentials` module.
"""


import gzip
import hiwi
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import SimpleITK as sitk

from collections import defaultdict
from hiwi import Image
from natsort import natsorted
from pathlib import Path
from time import time
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Union, IO
from typing import Tuple

from .evaluation import Criterion
from .potentials import Potential, select_reference_images
from .inference import EnergyGraph, EnergyPotential, Solver, ExactTreeSolver
from .inference import GraphStructure, AStarSolver
from .utils import working_dir


log = logging.getLogger(__name__)


class TestResult(NamedTuple):
    """Result of applying the graph to an image."""

    positions: Dict[str, Optional[np.ndarray]]
    """Map from part name to predicted position."""

    candidates: Dict[str, np.ndarray]
    """The used candidates."""

    physical_positions: Dict[str, Optional[np.ndarray]]
    """Map from part name to predicted position in world coordinates."""

    def pred_pos(self, parts: List[str]) -> np.ndarray:
        """Creates an array for the given `parts` where the first axis
        corresponds to the part id, absence indicated by `nan`s."""

        n_dims = next(cands.shape[1] for cands in self.candidates.values())

        pred_pos = np.full((len(parts), n_dims), np.nan)
        for idx, part in enumerate(parts):
            pos = self.positions.get(part)
            if pos:
                pred_pos[idx] = pos

        return pred_pos


class InfResult(NamedTuple):
    """Result of an inference run over the coordinate-labeled graph."""

    positions: Dict[str, Optional[np.ndarray]]
    """Mapping from part name to (optional) position."""

    energy: float
    """The energy of this configuration."""

    scaling: float
    """The used scaling."""

    time_energies: float
    """Duration in seconds that was needed to compute the energies."""

    time_inference: float
    """Duration of the inference over the pre-computed graph."""


class Graph:
    """A graphical model with coordinates as labels.

    Represents a graphical model that contains potentials that depend on
    the position(s) of certain parts.

    The structure is defined by the names of the parts used in different
    `Potential`s that were added to the graph.

    Args:
        potentials: The potentials this graph is composed of.
        n_dims: The image dimensionality to operate on.
        support_unknown: Set of parts that should support the "unknown" label.
        orientation: If given, we make sure to re-orient images to this
            orientation before actually processing them.
        spacing: Similar to orientation, we resample images to match this
            spacing before processing.
    """

    def __init__(self, potentials: List[Potential], n_dims: int = 2,
                 support_unknown: set = frozenset(),
                 orientation: Optional[str] = None,
                 spacing: Optional[np.ndarray] = None) -> None:
        assert len(potentials) > 0

        self.n_dims = n_dims
        """The image dimensionality to operate on."""

        self.orientation = orientation
        """An optional orientation to convert images to prior to processing."""

        self.spacing = spacing
        """An optional spacing to resample images to prior to processing."""

        self.use_mm = True

        self.potentials: List[Potential] = potentials
        """List of all potentials of the graph."""

        self.parts: List[str] = natsorted(set(part for pot in self.potentials
                                              for part in pot.parts))
        """List of individual parts derived from `potentials` and sorted
        "naturally".
        """

        self.part_idx: Dict[str, int] = {part: i for i, part
                                         in enumerate(self.parts)}
        """Maps part name to index."""

        assert all(part in self.parts for part in support_unknown)
        self.support_unknown = support_unknown
        """Parts that support the "unkown" label."""

        self.connections = natsorted(set(tuple(natsorted(c))
                                         for p in self.potentials
                                         for c in itertools.product(p.parts,
                                                                    p.parts)
                                         if c[0] != c[1]))
        """List of connections between parts derived from `potentials`
        and encoded as ('part_a', 'part_b').

        Note, the connection just states that the two parts are contained
        in any of the potentials, might be binary or larger potentials.
        """

        self.unknown_potentials: Dict[Potential, int] = \
            {pot: i for i, pot in enumerate(self.potentials)
             if any(part in self.support_unknown for part in pot.parts)}
        """Mapping of potential to index for all potentials that are connected
        to a part that should support the 'unknown' label.
        """

    @property
    def n_parts(self):
        """Number of parts to localize."""
        return len(self.parts)

    @property
    def n_potentials(self):
        """Number of potentials."""
        return len(self.potentials)

    @property
    def is_forest(self) -> bool:
        """Tests whether the current graph structure is a forest (tree)."""
        return self.structure.is_forest

    @property
    def is_chain(self) -> Optional[List[str]]:
        """Tests whether the current graph structure is a chain and returns it
        if it is one.
        """
        adjacency = defaultdict(set)
        for a, b in self.connections:
            adjacency[a].add(b)
            adjacency[b].add(a)

        chain = []

        try:
            chain.append(next(a for a, b in adjacency.items() if len(b) == 1))
        except StopIteration:
            return

        while True:
            neighbor = adjacency[chain[-1]] - set(chain)
            if len(neighbor) != 1:
                break
            chain.append(neighbor.pop())

        if len(chain) != self.n_parts:
            return

        return chain

    @property
    def structure(self) -> GraphStructure:
        adjacency = np.zeros((self.n_parts, self.n_parts), dtype=bool)

        for pot in self.potentials:
            assert len(pot.parts) < 3
            for a, b in itertools.product(pot.parts, pot.parts):
                if a != b:
                    i, j = self.part_idx[a], self.part_idx[b]

                    adjacency[i, j] = adjacency[j, i] = True

        return GraphStructure(adjacency)

    @property
    def weights(self) -> np.ndarray:
        return np.array([p.weight for p in self.potentials])

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        for weight, pot in zip(value, self.potentials):
            pot.weight = weight

    @property
    def unknown_energies(self) -> np.ndarray:
        return np.array([p.unknown_energy if p.unknown_energy is not None else
                         np.nan for p in self.potentials])

    @unknown_energies.setter
    def unknown_energies(self, value: np.ndarray) -> None:
        for pot, unknown_energy in zip(self.potentials, value):
            pot.unknown_energy = None if np.isnan(unknown_energy) else \
                unknown_energy

    def test(self, image: Image,
             parts: Optional[List[str]] = None,
             scalings: List[float] = [1]) -> TestResult:
        """Applies the graph to the given `image`.

        The image is automatically converted into the required coordinate
        system and resampled if necessary. However, the results returned
        correspond to the initially given image.

        Arguments:
            image: The image, does not need annotations, that we want to test.
            parts: Optional list of parts we are only interested in, if given,
                a subgraph is created and used for inference.
        """
        parts = set(parts or self.parts)

        org_image = image
        org_image_itk = sitk.ReadImage(str(org_image.path))

        image_itk, index_tx = self.prepare_image(org_image_itk)
        # the dummy part in the path is needed due to caching of some pots
        image = hiwi.Image(data=sitk.GetArrayViewFromImage(image_itk),
                           path=str(org_image.path) + '_DUMMY',
                           spacing=image_itk.GetSpacing())

        candidates = self.collect_candidates(image)

        for part, cands in candidates.items():
            if len(cands) == 0 and part not in self.support_unknown \
                    and part in parts:
                log.warning('Did not find any candidate for part %s, '
                            'using unknown label for it', part)
                parts.remove(part)

        graph = self.subgraph(list(parts))
        solver = ExactTreeSolver() if graph.is_forest else AStarSolver()
        inf_result = graph.infer(image, candidates, solver,
                                 support_unknown=graph.support_unknown,
                                 scalings=scalings)

        log.debug('CRF inference time %.3f ms plus %.3f ms energy '
                  'computation time', inf_result.time_inference * 1000,
                  inf_result.time_energies * 1000)

        positions = {n: None if p is None else index_tx(p)
                     for n, p in inf_result.positions.items()}
        candidates = {n: index_tx(c) for n, c in candidates.items()}
        physical_positions = {
            part: org_image_itk.TransformContinuousIndexToPhysicalPoint(
                pos[::-1])
            if pos is not None else None
            for part, pos in positions.items()}

        return TestResult(positions=positions,
                          candidates=candidates,
                          physical_positions=physical_positions)

    def subgraph(self, parts: List[str]) -> 'Graph':
        """Derives a sub-graph containing only the given `parts`."""
        assert all(p in self.parts for p in parts)

        potentials = [pot for pot in self.potentials
                      if all(p in parts for p in pot.parts)]

        support_unknown = set(s for s in self.support_unknown
                              if s in parts)

        return Graph(potentials, n_dims=self.n_dims,
                     support_unknown=support_unknown,
                     orientation=self.orientation, spacing=self.spacing)

    def prepare_annotated_image(self, image: Image) -> Image:
        # TODO: We need this for the training data!
        raise NotImplementedError

    def prepare_image(self, image_itk: sitk.Image) \
            -> Tuple[sitk.Image, Callable[[np.ndarray], np.ndarray]]:
        """Makes sure the image is in the proper orientation and
        resolution for testing.
        """
        org_image_itk = image_itk
        spacing = np.array(self.spacing)

        # re-orientating, if necessary
        if self.orientation is not None:
            orientation = hiwi.find_anatomical_orientation(image_itk)
            if orientation != self.orientation:
                log.info('Re-orienting image from %s to %s', orientation,
                         self.orientation)
                image_itk = hiwi.change_anatomical_orientation(
                    image_itk, self.orientation)

        # resampling, if necessary
        if self.spacing is not None \
                and (np.abs(image_itk.GetSpacing()[::-1] - spacing)
                     > 0.01).any():
            log.info('Resampling image from spacing %s to %s',
                     image_itk.GetSpacing(), tuple(spacing[::-1]))
            scaling = image_itk.GetSpacing()[::-1] / spacing
            image_itk = hiwi.resample_image(image_itk, scaling)

        if org_image_itk is not image_itk:
            log.info('New image has a size of %s, the orientation %s and '
                     'spacing %s', image_itk.GetSize(),
                     hiwi.find_anatomical_orientation(image_itk),
                     image_itk.GetSpacing())

        def index_tx(position: np.ndarray) -> np.ndarray:
            position = np.asarray(position)
            is_one = position.ndim == 1

            if is_one:
                position = [position]

            new_position = []

            for pos in position:
                if np.isnan(pos).all():
                    new_position.append(pos)
                    continue

                world_pos = image_itk.TransformContinuousIndexToPhysicalPoint(
                    pos[::-1])
                org_index = org_image_itk \
                    .TransformPhysicalPointToContinuousIndex(world_pos)[::-1]
                new_position.append(org_index)

            if is_one:
                new_position = new_position[0]

            return np.array(new_position)

        return image_itk, index_tx

    def infer(self, image: Image, candidates: Dict[str, np.ndarray],
              solver: Solver, use_weights: bool = True,
              support_unknown: Union[bool, List[str]] = False,
              scalings: List[float] = [1]) \
            -> InfResult:
        """Infers the most likely selection of candidates.

        Args:
            image: An image to compute the energies for w.r.t. to the given
                `candidates`.
            candidates: A mapping from part to an array containing candidates
                for the corresponding part.
            solver: The solver to use.
            use_weights: Whether to use the potential weights as factor for
                the corresponding energies or not.
            support_unknown: Whether to allow the unknown state and if so
                for which parts. `True` allows it for all parts.
            scalings: List of possible scalings to use. Depends on the
                potentials whether this has an effect.

        Returns:
            An `InferenceResult` providing the positions and meta information.
        """
        if type(support_unknown) is bool:
            support_unknown = self.parts if support_unknown else []

        time_energies = 0
        time_inference = 0

        state, energy, scaling = None, np.inf, None

        for s in scalings:
            t = time()
            energy_graph = self.compute_energies(
                image, candidates, use_weights, support_unknown, scaling=s)
            time_energies += time() - t

            t = time()
            state_tmp, energy_tmp = solver.infer(energy_graph)
            time_inference += time() - t

            if len(scalings) > 1:
                log.debug('- scaling = %.3f: energy = %f', s, energy_tmp)

            if energy_tmp < energy:
                state, energy, scaling = state_tmp, energy_tmp, s

        if len(scalings) > 1:
            log.debug('Use scaling: %.3f', scaling)

        positions = {}
        for part, label in zip(self.parts, state):
            if part in support_unknown:
                label -= 1

            positions[part] = candidates[part][label].tolist() \
                if label >= 0 else None

        return InfResult(positions=positions,
                         energy=energy,
                         scaling=scaling,
                         time_energies=time_energies,
                         time_inference=time_inference)

    def compute_energies(self, image: Image, candidates: Dict[str, np.ndarray],
                         use_weights: bool = True,
                         support_unknown: Set[str] = frozenset(),
                         unknown_energy: Optional[float] = None,
                         **kwargs) \
            -> EnergyGraph:
        """Computes an `EnergyGraph` for a concrete case.

        The variable ids are the indexes of the respective parts in the `parts`
        list and the label ids are the indexes of the candidates of the
        respective parts.

        Beware:
            If `use_weights` is `True`, we drop zero-weighted potentials.

        Args:
            image: An image to compute the energies for, w.r.t. to the given
                `candidates`.
            candidates: A mapping from part to an array containing candidates
                for the corresponding part. A `None` indicates the unknown
                state and makes use of `Potential`s `unknown_energy`.
            use_weights: Whether to respect the potential weights when
                computing the energy values.
            support_unknown: Whether to introduce an artificial 0-th state that
                corresponds to unknown. We make use of `Potential`s
                `unknown_energy` and fail if it is `None`. Specify the names of
                the parts this state should be introduced for.
            unknown_energy: An override value to use as unknown energy for all
                potentials.

        Returns:
            An `EnergyGraph` which you can use to perform inference on.
        """
        energy_potentials = []

        for potential in self.potentials:
            if use_weights and potential.weight == 0:
                continue

            potential_candidates = tuple(candidates[part] for part
                                         in potential.parts)

            shape = tuple(len(c) for c in potential_candidates)

            combinations = np.array(list(np.ndindex(shape)))
            combinations.shape = (len(combinations), len(potential.parts))

            if len(combinations) == 0:
                values = np.zeros(shape, np.float64)
            else:
                positions = tuple(potential_candidates[i][v] for i, v
                                  in enumerate(combinations.T))

                # perform the actual computation
                values = potential.compute(image, positions, **kwargs)
                assert not np.isnan(values).any()

                values = np.asarray(values, np.float64)
                values.shape = shape

            # introduce additional 0-th state that corresponds to unknown
            unknown_parts = [p for p in potential.parts
                             if p in support_unknown]
            if unknown_parts:
                pot_unknown_energy = unknown_energy

                if pot_unknown_energy is None:
                    pot_unknown_energy = potential.unknown_energy

                if pot_unknown_energy is None:
                    raise RuntimeError('Parts {} can not make use of unknown '
                                       'label, since potential {} does not '
                                       'specify an unknown energy'
                                       .format(', '.join(unknown_parts),
                                               potential))

                new_shape = tuple(len(candidates[p]) + (p in support_unknown)
                                  for p in potential.parts)
                new_values = np.full(new_shape, pot_unknown_energy,
                                     dtype=np.float64)

                old_region = tuple(slice(int(p in support_unknown), None)
                                   for p in potential.parts)
                new_values[old_region] = values

                values = new_values

            # as a last step, we apply the weighting
            if use_weights and potential.weight != 1:
                values *= np.float64(potential.weight)

            variables = [self.parts.index(p) for p in potential.parts]

            energy_potentials.append(EnergyPotential(variables, values))

        # safety check that all variables are defined, i.e., have a potential
        # associated
        if use_weights:
            undefined_parts = []

            for idx, part in enumerate(self.parts):
                had_pot = any(part in p.parts for p in self.potentials)
                has_pot = any(idx in p.variables for p in energy_potentials)

                if had_pot and not has_pot:
                    undefined_parts.append(f'{idx}:{part}')

            if undefined_parts:
                log.error('Some variables are not associated with a potential '
                          'after dropping zero-weighted ones: %s',
                          ', '.join(undefined_parts))

        n_labels = [len(candidates[p]) + (p in support_unknown)
                    for p in self.parts]

        return EnergyGraph(n_labels, energy_potentials)

    def collect_candidates(self, image: Image) -> Dict[str, np.ndarray]:
        """Collects candidates potentially proposed by some potentials.

        Iterates the potentials and collects the candidates proposed by them
        (`Potential.propose_candidates()`).

        Args:
            image: The image to propose candidates for.

        Returns:
            A (potentially empty or not fully specified) dictionary mapping
            part name to an array of candidates.
        """
        candidates = dict()

        for potential in self.potentials:
            for part, new_cands in potential.propose_candidates(image).items():
                assert new_cands.ndim == 2

                old_cands = candidates.get(part)
                if old_cands is not None:
                    new_cands = np.vstack((old_cands, new_cands))
                candidates[part] = new_cands

        missing_parts = ', '.join(p for p in self.parts if p not in candidates)

        if missing_parts:
            log.error('Could not collect candidates for all parts: %s',
                      missing_parts)

        return candidates

    def estimate_unknown_energies(self, images: List[Image],
                                  criterion: Criterion) -> None:
        """Uses a simple heuristic to estimate unknown energy levels.

        Args:
            images: The images to use in order to estimate the energy levels.
            criterion: The target criterion.
        """
        unknown_potentials = list(self.unknown_potentials.keys())

        @working_dir.cache('estimated_unknown_energies')
        def estimate_unknown_energies():
            corr_energies = [[] for _ in range(len(unknown_potentials))]

            for image in images:
                candidates = self.collect_candidates(image)

                for i, pot in enumerate(unknown_potentials):
                    correct_configs = criterion.correct_configurations(
                        image, pot.parts, candidates)

                    # remove configurations with nans
                    positions = correct_configs[
                        np.isfinite(correct_configs).all(axis=(1, 2))]

                    if len(positions) == 0:
                        continue

                    positions = np.unique(positions, axis=0)

                    energies = pot.compute(image,
                                           positions.transpose((1, 0, 2)))
                    energies = energies[np.isfinite(energies)]
                    corr_energies[i].extend(energies.astype(np.float64))

            unknown_energies = []

            log.debug('Deriving unknown energies from sets of correct energy '
                      'samples:')

            for pot, energies in zip(unknown_potentials, corr_energies):
                if len(energies) == 0:
                    log.error('Did not find a single correct energy for the '
                              'parts %s, thus we can not estimate the unknown '
                              'energy for potential %s', ', '.join(pot.parts),
                              pot)
                    unknown_energies.append(None)
                    continue

                unknown_energy = np.mean(energies)
                unknown_energies.append(unknown_energy)

                log.debug(' - %s: samples=%i min=%f max=%f median=%f mean=%f',
                          pot, len(energies), np.min(energies),
                          np.max(energies), np.median(energies),
                          np.mean(energies))

                # we plot this in order to better understand how correct
                # energies are distributed
                if working_dir and len(energies) > 1:
                    with working_dir.join('correct_energy_dist'):
                        plt.title(f'{len(energies)} energy values')
                        try:
                            sns.kdeplot(energies)
                        except np.linalg.LinAlgError:
                            pass
                        plt.plot(energies, np.zeros_like(energies), 'rx')
                        plt.plot([unknown_energy, unknown_energy], plt.ylim(),
                                 'g-')

                        name = f'{self.potentials.index(pot)}_{pot}.pdf'
                        plt.savefig(str(working_dir / name))
                        plt.close()

            return unknown_energies

        for pot, unknown_energy in zip(unknown_potentials,
                                       estimate_unknown_energies()):
            pot.unknown_energy = unknown_energy

    def plot_potential_energies(self, images: List[Image], output_dir: Path,
                                fixed_image_name: Optional[str] = None,
                                max_dist: float = 10.) \
            -> None:
        """Plots all potential energies w.r.t. some reference image.

        If missing parts are involved, multiple reference image might be taken
        from the list of `images` in case there are annotations missing. The
        images are chosen such that a minimal amount of reference images is
        needed to plot all potentials.

        Args:
            images: List of potential reference images to compute the energies
                for.
            output_dir: Directory where to store the resulting images.
            fixed_image_name: If this is given, creates potential-specific
                folders inside `output_dir` and stores `fixed_image_name`-named
                images in there. The default is to save potential-specific
                named images in `output_dir` if this is `None`.
            max_dist: The maximal distance (localization tolerance).
        """
        reference_images = select_reference_images(images, self.potentials)

        output_dir.mkdir(exist_ok=True, parents=True)

        # here starts the actual plotting
        for image, pots in reference_images:
            candidates = self.collect_candidates(image)

            for pot in pots:
                potential_name = f'{self.potentials.index(pot)}_{pot}'
                if fixed_image_name is None:
                    output_path = output_dir / f'{potential_name}.pdf'
                else:
                    pot_dir = output_dir / potential_name
                    pot_dir.mkdir(exist_ok=True, parents=True)
                    output_path = pot_dir / fixed_image_name

                pot.plot_energies(image, output_path, candidates=candidates,
                                  max_dist=max_dist)

    def dump(self, path_or_fp: Union[str, Path, IO]) -> None:
        """Saves the current instance to a given file.

        Args:
            path_or_fp: Either a path or a file-like object to store the
                instance in.
        """
        with gzip.open(path_or_fp, mode='wb',  compresslevel=1) as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)  # nosec

    @classmethod
    def load(cls, path_or_fp: Union[str, Path, IO]) -> 'Graph':
        """Loads a `Graph` from the given file.

        Args:
            path_or_fp: Either a path to a model or a file-like object
                containing a model.

        Returns: An instance of `Graph`.
        """
        with gzip.open(path_or_fp, 'rb') as fp:
            model = pickle.load(fp)  # nosec

            if type(model) != cls:
                raise EnvironmentError("not a Graph model")

        return model

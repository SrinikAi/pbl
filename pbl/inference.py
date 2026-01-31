"""
This module provides components to **model explicit Markov networks** and to
perform **inference (MAP)** on them.

Explicit Markov networks
------------------------

A n explicit Markov network is described by an `EnergyGraph`, which itself
contains one or more `EnergyPotential`. This is an index-based representation
of a finite state-space Markov network with explicit energy values.

MAP inference
-------------

We provide multiple inference strategies, most of which are powered by the
[OpenGM] library. Two generic interfaces are provided by `Solver` and
`NBestSolver`, the latter one supporting returning the `n`-best states.

**`NBestSolver`s:**

- `GibbsSolver` (approximate, arbitrary arity)
- `AStarSolver` (exact, arbitrary arity)

**Normal `Solver`s**

- `ExactTreeSolver` (exact, binary, forest)
- `LazyFlipperSolver` (approximate, arbitrary arity)

[OpenGM]: https://github.com/b52/opengm/
"""

import itertools
import logging
import numpy as np
import opengm
import sys
import time

from abc import ABC, abstractmethod
from enum import Enum
from heapq import heapify, heapreplace
from typing import List, Optional, Tuple, Union


log = logging.getLogger(__name__)


INF_ENERGY: float = 1.0e10
"""Energy value used to represent infinity."""


class EnergyGraph:
    """A graph representation with explicit energies.

    This is a heavily index-based representation of potentials that are going
    to be summed up. It's sort of an intermediate representation to perform
    inference and energy computations more easily, by lacking the higher
    level abstractions.

    Args:
        n_labels: Each element specifies the number of labels for each
            variable, thus the length defines the number of variables.
        potentials: An iterable that contains all `EnergyPotential`s involved.
    """

    def __init__(self, n_labels: np.ndarray,
                 potentials: List['EnergyPotential']) -> None:
        n_labels = np.asarray(n_labels, dtype=np.int32)
        assert (n_labels > 0).all()

        for potential in potentials:
            assert isinstance(potential, EnergyPotential)
            assert all(n_labels[var] == potential.values.shape[axis]
                       for axis, var in enumerate(potential.variables))

        self.n_labels = n_labels
        """Number of labels for each random variable. The i-th entry in the
        vector contains the number of labels for the i-th random variable.
        """

        self.potentials = potentials
        """List of `EnergyPotential`s that make up the energy sum."""

    @property
    def n_variables(self):
        """Number of variables."""
        return len(self.n_labels)

    @property
    def is_forest(self):
        """Tests whether the graph forms a forest."""
        return GraphStructure.from_energy_graph(self).is_forest

    def evaluate(self, states: np.ndarray) -> Union[float, np.ndarray]:
        """Computes the energy sum for one or more states.

        Args:
            states: One or more state configurations to evaluate.

        Returns:
            One or multiple energy sums.
        """
        states = np.asarray(states, dtype=np.int32)

        single_state = states.ndim == 1
        if single_state:
            states = np.array([states])

        energy_sum = np.zeros(len(states))

        for potential in self.potentials:
            energy_sum += potential.values[tuple(states[:, j]
                                                 for j in potential.variables)]

        if single_state:
            energy_sum = energy_sum[0]

        return energy_sum

    def potential_values(self, states: np.ndarray) -> np.ndarray:
        """Gets the individual potential values for some states.

        Composes a matrix of size `|states| x |potentials|` by aggregating the
        corresponding potential values.

        Args:
            states: One or multiple state configurations.

        Returns:
            Either a matrix of potential values if `|states| > 1` or else a
            row-vector of potential values.
        """
        states = np.asarray(states, dtype=np.int32)

        single_state = states.ndim == 1

        if single_state:
            states = np.array([states])

        energies = np.empty((len(states), len(self.potentials)),
                            dtype=np.float64)

        for i, potential in enumerate(self.potentials):
            energies[:, i] = potential.values[tuple(states[:, j] for j
                                                    in potential.variables)]

        if single_state:
            energies = energies[0]

        return energies

    @property
    def is_normalized(self) -> bool:
        """Determines whether potentials with identical scope are unified and
        if the corresponding variables are ordered.
        """
        existing = set()

        for potential in self.potentials:
            if (sorted(potential.variables) != potential.variables).any():
                return False

            variables = tuple(potential.variables)

            if variables in existing:
                return False

            existing.add(variables)

        return True

    def normalized(self, replace_inf: Optional[float] = None) -> 'EnergyGraph':
        """Creates a new `EnergyGraph` by merging potentials with identical
        scope.

        The returned potentials have their variables sorted to merge
        matching potentials.

        Args:
            replace_inf: If not `None`, an energy that is used to replace `inf`
                values.

        Returns:
            A new `EnergyGraph` that might have less `EnergyPotential`s
            as the original graph.
        """
        unified_pots = {}

        for potential in self.potentials:
            # re-order potential variables for easy unification
            order = np.argsort(potential.variables)
            variables = potential.variables[order]
            values = np.transpose(potential.values, order)

            if replace_inf is not None:
                inf_values = (values == np.inf)

                if inf_values.any():
                    larger_than_inf = (~inf_values) * (values >= replace_inf)

                    if larger_than_inf.any():
                        log.error('While replacing inf with %f, we noticed '
                                  'values larger than the replacement: %s',
                                  replace_inf, values[larger_than_inf])

                    if np.may_share_memory(values, potential.values):
                        values = np.copy(values)

                    values[inf_values] = replace_inf

            # we ignore all zero values
            if not np.any(values):
                continue

            idx = tuple(variables)
            existing_values = unified_pots.get(idx)

            if existing_values is not None:
                existing_values += values
            else:
                if np.may_share_memory(values, potential.values):
                    values = np.copy(values)

                unified_pots[idx] = values

        unified_pots = sorted(unified_pots.items(), key=lambda x: x[0])

        return EnergyGraph(self.n_labels, [EnergyPotential(*p)
                                           for p in unified_pots])

    def __repr__(self):
        return 'EnergyGraph(n_labels={!r}, potentials={}'.format(
            self.n_labels.tolist(), ', '.join(map(repr, self.potentials)))


class EnergyPotential:
    """Represents a term in an energy sum.

    The term has a finite state space and pre-computed energy values.

    Args:
        variables: The indices of the variables this term depends on.
        values: The energy values for the configurations of the given
            `variables`. The state of the first variable is encoded in the
            first axis and so forth.
    """

    def __init__(self, variables, values) -> None:
        variables = np.asarray(variables, dtype=np.int32)
        values = np.asarray(values, dtype=np.float64)

        if variables.ndim == 0:
            variables.shape = (1,)

        assert variables.size == values.ndim
        assert (variables >= 0).all()

        self.variables = variables
        """Indices of the dependent variables. The order is in accordance
        with the order of the `values` axes.
        """

        self.values = values
        """Array with dimensionality `|variables|` which contains the energy
        values for the different states.
        """

    def __repr__(self) -> str:
        return (f'EnergyPotential(variables={self.variables!r}, '
                f'values=[∀x {self.values.min()}≤x≤{self.values.max()}])')


class GraphStructure:
    """Provides information about the graph's structure.

    Args:
        adjacency: The graph structure is given as an adjacency matrix.
    """

    def __init__(self, adjacency: np.ndarray) -> None:
        assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]

        self.nodes = np.arange(adjacency.shape[0])
        """All node indices, consecutive and increasing, staring from 0."""

        self.adjacency = adjacency
        """Adjacency matrix of the nodes."""

    @staticmethod
    def from_energy_graph(graph: EnergyGraph):
        """Creates a `GraphStructure` from the given `EnergyGraph`."""
        adjacency = np.zeros((graph.n_variables, graph.n_variables),
                             dtype=bool)

        for term in graph.potentials:
            assert len(term.variables) < 3

            for a, b in itertools.product(term.variables, term.variables):
                if a != b:
                    adjacency[a, b] = adjacency[b, a] = True

        return GraphStructure(adjacency)

    @property
    def is_forest(self):
        """Whether the graph is a tree or a forest."""

        nodes = set(self.nodes)
        visited = set()

        def is_circular(node, parent):
            nodes.remove(node)
            visited.add(node)

            for connected in np.where(self.adjacency[node])[0]:
                if connected not in visited:
                    if is_circular(connected, node):
                        return True
                elif connected != parent:
                    return True

            return False

        while len(nodes) > 0:
            if is_circular(next(iter(nodes)), -1):
                return False

        return True


class VariableProposal(str, Enum):
    """Possible ways to draw a variable to change during Gibbs sampling.

    See:
        - `GibbsSolver`
    """

    RANDOM = 'random'
    """Draws variables using a uniform distribution."""

    CYCLIC = 'cyclic'
    """Iterates variables in order, starting from the beginning when reaching
    the end."""


def gibbs_chain(graph: EnergyGraph, start: Optional[np.ndarray] = None,
                variable_proposal: VariableProposal = VariableProposal.RANDOM,
                rng: Optional[np.random.RandomState] = None,
                inf_energy: float = INF_ENERGY):
    """Creates a Gibbs chain.

    The chain is represented by a generator yielding the state and the energy.

    **Beware,** you should drop a certain amount of initial states, prior to
    actually using the yieldes states (so called burn in).

    Args:
        graph: The graph we want to create the chain for.
        start: An optional start state. If none is given, we start from all
            labels set to 0.
        variable_proposal: Specifies the strategy how to chose a new variable
            to change the label for in each iteration.
        rng: An optional PRNG to use. If none is given, a new one is created.
        inf_energy: Value used to represent inf energies.

    Return:
        A generator that yields pairs of state and respective energy.
    """
    assert start is None or start.shape == graph.n_labels.shape

    graph = graph.normalized()

    for term in graph.potentials:
        energy = term.values

        inf_values = (energy == np.inf)
        if inf_values.any():
            valid_energy = energy[np.logical_not(inf_values)]
            assert (valid_energy < inf_energy).all(), \
                'the provided inf_energy value is to low'
            energy[inf_values] = inf_energy

        assert np.isfinite(energy).all()

    if rng is None:
        rng = np.random.RandomState()
    n_vars = len(graph.n_labels)

    # current state
    state = np.zeros(n_vars, np.int32) if start is None else \
        np.asarray(start, np.int32)
    energies = np.array([t.values[tuple(state[t.variables])]
                         for t in graph.potentials])
    energy = energies.sum()

    yield state, energy

    # maps variable indices to factors
    potentials = [[] for _ in range(n_vars)]
    for i, t in enumerate(graph.potentials):
        for v in t.variables:
            potentials[v].append((i, t))

    variable = 0

    while True:
        # chose a variable
        if variable_proposal == VariableProposal.RANDOM:
            variable = rng.randint(n_vars)

        new_state = np.copy(state)

        # build transition probs
        transition_probs = np.empty(graph.n_labels[variable])
        for label in range(len(transition_probs)):
            new_state[variable] = label
            transition_probs[label] = sum(
                t.values[tuple(new_state[t.variables])]
                for i, t in potentials[variable])

        # exploit the fact that softmax(x) = softmax(x - C) to prevent
        # overflow
        transition_probs = -(transition_probs - transition_probs.min())
        transition_probs = np.exp(transition_probs)

        normalization = transition_probs.sum()
        if normalization == 0:
            transition_probs = 1 / len(transition_probs)
        else:
            transition_probs /= normalization

        # choose a new label based on the transition probs
        label = rng.choice(len(transition_probs), p=transition_probs)
        new_state[variable] = label

        # compute the new energies
        new_energies = np.copy(energies)
        for i, potential in potentials[variable]:
            new_energies[i] = potential.values[
                tuple(new_state[potential.variables])]
        new_energy = new_energies.sum()

        yield new_state, new_energy

        # acceptance criterion
        with np.errstate(over='ignore'):
            acceptance_ratio = np.exp(np.nan_to_num(energy - new_energy))

        if rng.rand() < acceptance_ratio:
            state = new_state
            energies = new_energies
            energy = new_energy

        if variable_proposal == VariableProposal.CYCLIC:
            variable = (variable + 1) % n_vars


class Solver(ABC):
    """Provides a strategy to minimize the energy summation of a given graph.
    """

    @abstractmethod
    def infer(self, graph: EnergyGraph, *args, **kwargs) \
            -> Tuple[np.ndarray, float]:
        """Infers the most likely configuration of the given `graph`.

        Args:
            graph: The graph we want to solve.

        Returns:
            A tuple containing (0) the resulting selection and (1) the
            corresponding energy.
        """
        raise NotImplementedError


class NBestSolver(Solver):
    """A solver which supports returing the `n_best` states rather than just
    the single best.
    """

    @abstractmethod
    def infer(self, graph: EnergyGraph, *args, n_best: int = 1, **kwargs) \
            -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Infers the `n_best` likely configurations of the given `graph`.

        Args:
            graph: The graph we want to solve.
            n_best: The maximal amount of best states to return.

        Returns:
            A tuple containing (0) the resulting selection and (1) the
            corresponding energy.
        """
        raise NotImplementedError


class GibbsSolver(NBestSolver):
    """Uses Gibbs sampling to approximately solve the given graph.

    Args:
        sampling_steps: Number of sampling steps.
        max_time: Optional maximal amount of time to run the inference for.
        variable_proposal: Specifies how a variable to change is chosen.
        inf_energy: Value that is used instead of `inf` values.
    """

    def __init__(self, sampling_steps: Optional[int] = 1_000,
                 max_time: Optional[float] = None,
                 variable_proposal: VariableProposal =
                 VariableProposal.RANDOM,
                 inf_energy: float = INF_ENERGY):
        assert sampling_steps is not None or max_time is not None, \
            'we need at least one stop criterion'

        self.sampling_steps = sampling_steps or sys.maxsize
        """Number of sampling steps."""

        self.max_time = max_time or np.inf
        """Optional amount"""

        self.variable_proposal = variable_proposal
        """How a new variable is determined."""

        self.inf_energy = inf_energy
        """Energy used to represent inf values."""

    def infer(self, graph: EnergyGraph, start: Optional[np.ndarray] = None,
              n_best: int = 1, rng: Optional[np.random.RandomState] = None) \
            -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Performs inference using Gibbs sampling.

        Has the advantage that it is easy to generate approximate n-best
        states matching the posterior easily and quickly.

        Args:
            graph: The graph to optimize.
            start: An optional start configuration. This can make a huge
                difference.
            n_best: Number of n-best configurations to return.
            rng: An optional PRNG to use. If none, a new one is used.

        Return:
            A tuple containing (0) the resulting selection and (1) the
            corresponding energy.
        """
        assert start is None or start.shape == graph.n_labels.shape
        assert n_best > 0

        # maintain our n-best list
        best_states = set()
        best_configs = [(-np.inf, tuple())] * n_best
        heapify(best_configs)

        chain = gibbs_chain(graph, start=start,
                            variable_proposal=self.variable_proposal,
                            rng=rng, inf_energy=self.inf_energy)

        start = time.time()

        for step, (state, energy) in enumerate(chain):
            state2 = tuple(state)

            # maintain our n-best list
            if energy < -best_configs[0][0] and state2 not in best_states:
                _, old_state = heapreplace(best_configs,
                                           (-energy, state2))
                best_states.add(state2)
                best_states.discard(old_state)

            if step == self.sampling_steps:
                break

            # do the time check not that often...
            if self.max_time is not None and step % 10 == 0:
                delta = time.time() - start
                if delta >= self.max_time:
                    break

        best_configs = sorted((-e, s) for e, s in best_configs if s)
        states = np.array([s for _, s in best_configs], dtype=np.int32)
        energies = np.array([e for e, _ in best_configs], dtype=np.float32)

        if n_best == 1:
            states, energies = states[0], energies[0]

        return states, energies


class _OpenGMSolver(Solver):
    """Solver based on inference methods implemented in **OpenGM**.

    Args:
        inf_energy: Use this real value instead of `inf`, because OpenGM
            can not handle `inf`.
    """

    SUPPORT_N_BEST: bool = False
    """Whether the solver supports the retrieval of multiple best states."""

    def __init__(self, inf_energy: float = INF_ENERGY) -> None:
        assert inf_energy > 0

        super().__init__()

        self.inf_energy = inf_energy
        """The energy used in place of `inf` values when passed to OpenGM."""

    def infer(self, graph: EnergyGraph, n_best: int = 1) -> \
            Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Infers the most likely configuration of the given `graph`.

        Args:
            graph: The graph we want to solve.
            n_best: Return the n_best states instead of just the first best.
                Note, this is not supported by all algorithms.

        Returns:
            A tuple containing (0) the resulting selection and (1) the
            corresponding energy.
        """
        if n_best > 1 and not self.SUPPORT_N_BEST:
            raise RuntimeError('%s does not support multiple best states' %
                               self.__class__.__name__)

        # we first make sure that the variables are ordered and that potentials
        # of identical scope are merged, because opengm can not handle that
        # itself
        if not graph.is_normalized:
            graph = graph.normalized()

        gm = opengm.gm(np.array(graph.n_labels, dtype=opengm.label_type))

        for term in graph.potentials:
            energy = term.values

            inf_values = (energy == np.inf)
            if inf_values.any():
                valid_energy = energy[np.logical_not(inf_values)]
                assert (valid_energy < self.inf_energy).all(), 'the provided' \
                    'inf_energy value is to low'
                energy = np.copy(energy)
                energy[inf_values] = self.inf_energy

            assert np.isfinite(energy).all()

            gm.addFactor(gm.addFunction(energy), term.variables)

        solver = self._solver(gm, n_best) if self.SUPPORT_N_BEST else \
            self._solver(gm)
        solver.infer()

        best_state = solver.arg().astype(int)
        best_value = solver.value()
        assert np.isfinite(best_value)

        if n_best == 1:
            return best_state, best_value

        labels = np.empty((n_best, gm.numberOfVariables), dtype=int)
        values = np.empty(n_best, dtype=np.float64)

        n_best_idx = 0
        n_best_states = set()
        needs_sort = False

        for i in range(n_best):
            output = opengm.opengmcore.LabelVector()
            output.resize(gm.numberOfVariables)

            # that is an undocumented method which allows to extract the n_best
            # state, although this is not supported by all inference methods
            # and I dont know what happens when you call it on such a solver
            solver.inference._arg(output, i + 1)

            state = tuple(np.array(output).tolist())

            # no duplicated states
            if state in n_best_states:
                continue
            else:
                n_best_states.add(state)

            labels[n_best_idx] = state
            values[n_best_idx] = gm.evaluate(labels[n_best_idx])

            if n_best_idx > 0 and values[n_best_idx - 1] > values[n_best_idx]:
                log.error('The %i-best state (returned by OpenGM) has a lower '
                          'energy (%f) than the %i-best state (%f)',
                          n_best_idx, values[n_best_idx], n_best_idx - 1,
                          values[n_best_idx - 1])
                needs_sort = True

            n_best_idx += 1

        labels = labels[:n_best_idx]
        values = values[:n_best_idx]

        assert (labels[0] == best_state).all()
        assert values[0] == best_value
        assert np.isfinite(values).all()

        if needs_sort:
            idxs_sorted = sorted(np.arange(len(labels)),
                                 key=lambda i: (values[i],
                                                tuple(labels[i].tolist())))
            labels = labels[idxs_sorted]
            values = values[idxs_sorted]

        return labels, values

    @abstractmethod
    def _solver(self, gm, *args, **kwargs):
        """Returns the solver that is used to perform the inference, given
        the graphical model `gm`.
        """
        raise NotImplementedError


class BruteForceSolver(_OpenGMSolver):
    """Exhaustive brute force infernce."""

    def _solver(self, gm):
        return opengm.inference.Bruteforce(gm=gm)


class ExactTreeSolver(_OpenGMSolver):
    """Uses belief propagation to find the exact solution in polynomial time.
    """

    def infer(self, graph: EnergyGraph) -> \
            Tuple[np.ndarray, Union[float, np.ndarray]]:
        assert GraphStructure.from_energy_graph(graph).is_forest
        return super().infer(graph)

    def _solver(self, gm):
        parameter = opengm.InfParam(isAcyclic=True)
        return opengm.inference.BeliefPropagation(gm=gm, parameter=parameter)


class AStarSolver(_OpenGMSolver):
    """A* solver by Bergtholdt et al.

    Uses an admissable search heuristic in combination with a priority queue
    to find the n-best states.

    Args:
        bound: The upper bound of the objective. A good bound will speed up
            inference.
        max_heap_size: Maximal number of states in the queue.
    """

    SUPPORT_N_BEST = True
    """A* supports n-best inference."""

    def __init__(self, bound: float = np.inf, max_heap_size: int = 25_000_000,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.bound = bound
        """The upper bound of the objective function."""

        self.max_heap_size = max_heap_size
        """Maximum size of the heap which is used while inference."""

    def infer(self, graph: EnergyGraph, n_best: int = 1) -> \
            Tuple[np.ndarray, Union[float, np.ndarray]]:
        states, values = super().infer(graph=graph, n_best=n_best)

        if states.ndim == 1:
            if values > self.bound:
                return None
        else:
            invalid_mask = values > self.bound
            invalid = invalid_mask.sum()

            if invalid == values.size:
                return None
            elif invalid > 0:
                valid_mask = ~invalid_mask
                states = states[valid_mask]
                values = values[valid_mask]

        return states, values

    def _solver(self, gm, n_best):
        parameter = opengm.InfParam(numberOfOpt=n_best,
                                    maxHeapSize=self.max_heap_size,
                                    # that spelling error is correct
                                    obectiveBound=self.bound)
        return opengm.inference.AStar(gm=gm, parameter=parameter)


class LazyFlipperSolver(_OpenGMSolver):
    """The LazyFlipper move-making strategy by Andres et al.

    Args:
        max_subgraph_size: Min subgraph size to be optimal.
    """

    def __init__(self, max_subgraph_size: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)

        assert max_subgraph_size > 0

        self.max_subgraph_size = max_subgraph_size
        """The solver garuanties the optimility of a subgraph of this size."""

    def _solver(self, gm):
        parameter = opengm.InfParam(maxSubgraphSize=self.max_subgraph_size)
        return opengm.inference.LazyFlipper(gm=gm, parameter=parameter)


def solve_astar(graph, n_best, max_heap_size: int = 25_000_000):
    """Helper function for multi-threaded applications."""
    try:
        solver = AStarSolver(max_heap_size=max_heap_size)
        return solver.infer(graph, n_best)[0]
    except KeyboardInterrupt:
        return


def solve_gibbs(graph, n_best, start, max_time):
    """Helper function for multi-threaded applications."""
    try:
        solver = GibbsSolver(sampling_steps=None, max_time=max_time)
        return solver.infer(graph, n_best=n_best, start=start)
    except KeyboardInterrupt:
        return

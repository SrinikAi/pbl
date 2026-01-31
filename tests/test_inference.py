import numpy as np

from pbl.inference import BruteForceSolver, AStarSolver, LazyFlipperSolver
from pbl.inference import ExactTreeSolver, GibbsSolver
from pbl.inference import EnergyGraph, EnergyPotential


class TestEnergyGraph:
    def test_potential_values(self):
        graph = EnergyGraph([2, 2], [EnergyPotential([0], [1, 2]),
                                     EnergyPotential([1], [3, 4]),
                                     EnergyPotential([1, 0], [[5, 6],
                                                              [7, 8]])])

        assert graph.potential_values([0, 0]).tolist() == [1, 3, 5]
        assert graph.potential_values([[1, 0],
                                       [0, 1]]).tolist() == [[2, 3, 6],
                                                             [1, 4, 7]]

    def test_evaluate(self):
        graph = EnergyGraph([2, 2], [EnergyPotential([0], [1, 2]),
                                     EnergyPotential([1], [3, 4]),
                                     EnergyPotential([1, 0], [[5, 6],
                                                              [7, 8]])])

        assert graph.evaluate([0, 0]) == 9
        assert graph.evaluate([[1, 0],
                               [0, 1]]).tolist() == [11, 12]

    def test_normalized(self):
        graph = EnergyGraph([2, 2, 2], [EnergyPotential([0], [1, 2]),
                                        EnergyPotential([1], [4, 4]),
                                        EnergyPotential([0], [4, 1]),
                                        EnergyPotential([2], [0, 0]),
                                        EnergyPotential([1, 0], [[5, 6],
                                                                 [7, 8]]),
                                        EnergyPotential([0, 1], [[5, 3],
                                                                 [4, 2]])])

        normalized = graph.normalized()

        assert len(normalized.potentials) == 3, repr(normalized)
        assert normalized.potentials[0].variables.tolist() == [0]
        assert normalized.potentials[0].values.tolist() == [5, 3]
        assert normalized.potentials[1].variables.tolist() == [0, 1]
        assert normalized.potentials[1].values.tolist() == [[10, 10],
                                                            [10, 10]]
        assert normalized.potentials[2].variables.tolist() == [1]
        assert normalized.potentials[2].values.tolist() == [4, 4]


def test_gibbs_solver():
    graph = _random_graph(3, 3)

    results_bf = BruteForceSolver().infer(graph)
    results_gibbs = GibbsSolver().infer(graph)

    assert (results_bf[0] == results_gibbs[0]).all()
    assert np.isclose(results_bf[1], results_gibbs[1])

    result2 = GibbsSolver().infer(graph, n_best=10)

    assert all(a <= b for a, b in zip(result2[1], result2[1][1:]))


def test_astar_solver():
    graph = _random_graph(5, 5)

    results_bf = BruteForceSolver().infer(graph)
    results_astar = AStarSolver().infer(graph)

    assert (results_bf[0] == results_astar[0]).all()
    assert np.isclose(results_bf[1], results_astar[1])


def test_lazyflipper_solver():
    graph = _random_graph(5, 5)

    results_bf = BruteForceSolver().infer(graph)
    results_lf = LazyFlipperSolver().infer(graph)

    assert (results_bf[0] == results_lf[0]).all()
    assert np.isclose(results_bf[1], results_lf[1])


def test_exact_tree_solver():
    graph = _random_tree(5, 5)

    results_bf = BruteForceSolver().infer(graph)
    results_lf = ExactTreeSolver().infer(graph)

    assert (results_bf[0] == results_lf[0]).all()
    assert np.isclose(results_bf[1], results_lf[1])


def test_exact_tree_solver_disconnected():
    graph = _random_tree(10, 5)

    drop_term = next(term for term in graph.potentials
                     if np.all(term.variables == [3, 4]))
    graph.potentials.remove(drop_term)

    results_bf = BruteForceSolver().infer(graph)
    results_lf = ExactTreeSolver().infer(graph)

    assert (results_bf[0] == results_lf[0]).all()
    assert np.isclose(results_bf[1], results_lf[1])


def _random_graph(nodes: int, states: int) -> EnergyGraph:
    """Creates a fully connected (binaries only) graph with random energy
    values.
    """
    terms = []
    rng = np.random.RandomState(31337)

    for i in range(nodes):
        terms.append(EnergyPotential(i, rng.uniform(size=states)))

        for j in range(i + 1, nodes):
            terms.append(EnergyPotential([i, j], rng.uniform(
                size=(states, states))))

    return EnergyGraph([states] * nodes, terms)


def _random_tree(nodes: int, states: int) -> EnergyGraph:
    """Creates a tree (binaries only) graph with random energy values.
    """
    terms = []
    rng = np.random.RandomState(31337)

    for i in range(nodes):
        terms.append(EnergyPotential(i, rng.uniform(size=states)))

        if i < nodes - 1:
            terms.append(EnergyPotential([i, i + 1],
                                         rng.uniform(size=(states, states))))

    return EnergyGraph([states] * nodes, terms)

import numpy as np
import pytest

from pbl.graph import Graph, Potential
from pbl.inference import BruteForceSolver


class TestGraph:
    def test_initial_values(self):
        class P(Potential):
            @staticmethod
            def arity() -> int:
                return 2

            def compute(*args, **kwargs):
                return 0

        graph = Graph([P(['a', 'c']), P(['b', 'd']), P(['b', 'c']),
                       P(['d', 'e'])])
        assert graph.n_dims == 2
        assert graph.n_potentials == 4
        assert graph.n_parts == 5
        assert graph.parts == ['a', 'b', 'c', 'd', 'e']
        assert graph.connections == [('a', 'c'), ('b', 'c'), ('b', 'd'),
                                     ('d', 'e')]

    def test_compute_energies(self, caplog):
        class TestPotential(Potential):
            @staticmethod
            def arity() -> int:
                return 2

            def compute(self, image, positions, **kwargs):
                position_a, position_b = positions
                return np.sum(position_a + position_b, axis=1)

        graph = Graph([TestPotential(['a', 'b']),
                       TestPotential(['a', 'b'])])

        computed = graph.compute_energies(None, {'a': np.array([[0, 1],
                                                                [2, 3]]),
                                                 'b': np.array([[1, 3],
                                                                [2, 5]])})
        assert len(computed.potentials) == 2

        for term in computed.potentials:
            assert term.values.tolist() == [[5, 8],
                                            [9, 12]]

        graph = Graph([TestPotential(['a', 'b'], weight=1),
                       TestPotential(['b', 'c'], weight=0)])

        computed = graph.compute_energies(None, {'a': np.array([[0, 1],
                                                                [2, 3]]),
                                                 'b': np.array([[1, 3],
                                                                [2, 5]]),
                                                 'c': np.array([[1, 1]])},
                                          use_weights=False)
        assert len(computed.potentials) == 2

        computed = graph.compute_energies(None, {'a': np.array([[0, 1],
                                                                [2, 3]]),
                                                 'b': np.array([[1, 3],
                                                                [2, 5]]),
                                                 'c': np.array([[1, 1]])})
        assert len(computed.potentials) == 1

        assert 'Some variables are not' in caplog.record_tuples[0][2]
        assert ' 2:c' in caplog.record_tuples[0][2]

    def test_infer(self):
        class TestPotential(Potential):
            def __init__(self, *parts, values=None, **kwargs):
                super().__init__(parts, **kwargs)
                self.values = np.array(values)

            @staticmethod
            def arity() -> int:
                return 2

            def compute(self, image, positions, **kwargs):
                a, b = positions
                return self.values[a, b]

        graph = Graph([TestPotential('A', 'B', values=[[30, 5],
                                                       [1, 10]],
                                     weight=11, unknown_energy=-10),
                       TestPotential('B', 'C', values=[[100, 1],
                                                       [1, 100]],
                                     weight=0),
                       TestPotential('C', 'D', values=[[1, 100],
                                                       [100, 1]]),
                       TestPotential('D', 'A', values=[[100, 1],
                                                       [1, 100]],
                                     weight=0.5)])

        candidates = {'A': np.arange(2), 'B': np.arange(2),
                      'C': np.arange(2), 'D': np.arange(2)}

        solver = BruteForceSolver()

        result = graph.infer(None, candidates, solver, use_weights=False)
        assert result.energy == 13
        assert result.positions == {'A': 1, 'B': 1, 'C': 0, 'D': 0}

        result = graph.infer(None, candidates, solver)
        assert result.energy == 12.5
        assert result.positions == {'A': 1, 'B': 0, 'C': 0, 'D': 0}

        result = graph.infer(None, candidates, solver, support_unknown=['B'])
        assert result.energy == -108.5
        assert result.positions == {'A': 1, 'B': None, 'C': 0, 'D': 0}

        with pytest.raises(RuntimeError):
            graph.infer(None, candidates, solver, support_unknown=['B'],
                        use_weights=False)

        with pytest.raises(RuntimeError):
            graph.infer(None, candidates, solver,
                        support_unknown=['A'])

    def test_collect_candidates(self):
        class TestPotential(Potential):
            def __init__(self, *parts, cands=None, **kwargs):
                super().__init__(parts, **kwargs)
                self.cands = cands

            @staticmethod
            def arity() -> int:
                return 1

            def propose_candidates(self, image):
                return self.cands

            def compute(self, image, *args, **kwargs):
                return 0

        graph = Graph([TestPotential('a', cands={'a': np.array([[1], [2]])}),
                       TestPotential('b', cands={'a': np.array([[3], [4]]),
                                                 'b': np.array([[2], [2]])}),
                       TestPotential('c', cands=dict())])

        cands = graph.collect_candidates(None)

        assert set(cands.keys()) == {'a', 'b'}
        assert cands['a'].tolist() == [[1], [2], [3], [4]]
        assert cands['b'].tolist() == [[2], [2]]

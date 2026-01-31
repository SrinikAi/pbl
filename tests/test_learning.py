import hiwi
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Union

from hiwi import Image, LocalMaxLocator

from pbl.evaluation import MaxDistance
from pbl.graph import Graph, Potential
from pbl.potentials import UnaryPotential
from pbl.potentials import RteLocalizer, VectorPotential
from pbl.potentials import MultiPartCNN, GaussianVector
from pbl.learning import SgdMaxMarginLearning, CgMaxMarginLearning
from pbl.learning import FullSgdMaxMarginLearning
from pbl.learning import Sample, OptBatch
from pbl.utils import working_dir


def test_learning():
    class TestPot(Potential):
        @staticmethod
        def arity() -> int:
            return 1

        def compute(self, image, position, **kwargs):
            return np.ones(len(position[0]))

        def propose_candidates(self, image):
            return {self.parts[0]: np.arange(5).reshape(5, 1)}

    graph = Graph([TestPot(['eins']), TestPot(['zwei'])])

    img = hiwi.Image(path='test.jpg', objects=[hiwi.Object()])
    img.objects[0].parts['eins'] = hiwi.Object(position=[1])
    img.objects[0].parts['zwei'] = hiwi.Object(position=[2])

    learner = SgdMaxMarginLearning(criterion=MaxDistance(10, use_mm=False))
    learner.optimize(graph, [img], max_iterations=10)


# def notest_sgdmaxmarginlearning_weights():
#     learner = SgdMaxMarginLearning(margin=0.1)
#
#     opt = learner.optimizer(np.array([1., 1.]), np.array([np.nan, np.nan]))
#     opt.__next__()
#
#     rivals = Rivals(corr_energies=np.array([[4., 2.]]),
#                     incorr_energies=np.array([[2., 3.]]),
#                     error_reduction=np.array([0., 0.]))
#
#     prev_loss = np.inf
#     prev_margin = -np.inf
#     prev_weights = np.array([1., 1.])
#
#     for iter in range(1_000_000):
#         loss, weights, unknown_energies = opt.send(rivals)
#         opt.__next__()
#
#         margin = (weights * rivals.incorr_energies).sum() \
#             - (weights * rivals.corr_energies).sum()
#
#         assert loss < prev_loss
#         assert margin > prev_margin
#         assert weights[0] < weights[1]
#         assert weights[0] < prev_weights[0]
#         assert weights[1] > prev_weights[1]
#
#         if loss == 0:
#             break
#
#         prev_loss = loss
#         prev_weights = weights
#
#     assert iter < 1_000
#     assert loss == 0
#     assert margin >= .1
#
#
# def notest_sgdmaxmarginlearning_unknown_energies():
#     learner = SgdMaxMarginLearning(margin=0.1)
#
#     opt = learner.optimizer(np.array([1., 1.]), np.array([np.nan, 3.5]))
#     opt.__next__()
#
#     rivals = Rivals(corr_energies=np.array([[4., np.nan]]),
#                     incorr_energies=np.array([[2., 3.]]),
#                     error_reduction=np.array([0., 0.]))
#
#     prev_loss = np.inf
#     prev_margin = -np.inf
#     prev_weights = np.array([1., 1.])
#     prev_unknown_energies = np.array([np.nan, 3.5])
#
#     for iter in range(1_000_000):
#         loss, weights, unknown_energies = opt.send(rivals)
#         opt.__next__()
#
#         margin = (weights * np.where(np.isnan(rivals.incorr_energies),
#                                      unknown_energies,
#                                      rivals.incorr_energies)).sum() \
#             - (weights * np.where(np.isnan(rivals.corr_energies),
#                                   unknown_energies,
#                                   rivals.corr_energies)).sum()
#
#         assert loss < prev_loss
#         assert margin > prev_margin
#         assert weights[0] < weights[1]
#         assert weights[0] < prev_weights[0]
#         assert unknown_energies[1] < prev_unknown_energies[1]
#
#         if loss == 0:
#             break
#
#         prev_loss = loss
#         prev_margin = margin
#         prev_weights = weights
#         prev_unknown_energies = unknown_energies
#
#     assert iter < 1_000
#     assert loss == 0
#     assert unknown_energies[1] < 3
#     assert margin >= .1


def test_sgdmaxmarginlearning_real(tmpdir):
    images = hiwi.ImageList.load(Path(__file__).parent / 'data' / '2d'
                                 / 'images.iml')
    images = images[:4]

    parts = ['mouth_left', 'mouth_right']

    potentials = []
    for part in parts:
        potentials.append(RteLocalizer(part, n_dims=2,
                                       rfl_kw=dict(n_trees=1,
                                                   n_dims=2,
                                                   n_channels=3,
                                                   n_features=32,
                                                   max_depth=None),
                                       peak_finder=LocalMaxLocator()))

    for parts in _fully_connected(parts):
        potentials.append(VectorPotential(parts))

    graph = Graph(potentials)

    for pot in graph.potentials:
        pot.train(images)

    optimizer = SgdMaxMarginLearning(criterion=MaxDistance(10))
    optimizer.optimize(graph, images, max_iterations=3)


def test_fullsgdmaxmarginlearning_real(tmpdir):
    images = hiwi.ImageList.load(Path(__file__).parent / 'data' / '2d'
                                 / 'images.iml')
    images = images[:4]

    parts = ['mouth_left', 'mouth_right']

    session = tf.Session(graph=tf.Graph())

    cnn = MultiPartCNN(session=session, parts=parts, n_dims=2, n_channels=3,
                       input_shape=np.max([i.data.shape[:2] for i in images],
                                          axis=0),
                       peak_finder=LocalMaxLocator())

    potentials = []
    potentials.extend(cnn.potentials)

    for pot_parts in _fully_connected(parts):
        potentials.append(GaussianVector(session, pot_parts, plot=True))

    graph = Graph(potentials)

    with working_dir.set(tmpdir):
        optimizer = FullSgdMaxMarginLearning(criterion=MaxDistance(10),
                                             session=session)
        optimizer.optimize(graph, images, max_iterations=3)


def test_cgmaxmarginlearning_real(tmpdir):
    images = hiwi.ImageList.load(Path(__file__).parent / 'data' / '2d'
                                 / 'images.iml')
    images = images[:4]

    parts = ['mouth_left', 'mouth_right']

    potentials = []
    for part in parts:
        potentials.append(RteLocalizer(part, n_dims=2,
                                       rfl_kw=dict(n_trees=1,
                                                   n_dims=2,
                                                   n_channels=3,
                                                   n_features=32,
                                                   max_depth=None),
                                       peak_finder=LocalMaxLocator()))

    for pot_parts in _fully_connected(parts):
        potentials.append(VectorPotential(pot_parts))

    graph = Graph(potentials)

    with working_dir.set(tmpdir):
        for pot in graph.potentials:
            pot.train(images)

        optimizer = CgMaxMarginLearning(criterion=MaxDistance(10))
        optimizer.optimize(graph, images, max_iterations=3)


def _fully_connected(parts):
    for i in range(len(parts) - 1):
        for j in range(i + 1, len(parts)):
            yield parts[i], parts[j]


def test_cgmaxmarginlearning_fictional_old(tmpdir):
    images = hiwi.ImageList.load(Path(__file__).parent / 'data' / '2d'
                                 / 'images.iml')
    images = images[:4]

    parts = ['mouth_left', 'mouth_right']

    potentials = []
    for part in parts:
        potentials.append(RteLocalizer(part, n_dims=2,
                                       rfl_kw=dict(n_trees=1,
                                                   n_dims=2,
                                                   n_channels=3,
                                                   n_features=32,
                                                   max_depth=None),
                                       peak_finder=LocalMaxLocator()))

    for pot_parts in _fully_connected(parts):
        potentials.append(VectorPotential(pot_parts))

    graph = Graph(potentials)

    with working_dir.set(tmpdir):
        for pot in graph.potentials:
            pot.train(images)

        optimizer = CgMaxMarginLearning(criterion=MaxDistance(10))
        optimizer.optimize(graph, images, max_iterations=3)


def test_cgmaxmarginlearning_fictional():
    # log.root.setLevel(log.DEBUG)
    # log.basicConfig(stream=sys.stderr, level=log.DEBUG,
    #                 format='%(asctime)s %(message)s')
    import coloredlogs
    import random
    import time
    import logging as log
    import math

    coloredlogs.install(level='DEBUG', fmt='%(asctime)s %(message)s')
    random.seed(7645)
    np.random.seed(7645)

    rng = np.random.RandomState(7645)

    number_of_inputs = 10
    input_batch_size = 9
    # number of potentials
    input_length = 16

    number_of_np_inftys = math.floor(input_length/3)
    input = []
    input2 = []
    positions_of_np_inftys = []

    class GraphMock(Graph):
        def __init__(self, potentials: List[Potential], n_dims: int = 2,
                     support_unknown: set = frozenset()):
            super(GraphMock, self).__init__(potentials, n_dims,
                                            support_unknown)
            self.debug_slack = 0

    class CgMaxMarginLearningMock(CgMaxMarginLearning):
        def infer_rival_energies(self, graph: Graph, samples: List[Sample],
                                 use_unknown_energies: bool = False) \
                -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return samples

    for i in range(number_of_np_inftys):
        positions_of_np_inftys.append((rng.randint(0, input_batch_size - 1),
                                      rng.randint(0, input_length - 1)))

    # random.seed(7645)
    # np.random.seed(7645)

    criterion = MaxDistance(10)

    cg = CgMaxMarginLearningMock(criterion=criterion)

    class DummyPotential(UnaryPotential):
        def compute(self, image: Image, positions: List[np.ndarray],
                    **kwargs) -> Union[float, np.ndarray]:
            return 0

    graph = GraphMock([DummyPotential(str(i)) for i in range(input_length)])

    opt = cg.optimizer(graph)
    cg.debug_run = True
    opt.__next__()

    assert (cg.append_all_constraints is True and
            cg.delete_constraints is False),\
        "Test doesn't make sense with these configs"

    t1 = time.clock()
    for i in range(number_of_inputs):
        input.append(1500 * (np.random.random((input_batch_size,
                                               input_length)) - 0.45))

        if i == 0:
            for k in range(number_of_np_inftys):
                input[-1][positions_of_np_inftys[k]] = (k % 2 * 2 - 1) *\
                                                       np.infty

        input2.append(500 * (np.random.rand(input_batch_size)))

        opt.send(OptBatch(samples=(np.zeros((input_batch_size, input_length)),
                                   input[i], input2[i]),
                          prev_result=None,
                          best_result=None,
                          best_params=None))
        opt.__next__()

    log.critical(time.clock()-t1)

    weights = graph.weights
    slack = graph.debug_slack

    d = []

    assert cg.append_all_constraints is True
    assert slack.size == len(input) * input_batch_size, \
        "There is a problem extracting the slack values from the graph"
    assert cg.infinity_replacement < 1e40

    for i in range(number_of_inputs):
        # Sets np-infties to 1e50. This is done so infty*0 doesn't get
        # calculated to nan
        input[i][np.isinf(np.where(input[i] < 0, 0, input[i]))] = \
            1e50
        input[i][np.isinf(np.where(input[i] > 0, 0, input[i]))] = \
            - 1e50
        for k in range(input_batch_size):
            if cg.dynamic_margin:
                d.append(input[i][k] @ weights +
                         slack[input_batch_size * i + k]
                         - cg.error_reduction_function(input2[i][k]))

                assert d[-1] >= 0, \
                    "Following constraint was violated %d, %d by %s" %\
                    (i, k, input[i][k] @ weights + slack[
                        input_batch_size * i + k] -
                     cg.error_reduction_function(input2[i][k]))
            else:
                d.append(input[i][k] @ weights + slack[input_batch_size *
                                                       i + k] - cg.margin)

                assert d[-1] >= 0,\
                    "Following constraint was violated %d, %d by %s" %\
                    (i, k, input[i][k] @ weights + slack[
                        input_batch_size * i + k] - cg.margin)

        print("min/max values are")
        print(min(d))
        print(max(d))
        # asserts slack variables are not all 0
        assert min(d) < 1e-5

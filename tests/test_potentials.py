import io
import numpy as np
import pickle
import tensorflow as tf

from hiwi import ImageList, Image, Object, LocalMaxLocator
from scipy import stats

from pbl.potentials import BasePotentialMixin, MultiPartCNN
from pbl.potentials import VectorPotential


class TestPotential:
    class DummyPotential(BasePotentialMixin):
        @staticmethod
        def arity() -> int:
            return 1

        def compute(self, *args):
            return None

    def test_init(self):
        pot = TestPotential.DummyPotential(['one'])
        assert pot.parts == ['one']

    def test_positions(self):
        o1 = Object(position=[2, 3])
        o2 = Object(position=[4, 5])
        images = ImageList(Image(objects=[Object(parts={'asd': o1})],
                                 spacing=[2, 1.5]),
                           Image(objects=[Object()]),
                           Image(objects=[Object(parts={'asd': o2})],
                                 spacing=[0.5, 3]))

        pot = TestPotential.DummyPotential(['asd'])
        assert [p.tolist() for p in pot.positions(images)] == [[3, 2], [5, 4]]
        assert [p.tolist() for p in pot.positions(images, use_mm=True)] == \
               [[4.5, 4], [15, 2]]


class TestVectorPotential:
    def test_compute(self):
        pot = VectorPotential(['foo', 'bar'])
        pot.dist = stats.multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 1]])

        energies = pot.compute(None, [np.array([0, 1]), np.array([1, 0])])
        assert energies.ndim == 0

        energies = pot.compute(None, [np.array([[0, 1]]), np.array([[1, 0]])])
        assert energies.ndim == 1


class TestMultiPartCNN:
    def test_pickle(self):
        file = io.BytesIO()

        with tf.Session(graph=tf.Graph()) as session:
            cnn = MultiPartCNN(session, ['foo', 'bar'], n_dims=2, n_channels=1,
                               input_shape=np.array([50, 50]),
                               peak_finder=LocalMaxLocator())
            weights = cnn.get_weights()

            pickle.dump(cnn, file)  # nosec
            file.seek(0)

        with tf.Session(graph=tf.Graph()) as session:
            cnn2 = pickle.load(file)  # nosec

            assert cnn2.session is session
            assert cnn2.input_tf is not None
            assert cnn2.output_tf is not None
            assert cnn2.trainables_tf is not None

            assert all(np.array_equal(weights[var.name], session.run(var))
                       for var in cnn2.trainables_tf)

import hiwi
import numpy as np

from pbl.evaluation import Criterion, MaxDistance


def test_criterion_true_pos():
    image = hiwi.Image(objects=[hiwi.Object()])
    image.objects[0].parts['foo'] = hiwi.Object(position=[1, 2])

    true_pos = Criterion.true_pos(image, ['foo', 'bar'])
    assert true_pos.shape == (2, 2)
    assert true_pos[0].tolist() == [2, 1]
    assert np.isnan(true_pos[1]).all()


class TestMaxDistance:
    def test_evaluate(self):
        image = hiwi.Image(objects=[hiwi.Object()], spacing=[2, 2])
        image.objects[0].parts['foo'] = hiwi.Object(position=[1, 2])
        image.objects[0].parts['bar'] = hiwi.Object(position=[5, 3])
        image.objects[0].parts['fizz'] = hiwi.Object(position=[5, 3])

        parts = ['foo', 'bar', 'fizz', 'buzz', 'lorn']

        true_pos = Criterion.true_pos(image, parts)

        pred_pos = np.full((len(parts), 2), np.nan)
        pred_pos[0] = [3, 2]
        pred_pos[1] = [100, 30]
        pred_pos[3] = [1, 2]

        criterion = MaxDistance(max_dist=5, dist_cap=10,
                                classification_error=20, use_mm=True)
        result = criterion.evaluate(image, parts, pred_pos)
        assert result.correct.tolist() == [True, False, False, False, True]
        assert result.error.tolist() == [np.sqrt(8), 10, 20, 20, 0]

        result2 = criterion.evaluate(image, parts, pred_pos, true_pos)
        assert (result.correct == result2.correct).all()
        assert (result.error == result2.error).all()

        criterion = MaxDistance(max_dist=5, dist_cap=10,
                                classification_error=20)
        result = criterion.evaluate(image, parts, pred_pos)
        assert result.correct.tolist() == [True, False, False, False, True]
        assert result.error.tolist() == [np.sqrt(2), 10, 20, 20, 0]

    def test_error(self):
        image = hiwi.Image(objects=[hiwi.Object()], spacing=[2, 2])
        image.objects[0].parts['foo'] = hiwi.Object(position=[1, 2])

        criterion = MaxDistance(max_dist=5, dist_cap=10,
                                classification_error=20, use_mm=True)

        pred_pos = np.array([[2, 1],
                             [3, 2],
                             [np.nan, np.nan],
                             [100, 100]])

        assert criterion.error(image, 'foo', pred_pos[0]) == 0
        assert criterion.error(image, 'foo', pred_pos).tolist() \
            == [0, np.sqrt(8), 20, 10]
        assert criterion.error(image, 'bar', pred_pos).tolist() \
            == [20, 20, 0, 20]

        criterion = MaxDistance(max_dist=5, dist_cap=10,
                                classification_error=20)
        assert criterion.error(image, 'foo', pred_pos).tolist() \
            == [0, np.sqrt(2), 20, 10]

    def test_correct_configurations(self):
        image = hiwi.Image(objects=[hiwi.Object()], spacing=[2, 2])
        image.objects[0].parts['foo'] = hiwi.Object(position=[1, 2][::-1])
        image.objects[0].parts['bar'] = hiwi.Object(position=[10, 20][::-1])

        criterion = MaxDistance(max_dist=5, dist_cap=10,
                                classification_error=20, use_mm=True)

        candidates = {'foo': np.array([[30, 20]]),
                      'bar': np.array([[10, 17]])}

        configs = criterion.correct_configurations(image, ['foo', 'bar'],
                                                   candidates)
        assert len(configs) == 0

        candidates = {'foo': np.array([[30, 20], [2, 2], [2, 3]]),
                      'bar': np.array([[10, 18], [11, 21]])}

        configs = criterion.correct_configurations(image, ['foo', 'bar', 'jo'],
                                                   candidates)
        assert len(configs) == 4

        assert np.allclose(configs[0], [[2, 2], [11, 21], [np.nan, np.nan]],
                           equal_nan=True)
        assert np.allclose(configs[3], [[2, 3], [10, 18], [np.nan, np.nan]],
                           equal_nan=True)

        configs = criterion.correct_configurations(image, ['foo'], candidates)
        assert len(configs) == 2

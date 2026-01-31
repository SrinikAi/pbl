"""Command line interface - the **main entry point**

Although this library can be used in pure Python way, the easiest way to
interact with it is using the **command line interface** (CLI). The main
entry point is the **`main`** function, which is also called by the `pbl` CLI
command that is automatically generated when you install this package.

We use `click` to build the interface and created various additional things
to make our life easier:

- **Command class:** `Command`
- **Option class:** `Option`
- **Parameter types:**
    * `ModelType` / `MODEL`
    * `PotentialSpecType` / `POTENTIAL_SPEC`
    * `ListType` / `LIST`
    * `PathType`
    * `FloatRange`
    * `ChoiceType`
    * `TimeSpanType` / `TIME_SPAN`
"""


import click
import fnmatch
import functools
import hiwi
import itertools
import logging
import numpy as np
import os
import SimpleITK as sitk
import subprocess
import sys
import tabulate
import tensorflow as tf

from collections import OrderedDict
from datetime import datetime
from enum import Enum
from hiwi import ImageList, LocalMaxLocator
from hiwi.cli import IMAGE_LIST
from humanfriendly import parse_timespan
from natsort import natsorted
from pathlib import Path
from typing import List, Optional, Tuple

from pbl.graph import Graph
from pbl.evaluation import MaxDistance, plot_results, SetLocResult, digits
from pbl.evaluation import ImageResult, Metrics
from pbl.potentials import RteLocalizer, MultiPartCNN
from pbl.potentials import DistancePotential, AnglePotential
from pbl.potentials import VectorPotential, GaussianVector, KdeVectorPotential
from pbl.potentials import LearnablePotentialMixin
from pbl.learning import SgdMaxMarginLearning
from pbl.learning import FullSgdMaxMarginLearning, create_tf_session
from pbl.learning import Regularization as LearnRegularization
from pbl.learning import CgMaxMarginLearning
from pbl.utils import working_dir, find_image_mode


tabulate.PRESERVE_WHITESPACE = True


log = logging.getLogger(__name__)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class ModelType(click.ParamType):
    """Expects the path to a `pbl` model which is than deserialized."""

    name = 'model'

    def convert(self, value, param, ctx):
        try:
            session = tf.get_default_session() or tf.Session()

            # we do not want the session to close
            with session.as_default():
                return Graph.load(value)
        except Exception as e:
            self.fail(f'invalid model ({e})', param, ctx)


class PotentialSpecType(click.ParamType):
    """String-based specification of potentials and their cliques."""

    name = 'TYPE[:PARTS,…]'

    def __init__(self, types: type) -> None:
        super().__init__()
        self.types = types

    def convert(self, value, param, ctx):
        value = value.split(':')

        if len(value) > 2:
            self.fail('invalid format, must be TYPE[:PARTS,…]', param, ctx)

        try:
            pot_type = next(m for n, m in self.types.__members__.items()
                            if str(m) == value[0])
        except StopIteration:
            self.fail(f'unknown potential type "{value[0]}", must be one of: '
                      + ', '.join(map(str, self.types)),
                      param, ctx)

        try:
            parts = value[1].split(',')
        except IndexError:
            parts = None

        return pot_type, parts


class ListType(click.ParamType):
    """A Click string argument consisting of multiple separated arguments."""

    name = 'STR[,STR]...'

    def convert(self, value, param, ctx):
        if value.strip() == '':
            return []

        return value.split(',')


class IntListType(ListType):
    """A Click string argument consisting of comma separated integers."""

    name = 'INT[,INT]...'

    def convert(self, value, param, ctx):
        items = super().convert(value, param, ctx)

        try:
            return [int(x) for x in items]
        except Exception:
            self.fail('{} not a valid integer list'.format(value))


class FloatListType(ListType):
    """A Click string argument consisting of comma separated floats."""

    name = 'NUM[,NUM]...'

    def convert(self, value, param, ctx):
        items = super().convert(value, param, ctx)

        try:
            return [float(x) for x in items]
        except Exception:
            self.fail('{} not a valid float list'.format(value))


class PathType(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""

    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


class FloatRange(click.ParamType):
    """A Click float argument that lies within a range.

    Arguments:
        min_value: The minimal value, inclusive.
        max_value: The maximal value, inclusive.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        self.range = (min_value, max_value)

    def convert(self, value, param, ctx):
        try:
            value = float(value)
        except ValueError:
            self.fail(f'{value} not a number')

        if value < self.range[0] or value > self.range[1]:
            self.fail('{} not in range [{}, {}]'.format(value, *self.range))

        return value

    @property
    def name(self):
        return '{}-{}'.format(*self.range)


class ChoiceType(click.Choice):
    """A better choice type that supports enums inheriting `Choice`."""

    def __init__(self, enum):
        super().__init__(list(map(str, enum)))
        self.enum = enum

    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        return next(v for v in self.enum if str(v) == value)


class TimeSpanType(click.ParamType):
    """Parse human readable time spans, e.g., '2h' or '3d'."""

    name = 'TIME_SPAN'

    def convert(self, value, param, ctx):
        try:
            return parse_timespan(value)
        except Exception as e:
            self.fail('{} not a valid (known) time span: {}'.format(value, e))


class Choice(str, Enum):
    """Base enum choice type."""

    def __str__(self):
        return str(self.value)


class Missing(Choice):
    """Describes for which parts to learn the missing label."""

    NONE = 'none'
    MISSING_PARTS = 'missing-parts'
    ALL_PARTS = 'all-parts'


class Potential(Choice):
    DISTANCE = 'distance'
    ANGLE = 'angle'
    ANGLE_XY = 'angle_xy'
    ANGLE_YZ = 'angle_yz'
    ANGLE_XZ = 'angle_xz'
    VECTOR = 'vector'
    VECTOR_NEW = 'vector_new'
    VECTOR_KDE = 'vector_kde'

    @property
    def cls(self) -> type:
        return {Potential.DISTANCE: DistancePotential,
                Potential.ANGLE: AnglePotential,
                Potential.ANGLE_XY: AnglePotential,
                Potential.ANGLE_YZ: AnglePotential,
                Potential.ANGLE_XZ: AnglePotential,
                Potential.VECTOR: VectorPotential,
                Potential.VECTOR_NEW: GaussianVector,
                Potential.VECTOR_KDE: KdeVectorPotential}[self]


class Localizer(Choice):
    """Different kinds of localizers to use as unary potentials and
    candidate proposal generators.
    """

    RTE = 'rte'
    """Regression tree ensembles."""

    NCLASS_CNN = 'nclass-cnn'
    """A multi-class (part) fully CNN."""


class Learning(Choice):
    """Different learning approaches with different capabilities."""

    MAXMARGIN_SGD = 'maxmargin-sgd'
    """Simple stochastic gradient descent applied to max-margin hinge loss
    to learn weights and unknown energies.
    """

    MAXMARGIN_CG = 'maxmargin-cg'
    """Using constraint generation to learn potential weights."""

    FULL_MAXMARGIN_SGD = 'full-maxmargin-sgd'
    """Stochastic gradient descent applied to max-margin hinge loss enhanced
    by potential specific terms to learn the weights, unknown energies and
    potential params, if possible.
    """


class Regularization(Choice):
    """Different regularizations."""
    L1 = 'l1'
    L2 = 'l2'


class HelpGroup(Choice):
    RTE = 'Regression tree ensemble'
    NCLASS_CNN = 'N-class CNN'
    LEARNING = 'Common learning'
    SGD = 'Max-margin SGD learning'
    FSGD = 'Full (pot. + weights) max-margin SGD learning'
    CG_LEARNING = 'Max-margin Constraint Generation learning'


class Option(click.Option):
    """Custom option type that supports `help_group`s.

    Must be used in combination with the custom `Command`.
    """
    def __init__(self, *args, help_group: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.help_group = help_group


class Command(click.Command):
    def format_options(self, ctx, formatter):
        help_groups = OrderedDict()

        for param in self.get_params(ctx):
            help_record = param.get_help_record(ctx)
            if not help_record:
                continue
            help_group = getattr(param, 'help_group', None)
            help_groups.setdefault(help_group, []).append(help_record)

        for group, help_records in help_groups.items():
            with formatter.section(f'{group} options' if group else 'Options'):
                formatter.write_dl(help_records)


MODEL = ModelType()
POTENTIAL_SPEC = PotentialSpecType(Potential)
LIST = ListType()
INT_LIST = IntListType()
FLOAT_LIST = FloatListType()
TIME_SPAN = TimeSpanType()


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Part-based localization."""
    hiwi.show_logs(level=logging.WARNING)


@main.command(cls=Command)
@click.option('--parts', '-k', metavar='NAME[,…]', type=LIST, default='',
              help='Names of the parts we want to localize, if not given '
                   'all annotated parts are used; you can use glob characters '
                   '(?*[seq]) to match against available parts to simplify '
                   'the selection of subsets')
@click.option('--ignore', '-i', metavar='NAME[,…]', type=LIST, default='',
              help='Part names to ignore, is applied after selecting parts '
                   'and supports the same glob expressions as --parts')
@click.option('--missing', '-m', type=ChoiceType(Missing),
              default=Missing.MISSING_PARTS, show_default=True,
              help='Whether to support missing parts, and if so, which parts '
                   'must support the missing label')
@click.option('--localizer', '-l', type=ChoiceType(Localizer),
              default=Localizer.NCLASS_CNN, show_default=True,
              help='Specifies the localizer that is used to generate part '
                   'candidates and is represented as unary term in the '
                   'graph')
@click.option('--potential', '-p', 'potential_specs',
              type=POTENTIAL_SPEC, multiple=True,
              help='Creates a potential (valid types are: ' +
                   ', '.join(map(str, Potential)) +
                   ') for one clique if parts are specified, or all possible '
                   'cliques otherwise; this option can be used multiple times')
@click.option('--min-pot-samples', type=int,
              default=5, show_default=True,
              help='Required amount of training samples per potential before '
                   'discarding it. I.e., statistics of 0 samples are hard to '
                   'compute...')
@click.option('--candidates', '-n', type=int,
              default=15, show_default=True,
              help='Number of maximal candidates (localization hypothesis) to '
                   'use for each key point')
@click.option('--cand-dist', '-c', type=FLOAT_LIST,
              default='10', show_default=True,
              help='Minimal distance between localization hypotheses, this '
                   'can be specified for each axis individually (X[,Y[,Z]]), '
                   'for medical images this is assumed to be mm')
@click.option('--learning', '-t', type=ChoiceType(Learning),
              default=Learning.MAXMARGIN_SGD, show_default=True,
              help='Specifies the learning strategy to use.')
@click.option('--train-pot-frac', '-P', type=float,
              default=0.6, show_default=True,
              help='Amount of training images to use for training the '
                   'potential functions (i.e., CNN weights, RFs, vector '
                   'statistics, etc.). The rest is used to estimate the '
                   'weights and the unknown energies.')
@click.option('--max-dist', '-d', type=float, default=10., show_default=True,
              help='Maximally allowed distance prior to treating a '
                   'localization as incorrect')
@click.option('--val-frac', '-v', type=FloatRange(0, 1),
              help='Uses this fraction of training images, randomly sampled, '
                   'as validation images, if none are given.')
@click.option('--no-verbose', is_flag=True,
              help='Does not create debug outputs during training which is '
                   'generally faster but does not generate helpful artifacts.')
@click.option('--no-plot-potentials', is_flag=True,
              help='Does not create illustrations of the energies of the '
                   'different potentials.')
@click.option('--force', '-f', is_flag=True,
              help='Do not ask questions, just overwrite')
@click.option('--rte-trees', type=int, default=48, show_default=True,
              help='Number of regression trees in each ensemble',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-features', type=int, default=256, show_default=True,
              help='Number of intensity differences (features) to compute for '
                   'each position',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-patch-size', type=INT_LIST,
              default='100', show_default=True,
              help='Region in which to sample values from for a certain pixel',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-smoothing', type=FLOAT_LIST,
              help='Sigma of the Gaussian kernel that is used for smoothing, '
                   'if not given 2.15 is scaled anti-proportional with the '
                   'spacing',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-origin-features', type=float,
              default=0.3, show_default=True,
              help='Amount of features (fraction) that is computed w.r.t. to '
                   'the center pixel',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-max-depth', type=int, default=15, show_default=True,
              help='Maximal tree depth',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--rte-oob-value', type=float,
              help='Uses the given value for out-of-bounds pixel values '
                   'instead of using the data population mean',
              help_group=HelpGroup.RTE, cls=Option)
@click.option('--ncnn-patch-size', type=INT_LIST,
              help='Patch size to feed into the network. Defaults '
                   'to using the maximal image size as input shape. Specify '
                   'this if you get OOM errors.',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-l2', type=float, default=0., show_default=True,
              help='Amount of L2-norm over weights applied.',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-margin-loss', is_flag=True,
              help='Use a relative loss formulation based on the margin '
                   'between the correct value and all other values.',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-elastic-transform', type=float, metavar='0-1',
              help='If given, uses elastic transformation to add data '
                   'augmentation, the value corresponds to the chance of '
                   'applying elastic transformation to a training sample',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-dropout-rate', type=float, metavar='0-1',
              help='If given, applies dropout for better generalization. The '
                   'given number represents the dropout chance, good values '
                   'are in the range 0.2 - 0.5.',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-flip-axis', type=(click.Choice(['x', 'y', 'z']),
                                        float),
              metavar='x|y|z CHANCE', default=('x', 0),
              help='If given, performs flips along the given axis with the '
                   'given chance. E.g., "z 0.25".',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-value-aug', type=(float, float, float),
              metavar='SHIFT SCALE CHANCE', default=(0, 0, 0),
              help='If given, performs data augmentation by scaling the values'
                   ' with [1-SCALE, 1+SCALE] and shifting them by'
                   '[-SHIFT, SHIFT]. E.g., "0.25 0.25 0.25".',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-target-value', type=float,
              default=10_000, show_default=True,
              help='The target heatmap value that should be at the very peak. '
                   'Make sure that this value is a lot larger than the input '
                   'value range.',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-metrics-every', type=int, metavar='ITERATION',
              default=100, show_default=True,
              help='Computes metrics every X-th iteration',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--ncnn-max-time', type=TIME_SPAN,
              default='12h', show_default=True,
              help='How much time should be spent on training the CNN',
              help_group=HelpGroup.NCLASS_CNN, cls=Option)
@click.option('--learning-batch-size', type=int,
              default=8, show_default=True,
              help='Number of training samples to use in each iteration',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--learning-metrics-every', type=int, metavar='ITERATION',
              default=10, show_default=True,
              help='Computes metrics every X-th iteration',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--learning-max-time', type=TIME_SPAN,
              default='1h', show_default=True,
              help='How much time should be spent on optimizing the graph',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--learning-max-stagnation', type=int,
              default='1000', show_default=True,
              help='Number of iterations without improvement after which '
                   'the optimization is stoppped.',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--learning-correct-rivals', is_flag=True,
              help='This option allows rivals that are fully correct, but '
                   'have a higher error than the correct configuration. Per '
                   'default, only rivals that are `incorrect` are used.',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--learning-approx-test-inference', is_flag=True,
              help='This flag allows to use approximate inference to compute '
                   'evaluation results during testing (the generated values '
                   'are not used during learning but only logged)',
              help_group=HelpGroup.LEARNING, cls=Option)
@click.option('--sgd-learning-rate', type=float,
              default='0.1', show_default=True,
              help='The learning rate of the Adam algorithm.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-margin', type=float,
              default='0.1', show_default=True,
              help='The energy margin to satisfy.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-regularization', type=ChoiceType(Regularization),
              help='An optional regularization applied to the weights.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-regularization-factor', type=float,
              default='0.1', show_default=True,
              help='Scaling of the regularization, applies only if '
                   '--sgd-regularization is given.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-keep-rivals', is_flag=True,
              help='If given, store all found rivals so far and re-use them '
                   'in each iteration.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-min-weight', type=float,
              help='Ensures that the weight never gets below this value, if '
                   'given',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-nonzero-weight', is_flag=True,
              help='If given, ensures that weights never become zero.',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--sgd-energy-first', is_flag=True,
              help='If given, starts optimizing the energies before jointly '
                   'optimizing both',
              help_group=HelpGroup.SGD, cls=Option)
@click.option('--fsgd-no-learn-weights', is_flag=True,
              help='Do not optimize the potential weights',
              help_group=HelpGroup.FSGD, cls=Option)
@click.option('--fsgd-no-learn-unknown-energies', is_flag=True,
              help='Do not optimize the unknown energy values',
              help_group=HelpGroup.FSGD, cls=Option)
@click.option('--fsgd-margin-factor', type=float,
              default=10., show_default=True,
              help='Factor that is applied to the error, which is then used '
                   'as minimal margin that must be saturated',
              help_group=HelpGroup.FSGD, cls=Option)
@click.option('--cg-slack-factor', type=float,
              default=1000, show_default=True,
              help='The bigger this is the more the optimizer tries to reduce'
                   ' the slack variables towards 0',
              help_group=HelpGroup.CG_LEARNING, cls=Option)
@click.option('--cg-loss-function', type=int,
              default=1, show_default=False,
              help='chooses loss function (l1 or l2 norm)',
              help_group=HelpGroup.CG_LEARNING, cls=Option)
@click.option('--cg-solver', type=float,
              default=1, show_default=True,
              help='Chooses solver. In theory 1 should be best but at least'
                   ' once 2 was more stable',
              help_group=HelpGroup.CG_LEARNING, cls=Option)
@click.option('--cg-dynamic_margin', type=bool,
              default=False, show_default=True,
              help='use error reduction to get a dynamic margin or not',
              help_group=HelpGroup.CG_LEARNING, cls=Option)
@click.option('--cg-error-red-func', type=int,
              default=1, show_default=False,
              help='Chooses the error reduction function if dynamic margin'
                   ' is active. At this point only 1 exists',
              help_group=HelpGroup.CG_LEARNING, cls=Option)
@click.argument('train_images', type=IMAGE_LIST)
@click.argument('model_file', type=PathType(dir_okay=False))
@click.argument('validation_images', type=IMAGE_LIST, required=False)
def train(parts: List[str], ignore: List[str], missing: Missing,
          localizer: Localizer,
          potential_specs: Tuple[type, Optional[List[str]]],
          min_pot_samples: int,
          candidates: int, cand_dist: List[float],
          learning: Learning, train_pot_frac: float,
          max_dist: float, val_frac: Optional[float],
          no_verbose: bool, no_plot_potentials: bool, force: bool,
          rte_trees: int, rte_features: int, rte_patch_size: List[int],
          rte_smoothing: Optional[List[float]],
          rte_max_depth: int, rte_origin_features: float,
          rte_oob_value: Optional[float],
          ncnn_patch_size: Optional[List[int]], ncnn_l2: float,
          ncnn_margin_loss: bool, ncnn_elastic_transform: Optional[float],
          ncnn_dropout_rate: Optional[float],
          ncnn_flip_axis: Tuple[str, float],
          ncnn_value_aug: Tuple[float, float, float],
          ncnn_target_value: float,
          ncnn_metrics_every: int, ncnn_max_time: float,
          learning_batch_size: int, learning_metrics_every: int,
          learning_max_time: float, learning_max_stagnation: int,
          learning_correct_rivals: bool,
          learning_approx_test_inference: bool,
          sgd_learning_rate: float, sgd_margin: float,
          sgd_regularization: Optional[Regularization],
          sgd_regularization_factor: float,
          sgd_keep_rivals: bool,
          sgd_min_weight: Optional[float], sgd_nonzero_weight: bool,
          sgd_energy_first: bool,
          fsgd_no_learn_weights: bool, fsgd_no_learn_unknown_energies: bool,
          fsgd_margin_factor: float,
          cg_slack_factor,
          cg_loss_function,
          cg_solver,
          cg_dynamic_margin,
          cg_error_red_func,
          train_images: hiwi.ImageList, model_file: Path,
          validation_images: hiwi.ImageList):
    """Trains a new PBL model.

    Examples:

     - Train a new fully connected graphical model using vector potentials and
       CNN as localizer.

         $ pbl train -l cnn -p vector train_images.iml my_model.pbl

    """
    hiwi.show_logs(level=logging.WARNING if no_verbose else logging.DEBUG)

    # do not overwrite existing models
    if not force and model_file.exists():
        click.confirm(f'{model_file}: Already exists, overwrite?', abort=True)

    # create temp dir
    if not no_verbose:
        working_dir.set_inplace(model_file.parent / (model_file.name + '-tmp'))
        now = f'{datetime.now():%Y%m%d-%H%M%S}'

        hiwi.write_logs(working_dir / f'{now}_log.txt')
        log.debug('Debug information are written to %s', working_dir)

        log_python_dependencies(working_dir / f'{now}_requirements.txt')

    log.info('Working directory: %s', os.getcwd())
    log.info('Train command arguments: %s', ' '.join(sys.argv[2:]))

    n_dims, n_channels = hiwi.guess_image_shape(train_images[0].data)
    use_mm = train_images[0].spacing is not None

    orientation, spacing = find_image_mode(train_images)
    log.info('Always converting images to %s with spacing %s',
             orientation or 'original orientation (unchanged)',
             spacing[::-1] if spacing is not None else '--- (not available)')
    safe_spacing = np.ones(n_dims) if spacing is None else spacing

    # shuffle train images, also needed for proper validation sampling
    rng = np.random.RandomState(42)
    rng.shuffle(train_images)

    all_images = list(train_images) + list(validation_images or [])

    # sample validation images
    if val_frac is not None:
        if validation_images:
            raise click.BadOptionUsage('--val-frac', 'cannot use --val-frac '
                                                     'when VALIDATION_IMAGES '
                                                     'is given')

        n_val = int(np.round(val_frac * len(train_images)))
        if n_val == 0:
            raise click.BadParameter(f'must be at least {1/len(train_images)}',
                                     param_hint='--val-frac')

        validation_images = train_images[:n_val]
        train_images = train_images[n_val:]

    # split training images into training images for potentials and weights
    train_split_idx = int(np.ceil(len(train_images) * train_pot_frac))
    train_pots_images = train_images[:train_split_idx]
    train_weights_images = train_images[train_split_idx:]

    available_parts = set([p for image in train_images for
                           p in image.objects[0].parts.keys()])

    # use all available parts if none given
    if not parts:
        parts = available_parts
    else:
        parts = set(sum([fnmatch.filter(available_parts, part)
                         for part in parts], []))
    # remove ignored ones
    parts = [part for part in parts
             if not any(fnmatch.fnmatch(part, pat) for pat in ignore)]
    parts = natsorted(parts)

    # find parts that must support the unknown label
    support_unknown = set()
    if missing == Missing.ALL_PARTS:
        support_unknown = set(parts)
    elif missing == Missing.MISSING_PARTS:
        support_unknown = set([part for part in parts
                               if any(part not in image.objects[0].parts
                                      for image in train_images)])

    criterion = MaxDistance(max_dist=max_dist, use_mm=use_mm,
                            classification_error=max_dist,
                            dist_cap=max_dist)

    log.info('Training model to detect %i parts (%s%s) in %i-d images (%s '
             'channels) on %i training and %s validation images (use mm = %s)',
             len(parts),
             ', '.join('{}{}'.format(part, '?' if part in support_unknown
                                     else '')
                       for part in parts),
             '; "?" indicating support of missing label' if support_unknown
             else '', n_dims, n_channels or 'no', len(train_images),
             len(validation_images) if validation_images else 'no',
             use_mm)
    log.info('Using %i images for training potentials and %i images for '
             'learning weights (and unknown energies)',
             len(train_pots_images), len(train_weights_images))

    session = create_tf_session()

    potentials = []

    # the candidate position extractor
    peak_finder = LocalMaxLocator(
        max_peaks=candidates, min_distance=(np.ones(n_dims) * cand_dist)[::-1],
        use_mm=use_mm,
        refine=localizer == Localizer.NCLASS_CNN or (safe_spacing > 1.1).any())

    # add localizers
    if localizer == Localizer.NCLASS_CNN:
        log.debug('Use MultiPartCNN as localizer')

        if ncnn_patch_size is None:
            max_input_shape = np.max([image.data.shape[:n_dims]
                                      for image in all_images], axis=0)
        else:
            max_input_shape = np.array(ncnn_patch_size)[::-1]

        cnn = MultiPartCNN(session=session, parts=parts, n_dims=n_dims,
                           n_channels=n_channels or 1,
                           l2_norm=ncnn_l2,
                           input_shape=max_input_shape,
                           peak_finder=peak_finder,
                           target_value=ncnn_target_value,
                           dropout_rate=ncnn_dropout_rate)
        potentials.extend(cnn.potentials)

        log.debug('Use a fixed input shape of %s for CNN', cnn.input_shape)

        if learning != Learning.FULL_MAXMARGIN_SGD:
            ncnn_flip_axis = (n_dims - 1 - 'xyz'.index(ncnn_flip_axis[0]),
                              ncnn_flip_axis[1])

            cnn.train(criterion=criterion,
                      train_images=train_pots_images,
                      val_images=train_weights_images,
                      relative_loss=ncnn_margin_loss,
                      elastic_transform=ncnn_elastic_transform,
                      metrics_every=ncnn_metrics_every,
                      max_train_time=ncnn_max_time,
                      flip_axis=ncnn_flip_axis,
                      value_aug=ncnn_value_aug)

    elif localizer == Localizer.RTE:
        log.debug('Use regression tree ensembles as localizer')

        if rte_oob_value is None:
            value_means = np.array([[np.mean(img.data), img.data.size]
                                    for img in train_images])
            value_mean = (value_means[:, 0]
                          * value_means[:, 1] / value_means[:, 1].sum()).sum()
            log.debug('Using the OOB value %f for RTEs', value_mean)
            rte_oob_value = value_mean
        else:
            rte_oob_value = 1.0e10

        rte_patch_size = np.ones(n_dims) * rte_patch_size
        rte_smoothing = 1 / safe_spacing[::-1] * 2.15 \
            if rte_smoothing is None \
            else np.ones(n_dims) * rte_smoothing

        # fixes incorrect arg handling in RFL
        if len(rte_smoothing) == 2:
            rte_smoothing = np.concatenate((rte_smoothing, [1]))

        rte_cache = RteLocalizer.Cache()

        for part in parts:
            potential = RteLocalizer(
                part, cache=rte_cache, n_dims=n_dims,
                rfl_kw=dict(n_trees=rte_trees,
                            n_features=rte_features,
                            n_dims=n_dims,
                            n_channels=n_channels or 1,
                            max_depth=rte_max_depth,
                            patch_size=rte_patch_size,
                            pre_smoothing=rte_smoothing,
                            frac_origin_features=rte_origin_features,
                            oob_value=rte_oob_value),
                peak_finder=peak_finder)
            potentials.append(potential)
    else:
        raise RuntimeError('should never happen!')

    # add all kinds of possible potentials
    for pot_type, pot_parts in potential_specs:
        pot_cls = pot_type.cls

        if pot_type == Potential.ANGLE and n_dims == 3:
            raise click.BadParameter(
                f'{pot_type} in 3D is not allowed, use the more specific '
                'ones stating the view', param_hint='--potential')
        if pot_cls == AnglePotential and n_dims == 2 \
                and pot_type != Potential.ANGLE:
            raise click.BadParameter(
                f'{pot_type} in 2D is not allowed, use the general option '
                f'{Potential.ANGLE}', param_hint='--potential')

        cliques = []

        if pot_parts:
            if len(pot_parts) != pot_cls.arity():
                raise click.BadParameter(
                    f'{pot_type} requires {pot_cls.arity()} parts',
                    param_hint='--potential')

            for part in pot_parts:
                if part not in parts:
                    raise click.BadParameter(
                        f'invalid part "{part}", must be one of ' +
                        ', '.join(parts), param_hint='--potential')

            cliques.append(pot_parts)
        else:
            if len(parts) < pot_cls.arity():
                raise click.BadParameter('fully connected {} potentials '
                                         'requires at least {} parts'
                                         .format(pot_type, pot_cls.arity()),
                                         param_hint='--potential')

            # all possible cliques (no dup. parts, part order does not matter)
            for pot_parts in natsorted(itertools.product(
                    *((parts,) * pot_cls.arity()))):
                pot_parts = tuple(natsorted(pot_parts))
                if len(set(pot_parts)) == len(pot_parts) \
                        and pot_parts not in cliques:
                    cliques.append(pot_parts)

            log.debug('Use %s for all possible %i connections using %i parts'
                      '(fully connected),', pot_cls.__name__, len(cliques),
                      pot_cls.arity())

        for pot_parts in cliques:
            log.debug('Use %s for connecting %s', pot_cls.__name__,
                      '<->'.join(pot_parts))

            pot_samples = np.sum([all(p in img.parts for p in pot_parts)
                                  for img in train_pots_images])
            if pot_samples < min_pot_samples:
                log.warning('Only %i (< %i) samples matching that '
                            'connection, ignoring it',
                            pot_samples, min_pot_samples)
                continue

            if pot_cls == GaussianVector:
                pot = pot_cls(session, pot_parts, n_dims=n_dims)
            elif pot_cls == AnglePotential:
                view = {Potential.ANGLE: [0, 1],
                        Potential.ANGLE_XY: [2, 1],
                        Potential.ANGLE_YZ: [1, 0],
                        Potential.ANGLE_XZ: [2, 0]}[pot_type]
                pot = pot_cls(pot_parts, view=view)
            else:
                pot = pot_cls(pot_parts)
            potentials.append(pot)

    graph = Graph(potentials, n_dims=n_dims, support_unknown=support_unknown,
                  orientation=orientation, spacing=spacing)

    log.info('The create graph structure is%s a forest',
             '' if graph.is_forest else ' not')

    # training the potentials
    for pot in graph.potentials:
        if learning != Learning.FULL_MAXMARGIN_SGD \
                or not isinstance(pot, LearnablePotentialMixin) \
                or isinstance(pot, GaussianVector):
            log.info('Training potential %s', pot)
            pot.train(train_pots_images)
        else:
            log.info('Not pre-training potential %s', pot)

    # initialize the unknown energies with some decent values
    if support_unknown:
        if localizer == Localizer.NCLASS_CNN \
                and learning == Learning.FULL_MAXMARGIN_SGD:
            log.info('Setting unknown energies initially to 1')
            for pot in graph.unknown_potentials.keys():
                pot.unknown_energy = 1.
        else:
            log.info('Estimating unknown energies heuristically')
            graph.estimate_unknown_energies(train_weights_images, criterion)

    if working_dir and not no_plot_potentials:
        log.info('Plotting potential energies using reference images')
        graph.plot_potential_energies(train_pots_images, working_dir
                                      / 'potential_energies_train',
                                      max_dist=max_dist)
        if validation_images:
            graph.plot_potential_energies(validation_images, working_dir
                                          / 'potential_energies_val',
                                          max_dist=max_dist)
        else:
            graph.plot_potential_energies(train_weights_images, working_dir
                                          / 'potential_energies_train_weights',
                                          max_dist=max_dist)

    # different weight optimization strategies
    if learning == Learning.MAXMARGIN_SGD:
        reg_mapping = {None: None,
                       Regularization.L1: LearnRegularization.L1,
                       Regularization.L2: LearnRegularization.L2}

        optimizer = SgdMaxMarginLearning(
            criterion=criterion,
            allow_correct_rivals=learning_correct_rivals,
            approx_test_inference=learning_approx_test_inference,
            unknown_energies_first=sgd_energy_first,
            keep_rivals=sgd_keep_rivals,
            min_weight=sgd_min_weight,
            nonzero_weight=sgd_nonzero_weight,
            margin=sgd_margin,
            learning_rate=sgd_learning_rate,
            regularization=reg_mapping[sgd_regularization],
            regularization_factor=sgd_regularization_factor)

    elif learning == Learning.FULL_MAXMARGIN_SGD:
        optimizer = FullSgdMaxMarginLearning(
            criterion=criterion, session=session,
            margin_factor=fsgd_margin_factor,
            learn_weights=not fsgd_no_learn_weights,
            learn_unknown_energies=not fsgd_no_learn_unknown_energies)

        # we use all training images if we do a joint learning
        train_weights_images = train_images

    elif learning == Learning.MAXMARGIN_CG:
        optimizer = CgMaxMarginLearning(criterion=criterion,
                                        slack_factor=cg_slack_factor,
                                        loss_func=cg_loss_function,
                                        solver=cg_solver,
                                        d_margin=cg_dynamic_margin,
                                        error_red_func=cg_error_red_func
                                        )
    else:
        raise RuntimeError('will never happen')

    # the graph optimization, that might take some time
    optimizer.optimize(graph, train_weights_images,
                       val_images=validation_images,
                       batch_size=learning_batch_size,
                       log_metrics_every=learning_metrics_every,
                       max_stagnation=learning_max_stagnation,
                       max_time=learning_max_time)

    graph.dump(model_file)


def test_params(func):
    @click.option('--verbose', '-v', is_flag=True, help='Show debug log '
                                                        'output.')
    @click.option('--output-dir', '-o', type=PathType(file_okay=False),
                  help='Path to a directory to store additional output like '
                       'potential energies, result images and prediction '
                       'results')
    @click.option('--scalings', '-s', type=(float, float, int),
                  metavar='START STOP STEPS', default=(1, 1, 1),
                  help='If specified, uses scaling as a global latent '
                       'variable.')
    @click.argument('model', type=MODEL, metavar='MODEL_FILE')
    @functools.wraps(func)
    def wrapper(verbose: bool, output_dir: Optional[Path],
                scalings: Tuple[float, float, int],
                model: Graph,
                *args, **kwargs):
        hiwi.show_logs(level=logging.DEBUG if verbose else logging.WARNING)

        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            hiwi.write_logs(output_dir
                            / f'log_{datetime.now():%Y%m%d-%H%M%S}.txt')

        parts = model.is_chain or model.parts
        scalings = np.linspace(*scalings)

        def test_image(image: hiwi.Image, print_positions: bool = False):
            log.info('Processing image %s', image.path)

            result = model.test(image, scalings=scalings)

            formatted_positions = format_positions_list(
                parts, result.physical_positions)
            if print_positions:
                click.echo(formatted_positions)

            if output_dir:
                (output_dir / f'{image.name}.txt').write_text(
                    formatted_positions)

                world_pos = np.array([
                    p for p in result.physical_positions.values()
                    if p is not None])
                hiwi.write_pointset_file(world_pos,
                                         output_dir / f'{image.name}.mps')

            return result

        func(*args, output_dir=output_dir, model=model, test_image=test_image,
             **kwargs)

    return wrapper


@main.command()
@test_params
@click.argument('images', type=PathType(dir_okay=False), metavar='IMAGE',
                nargs=-1)
def test(output_dir: Optional[Path], test_image, model: Graph,
         images: List[Path]) -> None:
    """Infers the result of applying a PBL model on image(s).

    Writes the localization results formatted as "label: X Y Z" in world
    coordinates onto the standard output.

    If multiple images are given, the output is annotated with the path to the
    image prior to listing the results.
    """
    for image_idx, image_path in enumerate(images):
        if len(images) > 1:
            if image_idx > 0:
                click.echo()
            click.echo(str(image_path.resolve()))

        image_itk = sitk.ReadImage(str(image_path))
        image = hiwi.Image(data=sitk.GetArrayViewFromImage(image_itk),
                           path=image_path,
                           spacing=image_itk.GetSpacing())

        result = test_image(image, print_positions=True)

        if output_dir:
            pred_pos = np.full((model.n_parts, model.n_dims), np.nan)
            for part, pos in result.positions.items():
                if pos is not None:
                    pred_pos[model.part_idx[part]] = pos

            plot_results(image, output_dir / f'{image.name}.pdf',
                         parts=model.parts,
                         pred_pos=pred_pos)


@main.command()
@click.option('--max-dist', '-d', type=float, default=10., show_default=True,
              help='Maximally allowed distance prior to treating a '
                   'localization as incorrect')
@test_params
@click.argument('image_list', type=IMAGE_LIST)
def evaluate(max_dist: float, output_dir: Optional[Path],
             model: Graph, test_image, image_list: ImageList):
    """Evaluates a trained PBL model on a set of annotated images."""
    use_mm = image_list[0].spacing is not None
    criterion = MaxDistance(max_dist=max_dist, use_mm=use_mm)

    results = []

    name_length = max(len(img.name) for img in image_list)
    n_length = digits(len(image_list))

    chain = model.is_chain
    if chain is not None:
        click.echo('Parts are illustrated as chain from {} to {}'.format(
            chain[0], chain[-1]))
        chain = [model.part_idx[c] for c in chain]

    for image_idx, image in enumerate(image_list):
        test_result = test_image(image)

        true_pos = criterion.true_pos(image, model.parts)

        best_pos = criterion.best_configuration(image, model.parts,
                                                test_result.candidates)
        localizer_pos = np.array([test_result.candidates[p][0]
                                  for p in model.parts])

        pred_pos = np.full((len(test_result.positions), model.n_dims), np.nan)
        for i, part in enumerate(model.parts):
            pos = test_result.positions[part]
            if pos is not None:
                pred_pos[i] = pos

        best_result = criterion.evaluate(image, model.parts,
                                         best_pos, true_pos)
        pred_result = criterion.evaluate(image, model.parts,
                                         pred_pos, true_pos)
        if chain is not None:
            pred_result.order = chain
        localizer_result = criterion.evaluate(image, model.parts,
                                              localizer_pos, true_pos)

        errors = pred_result.correct.size - pred_result.correct.sum()

        image_result = ImageResult(predicted=pred_result,
                                   localizer=localizer_result,
                                   best=best_result,
                                   best_cands=None,
                                   candidates=test_result.candidates)
        results.append(image_result)

        click.echo('%*d/%*d %-*s %s' % (n_length, image_idx + 1, n_length,
                                        len(image_list), name_length,
                                        image.name + ':', pred_result))

        if output_dir:
            output_name = f'{errors}_{image.name}'
            plot_results(image, output_dir / f'{output_name}.pdf',
                         parts=model.parts,
                         pred_pos=pred_result.pred_pos,
                         correct=pred_result.correct,
                         candidates=test_result.candidates)

            if errors > 0:
                image_dir = output_dir / output_name
                image_dir.mkdir(exist_ok=True)

            # FIXME: This is not correct as the positions used here might
            # not be the positions used internally due to resampling to a
            # fixed orientation and resolution, which is lost here
            # and the positions are w.r.t. the originally supplied image
            # for part, case in zip(model.parts, pred_result.cases):
            #     if case in (PartCase.LOCALIZED, PartCase.TN):
            #         continue
            #
            #     pot = next(pot for pot in model.potentials
            #                if pot.parts == [part])
            #
            #     pot_pos = []
            #
            #     for part in pot.parts:
            #         try:
            #             pot_pos.append(image.parts[part].position[::-1])
            #         except KeyError:
            #             pot_pos.append(test_result.positions[part])
            #
            #     plot_results(image,
            #                  image_dir / f'{part}_{case}_{pot.name()}.pdf',
            #                  potential=pot,
            #                  candidates=test_result.candidates,
            #                  pot_pos=np.array(pot_pos), max_dist=max_dist)

    metrics = Metrics.compute(results)
    click.echo('Final results: {}'.format(SetLocResult([r.predicted
                                                        for r in results])))
    for m, v in metrics.items():
        click.echo(' - {} = {}'.format(m, m.format(v)))


def log_python_dependencies(output: Path) -> None:
    """Helper function to write a `requirements.txt` file to `output`."""
    try:
        result = subprocess.run(['pip', 'freeze'], universal_newlines=True,
                                stdout=subprocess.PIPE, check=True)
    except Exception:
        return

    output.write_text(result.stdout, encoding='utf8')


def format_positions_list(parts: List[str], positions) -> str:
    """Creates a list of positions for the given `parts` by looking at the
    XYZ encoded (potentially none-existing) given `positions`.
    """
    formatted_positions = []
    for part in parts:
        pos = positions.get(part)
        if pos is not None:
            formatted_positions.append(pos)
    formatted_positions = tabulate.tabulate(
        formatted_positions, tablefmt='plain').split('\n')

    table = []
    for part in parts:
        pos_str = '' if positions.get(part) is None else \
            formatted_positions.pop(0)
        table.append([part + ':', pos_str])
    table = tabulate.tabulate(table, tablefmt='plain', numalign='left')

    return table

import shutil
import pytest

import psiflow
from psiflow.reference import EMTReference
from psiflow.walkers import RandomWalker, PlumedBias, BiasedDynamicWalker
from psiflow.models import MACEModel
from psiflow.learning import SequentialLearning, load_learning
from psiflow.metrics import Metrics


def test_learning_save_load(context, tmp_path):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    learning = SequentialLearning(
            path_output=path_output,
            metrics=None,
            pretraining_nstates=100,
            )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 100

    shutil.rmtree(path_output)
    path_output.mkdir()
    metrics = Metrics(
            wandb_project='pytest',
            wandb_group='test_learning_save_load',
            )
    learning = SequentialLearning(
            path_output=path_output,
            metrics=metrics,
            pretraining_nstates=99,
            )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 99
    assert learning_.metrics is not None


def test_sequential_learning(context, tmp_path, mace_config, dataset):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    reference = EMTReference()
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(plumed_input)

    metrics = Metrics('pytest', 'test_sequential_index')

    learning = SequentialLearning(
            path_output=path_output,
            metrics=metrics,
            pretraining_nstates=50,
            train_from_scratch=True,
            atomic_energies_box_size=8,
            use_formation_energy=False,
            train_valid_split=0.8,
            niterations=1,
            )
    model = MACEModel(mace_config)
    model.config_raw['max_num_epochs'] = 1
    model.initialize(dataset[:2])

    walkers = RandomWalker.multiply(
            5,
            dataset,
            amplitude_pos=0.05,
            amplitude_box=0,
            )
    assert dataset.assign_identifiers().result() == dataset.labeled().length().result()
    data = learning.run(model, reference, walkers, dataset)
    psiflow.wait()
    assert data.labeled().length().result() == len(walkers) + dataset.length().result()
    assert learning.identifier.result() == 25
    assert data.assign_identifiers().result() == 25

    data = learning.run(model, reference, walkers)
    assert data.length().result() == 0 # iteration 0 already performed

    model.reset()
    metrics = Metrics('pytest', 'test_sequential_cv')
    path_output = tmp_path / 'output_'
    path_output.mkdir()
    learning = SequentialLearning(
            path_output=path_output,
            metrics=metrics,
            pretraining_nstates=50,
            atomic_energies_box_size=8,
            train_from_scratch=True,
            use_formation_energy=True,
            train_valid_split=0.8,
            niterations=1,
            )
    walkers = BiasedDynamicWalker.multiply(
            5,
            dataset,
            bias=bias,
            steps=10,
            step=1,
            )
    data = learning.initialize_run(model, reference, walkers)
    #assert model.use_formation_energy
    #assert 'formation_energy' in data.energy_labels().result()
    #assert 'formation_energy' in model.evaluate(data).energy_labels().result()
    #assert (path_output / 'pretraining').is_dir()
    #assert data.length().result() == 55

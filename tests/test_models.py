import copy

import numpy as np
from parsl.app.futures import DataFuture

import psiflow
from psiflow.data import compute_rmse
from psiflow.models import MACE, load_model


def test_mace_init(mace_config, dataset):
    model = MACE(**mace_config)
    assert "model_future" in model._files
    assert model.model_future is None
    model.initialize(dataset[:1])
    assert model.model_future is not None

    _config = model._config

    data_str = psiflow.serialize(model).result()
    model = psiflow.deserialize(data_str)

    _config_ = model._config
    for key, value in _config.items():
        assert key in _config_
        if type(value) is not list:
            assert value == _config_[key]

    config = copy.deepcopy(mace_config)
    config["batch_size"] = (
        100000  # bigger than ntrain --> should get reduced internally
    )
    model = MACE(**config)
    model.seed = 1
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)

    # create hamiltonian and verify addition of atomic energies
    hamiltonian = model.create_hamiltonian()
    assert hamiltonian == model.create_hamiltonian()
    evaluated = hamiltonian.evaluate(dataset)

    nstates = dataset.length().result()
    energies = np.array([evaluated[i].result().energy for i in range(nstates)])
    assert not np.any(np.allclose(energies, 0.0))
    energy_Cu = 3
    energy_H = 7
    hamiltonian.atomic_energies = {
        "Cu": energy_Cu,
        "H": energy_H,
    }
    assert hamiltonian != model.create_hamiltonian()  # atomic energies

    evaluated_ = hamiltonian.evaluate(dataset)
    for i in range(nstates):
        assert np.allclose(
            energies[i],
            evaluated_.subtract_offset(Cu=energy_Cu, H=energy_H)[i].result().energy,
        )

    second = psiflow.deserialize(psiflow.serialize(hamiltonian).result())
    evaluated = second.evaluate(dataset)
    assert np.allclose(
        evaluated.get("energy").result(),
        evaluated_.get("energy").result(),
    )

    hamiltonian.atomic_energies = {"Cu": 0, "H": 0, "jasldfkjsadf": 0}
    evaluated__ = hamiltonian.evaluate(dataset)
    assert np.allclose(
        energies.astype(np.float32),
        evaluated__.get("energy").result().reshape(-1),
    )
    hamiltonian = model.create_hamiltonian()
    model.reset()
    model.initialize(dataset[:3])
    assert hamiltonian != model.create_hamiltonian()


def test_mace_train(gpu, mace_config, dataset, tmp_path):
    # as an additional verification, this test can be executed while monitoring
    # the mace logging, and in particular the rmse_r during training, to compare
    # it with the manually computed value
    training = dataset[:-5]
    validation = dataset[-5:]
    mace_config["start_swa"] = 100
    model = MACE(**mace_config)
    model.initialize(training)
    hamiltonian0 = model.create_hamiltonian()
    rmse0 = compute_rmse(
        validation.get("per_atom_energy"),
        hamiltonian0.evaluate(validation).get("per_atom_energy"),
    )
    model.train(training, validation)
    hamiltonian1 = model.create_hamiltonian()
    rmse1 = compute_rmse(
        validation.get("per_atom_energy"),
        hamiltonian1.evaluate(validation).get("per_atom_energy"),
    )
    assert rmse0.result() > rmse1.result()


def test_mace_save_load(mace_config, dataset, tmp_path):
    model = MACE(**mace_config)
    model.add_atomic_energy("H", 3)
    model.add_atomic_energy("Cu", 4)
    model.save(tmp_path)
    model.initialize(dataset[:2])
    e0 = model.create_hamiltonian().evaluate(dataset[[3]])[0].result().energy

    psiflow.wait()
    assert (tmp_path / "MACE.yaml").exists()
    assert not (tmp_path / "MACE.pth").exists()

    model.save(tmp_path)
    psiflow.wait()
    assert (tmp_path / "MACE.pth").exists()

    model_ = load_model(tmp_path)
    assert type(model_) is MACE
    assert model_.model_future is not None
    e1 = model_.create_hamiltonian().evaluate(dataset[[3]])[0].result().energy
    assert np.allclose(e0, e1, atol=1e-4)  # up to single precision


def test_mace_seed(mace_config):
    model = MACE(**mace_config)
    assert model.seed == 0
    model.seed = 111
    assert model.seed == 111
    model._config["seed"] = 112
    assert model.seed == 112

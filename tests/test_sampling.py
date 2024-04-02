import numpy as np
from ase.units import Bohr

from psiflow.hamiltonians import EinsteinCrystal, MACEHamiltonian, PlumedHamiltonian
from psiflow.models import MACEModel
from psiflow.sampling.propagate import _propagate, template
from psiflow.sampling.server import parse_checkpoint
from psiflow.sampling.walker import Walker, partition


def test_walkers(dataset):
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    einstein_ = EinsteinCrystal(dataset[0], force_constant=0.2)
    walker = Walker(dataset[0], dataset[1], einstein, temperature=300)
    assert walker.nvt
    assert not walker.npt
    assert not walker.pimd

    walkers = [walker]
    walkers.append(Walker(dataset[0], dataset[1], 0.5 * einstein_, nbeads=4))
    assert not Walker.is_similar(walkers[0], walkers[1])
    assert len(partition(walkers)) == 2
    walkers.append(Walker(dataset[0], dataset[1], einstein + plumed, nbeads=8))
    assert Walker.is_similar(walkers[1], walkers[2])
    assert len(partition(walkers)) == 2
    walkers.append(
        Walker(dataset[0], dataset[1], einstein, pressure=0, temperature=300)
    )
    assert not Walker.is_similar(walkers[0], walkers[-1])
    assert len(partition(walkers)) == 3
    walkers.append(
        Walker(dataset[0], dataset[1], einstein_, pressure=100, temperature=600)
    )
    assert len(partition(walkers)) == 3
    walkers.append(Walker(dataset[0], dataset[1], einstein, temperature=600))
    partitions = partition(walkers)
    assert len(partitions) == 3
    assert len(partitions[0]) == 2
    assert len(partitions[1]) == 2
    assert len(partitions[2]) == 2

    # nvt partition
    hamiltonians_map, weights_table = template(partitions[0])
    assert partitions[0][0].nvt
    assert len(hamiltonians_map) == 1
    assert weights_table[0] == ("TEMP", "EinsteinCrystal0")
    assert weights_table[1] == (300, 1.0)
    assert weights_table[2] == (600, 1.0)

    # pimd partition
    hamiltonians_map, weights_table = template(partitions[1])
    assert partitions[1][0].pimd
    assert len(hamiltonians_map) == 3
    assert weights_table[0] == (
        "TEMP",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "PlumedHamiltonian0",
    )
    assert weights_table[1] == (300, 0.0, 0.5, 0.0)
    assert weights_table[2] == (300, 1.0, 0.0, 1.0)

    # npt partition
    hamiltonians_map, weights_table = template(partitions[2])
    assert partitions[2][0].npt
    assert len(hamiltonians_map) == 2
    assert weights_table[0] == (
        "TEMP",
        "PRESSURE",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
    )
    assert weights_table[1] == (300, 0, 0.0, 1.0)
    assert weights_table[2] == (600, 100, 1.0, 0.0)


def test_parse_checkpoint(checkpoint):
    states, temperatures = parse_checkpoint(checkpoint)
    assert np.allclose(
        states[0].cell,
        np.array([[1, 0.0, 0], [0.1, 2, 0], [0, 0, 3]]) / Bohr,
    )
    assert np.allclose(temperatures, 0.0)


def test_propagate(dataset, mace_config):
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)

    walker0 = Walker(
        start=dataset[0],
        state=dataset[0],
        temperature=300,
        pressure=None,
        hamiltonian=plumed + einstein,
    )
    walker1 = Walker(
        start=dataset[0],
        state=dataset[0],
        temperature=600,
        pressure=None,
        hamiltonian=einstein,
    )
    simulation_outputs = _propagate(
        [walker0, walker1],
        steps=100,
        step=10,
        timestep=0.5,
        keep_trajectory=True,
    )
    assert len(simulation_outputs) == 2
    energies = [
        simulation_outputs[0]["potential{electronvolt}"].result(),
        simulation_outputs[1]["potential{electronvolt}"].result(),
    ]
    evaluated = [
        (plumed + einstein).evaluate(simulation_outputs[0].trajectory),
        einstein.evaluate(simulation_outputs[1].trajectory),
    ]
    energies_ = [
        np.array([a.info["energy"] for a in evaluated[0].as_list().result()]),
        np.array([a.info["energy"] for a in evaluated[1].as_list().result()]),
    ]
    assert len(energies[0]) == evaluated[0].length().result()
    assert np.allclose(
        energies[0],
        energies_[0],
    )
    time = simulation_outputs[0]["time{picosecond}"].result()
    assert np.allclose(
        time,
        np.arange(11) * 5e-4 * 10,
    )

    model = MACEModel(**mace_config)
    model.initialize(dataset[:3])
    hamiltonian = MACEHamiltonian.from_model(model)
    walker = Walker(
        start=dataset[0],
        state=dataset[1],
        temperature=600,
        pressure=None,
        hamiltonian=hamiltonian,
    )
    simulation_output = _propagate(
        [walker],
        steps=10,
        step=1,
        timestep=0.5,
        keep_trajectory=False,
    )[0]
    assert np.allclose(
        hamiltonian.evaluate(dataset)[0].result().info["energy"],
        simulation_output["potential{electronvolt}"].result()[0],
        atol=1e-3,
    )

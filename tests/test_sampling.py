from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

import psiflow
from psiflow.geometry import check_equality
from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian
from psiflow.models import MACE
from psiflow.sampling import optimize, optimize_dataset
from psiflow.sampling.metadynamics import Metadynamics
from psiflow.sampling.sampling import sample, template
from psiflow.sampling.server import parse_checkpoint
from psiflow.sampling.walker import (
    Walker,
    partition,
    quench,
    randomize,
    replica_exchange,
)


def test_walkers(dataset):
    mtd0 = Metadynamics("METAD: FILE=bla")  # dummy
    mtd1 = Metadynamics("METAD: FILE=bla")  # dummy
    assert not (mtd0 == mtd1)

    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    einstein_ = EinsteinCrystal(dataset[0], force_constant=0.2)
    walker = Walker(dataset[0], einstein, temperature=300, metadynamics=mtd0)
    assert walker.nvt
    assert not walker.npt
    assert not walker.pimd

    walkers = [walker]
    walkers.append(Walker(dataset[0], 0.5 * einstein_, nbeads=4, metadynamics=mtd1))
    walkers.append(Walker(dataset[0], einstein + plumed, nbeads=4))
    walkers.append(
        Walker(dataset[0], einstein, pressure=0, temperature=300, metadynamics=mtd1)
    )
    walkers.append(
        Walker(dataset[0], einstein_, pressure=100, temperature=600, metadynamics=mtd1)
    )
    walkers.append(Walker(dataset[0], einstein, temperature=600, metadynamics=mtd1))

    # nvt
    _walkers = [walkers[0], walkers[-1]]
    hamiltonians_map, weights_table, plumed_list = template(_walkers)
    assert _walkers[0].nvt
    assert len(hamiltonians_map) == 1
    assert weights_table[0] == ("TEMP", "EinsteinCrystal0", "METAD0", "METAD1")
    assert len(plumed_list) == 2
    assert weights_table[1] == (300, 1.0, 1.0, 0.0)
    assert weights_table[2] == (600, 1.0, 0.0, 1.0)

    # remove
    _walkers[0].metadynamics = None
    _walkers[1].metadynamics = None
    hamiltonians_map, weights_table, plumed_list = template(_walkers)
    assert _walkers[0].nvt
    assert len(hamiltonians_map) == 1
    assert weights_table[0] == ("TEMP", "EinsteinCrystal0")
    assert len(plumed_list) == 0
    assert weights_table[1] == (300, 1.0)
    assert weights_table[2] == (600, 1.0)

    # pimd partition
    _walkers = [walkers[1], walkers[2]]
    hamiltonians_map, weights_table, plumed_list = template(_walkers)
    assert _walkers[0].pimd
    assert len(hamiltonians_map) == 3
    assert weights_table[0] == (
        "TEMP",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "PlumedHamiltonian0",
        "METAD0",
    )
    assert weights_table[1] == (300, 0.0, 0.5, 0.0, 1.0)
    assert weights_table[2] == (300, 1.0, 0.0, 1.0, 0.0)

    # npt partition
    _walkers = [walkers[3], walkers[4]]
    with pytest.raises(AssertionError):  # mtd objects were equal
        hamiltonians_map, weights_table, plumed_list = template(_walkers)
    _walkers[0].metadynamics = Metadynamics("METAD: FILE=bla")
    hamiltonians_map, weights_table, plumed_list = template(_walkers)
    assert _walkers[0].npt
    assert len(hamiltonians_map) == 2
    assert weights_table[0] == (
        "TEMP",
        "PRESSURE",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "METAD0",
        "METAD1",
    )
    assert weights_table[1] == (300, 0, 0.0, 1.0, 1.0, 0.0)
    assert weights_table[2] == (600, 100, 1.0, 0.0, 0.0, 1.0)


def test_parse_checkpoint(checkpoint):
    states = parse_checkpoint(checkpoint)
    assert "time" in states[0].order
    assert np.allclose(
        states[0].cell,
        np.array([[1, 0.0, 0], [0.1, 2, 0], [0, 0, 3]]) * Bohr,
    )


def test_sample(dataset, mace_config):
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)

    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=5 SIGMA=0.05 HEIGHT=5
"""
    metadynamics = Metadynamics(plumed_str)
    walker0 = Walker(
        start=dataset[0],
        temperature=300,
        pressure=None,
        metadynamics=metadynamics,
        hamiltonian=0.9 * plumed + einstein,
    )
    walker1 = Walker(
        start=dataset[0],
        temperature=600,
        pressure=None,
        hamiltonian=einstein,
    )
    simulation_outputs = sample(
        [walker0, walker1],
        steps=100,
        step=10,
    )
    assert simulation_outputs[0].trajectory.length().result() == 11
    pos0 = simulation_outputs[0].trajectory[-1].result().per_atom.positions
    pos1 = walker0.state.result().per_atom.positions
    assert np.allclose(
        pos0,
        pos1,
    )
    assert np.allclose(
        simulation_outputs[0].trajectory[-1].result().cell,
        walker0.start.result().cell,
    )
    e = simulation_outputs[0]["potential{electronvolt}"].result()
    assert len(e) == 11

    o = sample([walker0], steps=20, step=2, start=10, keep_trajectory=False)[0]
    assert o.trajectory is None
    assert len(o["potential{electronvolt}"].result()) == 6
    o = sample([walker0], steps=20, step=2, start=10, keep_trajectory=True)[0]
    assert o.trajectory.length().result() == 6
    assert len(o["potential{electronvolt}"].result()) == 6

    # check whether metadynamics file has correct dependency
    with open(metadynamics.external.result().filepath, "r") as f:
        content = f.read()
        nhills = len(content.split("\n"))
        assert nhills > 3

    assert len(simulation_outputs) == 2
    energies = [
        simulation_outputs[0]["potential{electronvolt}"].result(),
        simulation_outputs[1]["potential{electronvolt}"].result(),
    ]
    energies_ = [
        (0.9 * plumed + einstein).compute(simulation_outputs[0].trajectory, "energy"),
        einstein.compute(simulation_outputs[1].trajectory, "energy"),
    ]
    assert len(energies[0]) == len(energies_[0].result())
    assert np.allclose(
        energies[0],
        energies_[0].result(),
    )
    time = simulation_outputs[0]["time{picosecond}"].result()
    assert np.allclose(
        time,
        np.arange(11) * 5e-4 * 10,
    )

    # check pot components
    assert np.allclose(
        simulation_outputs[1].get_energy(walker1.hamiltonian).result(),
        simulation_outputs[1]["potential{electronvolt}"].result(),
    )
    assert np.allclose(
        simulation_outputs[0].get_energy(walker0.hamiltonian).result(),
        simulation_outputs[0]["potential{electronvolt}"].result(),
    )

    manual_total = simulation_outputs[0].get_energy(plumed).result() * 0.9
    manual_total += simulation_outputs[0].get_energy(einstein).result()
    assert np.allclose(
        manual_total,
        simulation_outputs[0]["potential{electronvolt}"].result(),
    )

    # check PIMD output
    walker = Walker(
        start=dataset[2],
        temperature=200,
        pressure=None,
        hamiltonian=einstein,
        nbeads=11,
    )
    output = sample([walker], steps=10, step=5)[0]
    for state in output.trajectory.geometries().result():
        assert len(state) == len(dataset[2].result())
    assert output.trajectory.length().result() == 3
    assert output.temperature is not None
    assert np.abs(output.temperature.result() - 200 < 200)

    simulation_outputs = sample(
        [walker0, walker1], steps=100, step=10, observables=["ensemble_bias"]
    )
    bias = np.stack(
        [
            simulation_outputs[0]["ensemble_bias"].result(),
            simulation_outputs[1]["ensemble_bias"].result(),
        ],
        axis=0,
    )
    assert np.allclose(bias[1, :], 0)
    assert np.all(bias[0, :] > 0)

    # check that old hills are there too
    with open(metadynamics.external.result().filepath, "r") as f:
        content = f.read()
        new_nhills = len(content.split("\n"))
        assert new_nhills > 3
        assert new_nhills > nhills

    model = MACE(**mace_config)
    model.initialize(dataset[:3])
    hamiltonian = model.create_hamiltonian()
    walker = Walker(
        start=dataset[0],
        temperature=600,
        pressure=None,
        hamiltonian=hamiltonian,
    )
    walker.state = dataset[1]
    simulation_output = sample(
        [walker],
        steps=10,
        observables=[
            "potential{electronvolt}",
            "kinetic_md{electronvolt}",
            "temperature{kelvin}",
        ],
        checkpoint_step=1,
        fix_com=True,  # otherwise temperature won't match
    )[0]
    assert np.allclose(
        hamiltonian.compute(dataset, "energy").result()[0],
        simulation_output["potential{electronvolt}"].result()[0],
        atol=1e-3,
    )
    assert np.allclose(
        simulation_output.time.result(),
        10 * 0.5 / 1000,
    )
    T0 = simulation_output.temperature.result()
    T1 = simulation_output["temperature{kelvin}"].result()[-1]
    assert np.allclose(T0, T1)


def test_npt(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1e-4)
    walker = Walker(dataset[0], einstein, temperature=600, pressure=0, nbeads=2)
    output = sample([walker], steps=30)[0]
    assert output.status.result() == 0
    assert output.trajectory is None

    # cell should have changed during NPT
    assert not np.allclose(
        walker.start.result().cell,
        output.state.result().cell,
    )


def test_reset(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    walker = Walker(
        start=dataset[0],
        temperature=300,
        pressure=None,
        hamiltonian=einstein,
    )
    walker.state = dataset[1]
    assert not walker.is_reset().result()
    assert not check_equality(walker.start, walker.state).result()
    assert check_equality(walker.start, dataset[0]).result()
    assert check_equality(walker.state, dataset[1]).result()

    walker.reset(False)
    assert not walker.is_reset().result()
    walker.reset()
    assert walker.is_reset().result()
    simulation_output = sample(
        [walker],
        steps=50,
        step=10,
        keep_trajectory=True,
    )[0]
    assert simulation_output.status.result() == 0
    assert not walker.is_reset().result()
    assert simulation_output.trajectory.length().result() == 6
    assert np.allclose(
        walker.state.result().per_atom.positions,
        simulation_output.trajectory[-1].result().per_atom.positions,
    )

    walker.hamiltonian = EinsteinCrystal(dataset[0], force_constant=1000)
    simulation_output = sample(
        [walker],
        steps=50,
        step=10,
        max_force=10,
    )[0]
    assert simulation_output.status.result() == 2
    assert walker.is_reset().result()
    assert not check_equality(walker.state, simulation_output.state).result()
    assert simulation_output.trajectory.length().result() == 1

    # check timeout
    simulation_output = sample(
        [walker],
        steps=5000000,
        step=100,
    )[0]
    assert simulation_output.status.result() == 1
    assert not walker.is_reset().result()
    assert check_equality(walker.state, simulation_output.state).result()
    assert simulation_output.time.result() > 0


def test_quench(dataset):
    dataset = dataset[:20]
    einstein0 = EinsteinCrystal(dataset[3], force_constant=0.1)
    einstein1 = EinsteinCrystal(dataset[11], force_constant=0.1)
    walkers = Walker(
        start=dataset[0],
        hamiltonian=einstein0,
        temperature=300,
    ).multiply(30)

    walkers[2].hamiltonian = einstein1
    quench(walkers, dataset)

    assert check_equality(walkers[0].start, dataset[3]).result()
    assert check_equality(walkers[1].start, dataset[3]).result()
    assert check_equality(walkers[2].start, dataset[11]).result()
    assert check_equality(walkers[3].start, dataset[3]).result()


def test_randomize(dataset):
    walkers = Walker(dataset[0]).multiply(300)
    randomize(walkers, dataset)
    length = dataset.length().result()
    checks = []
    for i, walker in enumerate(walkers):
        assert walker.is_reset().result()
        checks.append(check_equality(walker.start, dataset[i % length]))
    checks = [bool(c.result()) for c in checks]
    assert not all(checks)


def test_walker_nonperiodic(dataset_h2):
    einstein = EinsteinCrystal(dataset_h2[3], force_constant=1.0)
    walker = Walker(dataset_h2[0], einstein)

    output = sample([walker], steps=20, step=2)[0]
    assert not output.state.result().periodic
    for state in output.trajectory.geometries().result():
        assert not state.periodic


def test_rex(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    walker = Walker(
        dataset[0],
        hamiltonian=einstein,
        temperature=600,
    )
    walkers = walker.multiply(2)
    replica_exchange(walkers, trial_frequency=5)
    assert walkers[0].coupling.nwalkers == len(walkers)
    assert len(partition(walkers)) == 1
    assert len(partition(walkers)[0]) == 2

    outputs = sample(walkers, steps=50, step=10)
    assert outputs[0].trajectory.length().result() == 6

    swaps = np.loadtxt(walkers[0].coupling.swapfile.result().filepath)
    assert len(swaps) > 0  # at least some successful swaps
    assert np.allclose(swaps[0, 1:], np.array([1, 0]))  # 0, 1 --> 1, 0

    walkers += Walker(dataset[0], hamiltonian=10 * einstein).multiply(2)
    with pytest.raises(AssertionError):
        replica_exchange(walkers)
    assert len(partition(walkers)) == 3
    assert partition(walkers)[0][0] == 0
    assert partition(walkers)[0][1] == 1
    assert partition(walkers)[1][0] == 2
    assert partition(walkers)[2][0] == 3


def test_walker_serialization(dataset, tmp_path):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV PACE=2 SIGMA=0.05 HEIGHT=5
FLUSH STRIDE=1
"""
    metadynamics = Metadynamics(plumed_str)
    walkers = Walker(
        dataset[0],
        hamiltonian=einstein,
        temperature=300,
        metadynamics=metadynamics,
    ).multiply(3)
    for i, walker in enumerate(walkers):
        walker.hamiltonian *= 1 / (1 + i)

    sample(walkers, steps=10, step=2)
    walkers[0].metadynamics.external.result()
    with open(walkers[0].metadynamics.external.filepath, "r") as f:
        assert len(f.read()) > 0

    data = []
    for obj in walkers:
        data.append(psiflow.serialize(obj, copy_to=tmp_path))
    for d in data:
        print(d.result())

    new_objects = [psiflow.deserialize(d.result()) for d in data]
    psiflow.wait()

    walkers_ = new_objects[:3]

    assert check_equality(walkers_[0].start, walkers[0].start).result()
    assert check_equality(walkers_[0].state, walkers[0].state).result()

    for mtd in [w.metadynamics for w in walkers_]:
        assert Path(mtd.external.filepath).exists
        with open(mtd.external.filepath, "r") as f:
            assert len(f.read()) > 0


def test_optimize(dataset):
    # TODO: test applied_pressure?

    einstein = EinsteinCrystal(dataset[2], force_constant=10)
    final = optimize(dataset[0], einstein, mode='fix_cell', f_max=1e-4).result()

    assert np.allclose(
        final.per_atom.positions,
        dataset[2].result().per_atom.positions,
        atol=1e-4,
    )
    # assert np.allclose(
    #        final.cell,
    #        dataset[2].result().cell,
    #        atol=1e-4,
    #        )
    assert np.allclose(final.energy, 0.0)  # einstein energy >= 0

    optimized = optimize_dataset(dataset[3:5], einstein, mode='fix_cell', f_max=1e-4)
    for g in optimized.geometries().result():
        assert np.allclose(g.energy, 0.0)

    geom = dataset[0].result()
    plumed_input = """
    UNITS LENGTH=A ENERGY=kj/mol TIME=fs
    CV: VOLUME
    RESTRAINT ARG=CV AT=75 KAPPA=1
    """
    plumed_v = PlumedHamiltonian(plumed_input)
    final = optimize(geom, plumed_v, f_max=1e-8).result()
    assert np.allclose(final.energy, 0.0) and np.allclose(np.linalg.det(final.cell), 75)

    plumed_input = """
    UNITS LENGTH=A ENERGY=kj/mol TIME=fs
    cell: CELL
    RESTRAINT ARG=cell.ax AT=3 KAPPA=1
    RESTRAINT ARG=cell.ay AT=.2 KAPPA=1
    RESTRAINT ARG=cell.az AT=.1 KAPPA=1
    """
    plumed_c = PlumedHamiltonian(plumed_input)
    final = optimize(geom, plumed_c, mode='fix_shape', f_max=1e-8).result()
    ratio = geom.cell / final.cell
    assert np.allclose(final.cell[0], [3, 0, 0])
    assert np.allclose(ratio[ratio != 0], ratio[0, 0])          # check isotropic scaling (for nonzero components)

    final = optimize(geom, plumed_c, mode='fix_volume', f_max=1e-8).result()
    assert np.allclose(final.cell[0], [3, .2, .1], atol=1e-4)
    assert np.allclose(np.linalg.det(geom.cell), np.linalg.det(final.cell))

    final = optimize(geom, plumed_v + plumed_c, f_max=1e-8).result()
    assert np.allclose(final.cell[0], [3, .2, .1], atol=1e-4)
    assert np.allclose(np.linalg.det(final.cell), 75)



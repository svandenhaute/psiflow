import json

import numpy as np
from ase.units import kJ, mol # type: ignore
from parsl.data_provider.files import File # type: ignore

import psiflow
from psiflow.data import Dataset
from psiflow.functions import (
    EinsteinCrystalFunction,
    HarmonicFunction,
    MACEFunction,
    PlumedFunction,
    function_from_json,
)
from psiflow.hamiltonians import (
    EinsteinCrystal,
    Harmonic,
    MixtureHamiltonian,
    PlumedHamiltonian,
    Zero,
)
from psiflow.utils._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.utils.apps import copy_app_future
from psiflow.utils.io import dump_json


def test_einstein_crystal(dataset):
    function = EinsteinCrystalFunction(
        force_constant=1.0,
        centers=dataset[0].result().per_atom.positions,
        volume=0.0,
    )

    nstates = 4
    geometries = dataset[:nstates].reset().geometries().result()
    energy, forces, stress = function.compute(geometries).values()
    assert np.all(energy >= 0)
    assert energy[0] == 0
    assert np.allclose(  # forces point to centers
        forces,
        function.centers.reshape(1, -1, 3) - dataset[:4].get("positions").result(),
    )
    assert geometries[0].energy is None
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1.0)

    forces_, stress_, energy_ = hamiltonian.compute(
        dataset[:4], "forces", "stress", "energy"
    )
    assert np.allclose(
        energy_.result(),
        energy,
    )
    assert np.allclose(
        forces_.result(),
        forces,
    )

    forces = hamiltonian.compute(dataset[:4], "forces", batch_size=3)
    assert np.allclose(
        forces.result(),
        forces_.result(),
    )


def test_einstein_force(dataset):
    einstein = EinsteinCrystal(dataset[0], 5.0)
    reference = dataset[0].result()
    delta = 0.1
    for i in range(len(reference)):
        for j in range(3):  # x, y, z
            for sign in [+1, -1]:
                geometry = reference.copy()
                geometry.per_atom.positions[i, j] += sign * delta
                forces = einstein.compute(geometry, "forces").result()
                assert np.sign(forces[0, i, j]) == (-1.0) * sign
                forces[0, i, j] = 0.0
                assert np.allclose(forces, 0.0)


def test_get_filename_hills():
    plumed_input = """
#METAD COMMENT TO BE REMOVED
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV #lkasdjf
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld #fjalsdkfj
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    plumed_input = remove_comments_printflush(plumed_input)
    plumed_input = set_path_in_plumed(plumed_input, "METAD", "/tmp/my_input")
    plumed_input = set_path_in_plumed(plumed_input, "METADD", "/tmp/my_input")
    assert (
        plumed_input.strip()
        == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
""".strip()
    )


def test_plumed_function(tmp_path, dataset, dataset_h2):
    data = dataset + dataset_h2
    plumed_str = """
D1: DISTANCE ATOMS=1,2 NOPBC
CV: BIASVALUE arg=D1
"""
    function = PlumedFunction(plumed_str)
    outputs = function.compute(data.geometries().result())

    f = 1 / (kJ / mol) * 10  # eV --> kJ/mol and nm --> A
    positions = data.get("positions").result()
    manual = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    assert np.allclose(
        outputs["energy"] * f,
        manual,
    )
    gradient = (positions[:, 0, :] - positions[:, 1, :]) / manual.reshape(-1, 1)
    assert np.allclose(
        outputs["forces"][:, 0, :] * f,
        gradient * (-1.0),
    )

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=50 KAPPA=1
"""
    function = PlumedFunction(plumed_input)
    energy, forces, stress = function.compute(dataset.geometries().result()).values()

    volumes = np.linalg.det(dataset.get("cell").result())
    energy_ = (volumes - 50) ** 2 * (kJ / mol) / 2
    assert np.allclose(
        energy,
        energy_,
    )

    hamiltonian = PlumedHamiltonian(plumed_input)
    energy_, forces_, stress_ = hamiltonian.compute(dataset)

    assert np.allclose(energy, energy_.result())
    assert np.allclose(stress, stress_.result())

    # use external grid as bias, check that file is read
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.5     2.0     70  0
     2.00000     2.6     2.0     70  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        path_hills
    )
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))

    nstates = 10
    positions = dataset[:nstates].get("positions").result()
    distance = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    distance = distance.reshape(-1, 1)

    energy = hamiltonian.compute(dataset[:10], "energy").result()

    sigma = 2 * np.ones((1, 2))
    height = np.array([70, 70]).reshape(1, -1) * (kJ / mol)  # unit consistency
    center = np.array([2.5, 2.6]).reshape(1, -1)
    energy_per_hill = height * np.exp((distance - center) ** 2 / (-2 * sigma**2))
    energy_ = np.sum(energy_per_hill, axis=1)
    assert np.allclose(
        energy,
        energy_,
        atol=1e-3,
    )

    # check that hills file didn't change
    with open(path_hills, "r") as f:
        assert f.read() == hills


def test_harmonic_function(dataset):
    reference = dataset[0].result()
    function = HarmonicFunction(
        reference.per_atom.positions,
        np.eye(3 * len(reference)),
        reference.energy,
    )
    einstein = EinsteinCrystalFunction(1.0, reference.per_atom.positions)

    energy, forces, _ = function.compute(dataset[:10].geometries().result()).values()
    energy_, forces_, _ = einstein.compute(dataset[:10].geometries().result()).values()

    assert np.allclose(energy - reference.energy, energy_)
    assert np.allclose(forces_, forces)

    harmonic = Harmonic(dataset[0], np.eye(3 * len(reference)))
    assert Harmonic.outputs == ("energy", "forces", "stress")

    energy, forces, _ = harmonic.compute(dataset[:10])
    assert np.allclose(energy.result() - reference.energy, energy_, atol=1e-5)
    assert np.allclose(forces.result(), forces_)


def test_hamiltonian_arithmetic(dataset):
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1.0)
    hamiltonian_ = EinsteinCrystal(dataset[0].result(), force_constant=1.1)
    assert not hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[0], force_constant=1.0)
    assert hamiltonian != hamiltonian_  # app future copied
    hamiltonian_.reference_geometry = hamiltonian.reference_geometry
    assert hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[1], force_constant=1.0)
    assert not hamiltonian == hamiltonian_
    assert not hamiltonian == PlumedHamiltonian(plumed_input="")

    # consider linear combination
    scaled = 0.5 * hamiltonian
    assert len(scaled) == 1
    assert scaled.get_coefficient(hamiltonian) == 0.5
    actually_scaled = EinsteinCrystal(dataset[0], force_constant=0.5)
    assert scaled.get_coefficient(actually_scaled) is None

    energy_scaled = scaled.compute(dataset[:10], "energy")
    energy_actually = actually_scaled.compute(dataset[:10], "energy")
    assert np.allclose(energy_scaled.result(), energy_actually.result())

    energy, forces, _ = hamiltonian.compute(dataset[:10])
    other = EinsteinCrystal(dataset[0], 4.0)
    mixture = hamiltonian + other
    assert len(mixture) == 2
    assert mixture == 0.9 * other + 0.1 * other + 1.0 * hamiltonian
    _ = mixture + other
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + actually_scaled) is None
    energy_, forces_, _ = mixture.compute(dataset[:10])
    assert np.allclose(energy_.result(), 5 * energy.result())
    assert np.allclose(forces_.result(), 5 * forces.result())

    zero = Zero()
    energy, forces, stress = zero.compute(dataset[:10])
    assert np.allclose(energy.result(), 0.0)
    assert hamiltonian == hamiltonian + zero
    assert 2 * hamiltonian + zero == 2 * hamiltonian

    geometries = [dataset[i].result() for i in [0, -1]]
    natoms = [len(geometry) for geometry in geometries]
    forces = zero.compute(geometries, 'forces', batch_size=1).result()
    assert np.all(forces[0, :natoms[0]] == 0.0)
    assert np.all(forces[-1, :natoms[1]] == 0.0)


def test_subtract(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1.0)
    h = einstein - einstein
    assert isinstance(h, MixtureHamiltonian)
    assert np.allclose(h.coefficients, 0.0)


def test_hamiltonian_serialize(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1.0)

    kappa = 1
    center = 100
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    plumed = PlumedHamiltonian(plumed_input)
    data = json.loads(psiflow.serialize(einstein).result())
    assert "EinsteinCrystal" in data
    assert "reference_geometry" in data["EinsteinCrystal"]["_geoms"]
    einstein_ = psiflow.deserialize(json.dumps(data))
    assert np.allclose(
        einstein.compute(dataset[:10], "energy").result(),
        einstein_.compute(dataset[:10], "energy", batch_size=3).result(),
    )

    mixed = 0.1 * einstein + 0.9 * plumed
    assert "hamiltonians" in mixed._serial
    assert "coefficients" in mixed._attrs
    data = json.loads(psiflow.serialize(mixed).result())
    assert "MixtureHamiltonian" in data
    assert "hamiltonians" in data["MixtureHamiltonian"]["_serial"]
    mixed_ = psiflow.deserialize(json.dumps(data))
    for i, h in enumerate(mixed.hamiltonians):
        if isinstance(h, EinsteinCrystal):
            assert h.force_constant == mixed_.hamiltonians[i].force_constant
            assert (
                h.reference_geometry.result()
                == mixed_.hamiltonians[i].reference_geometry
            )
        else:
            assert h == mixed_.hamiltonians[i]
        assert mixed.coefficients[i] == mixed_.coefficients[i]
    assert np.allclose(
        mixed.compute(dataset[:10], "energy").result(),
        mixed_.compute(dataset[:10], "energy").result(),
    )

    data = json.loads(psiflow.serialize(Zero()).result())
    assert "Zero" in data
    zero = psiflow.deserialize(json.dumps(data))
    assert isinstance(zero, Zero)


def test_evaluate(dataset):
    hamiltonian = EinsteinCrystal(
        geometry=dataset[0],
        force_constant=1.0,
    )
    data = dataset[:10].reset()
    evaluated = data.evaluate(
        hamiltonian,
        batch_size=None,
    )
    evaluated_ = data.evaluate(
        hamiltonian,
        batch_size=2,
    )
    energy = hamiltonian.compute(evaluated, "energy")
    for i, geometry in enumerate(evaluated.geometries().result()):
        assert not np.all(np.isnan(geometry.per_atom.forces))
        assert np.allclose(geometry.energy, energy.result()[i])
        assert np.allclose(
            evaluated_[i].result().energy,
            geometry.energy,
        )


def test_json_dump():
    data = {
        "a": np.ones((3, 3, 3, 2)),
        "b": [1, 2, 3],
        "c": (1, 2, 4),
        "d": "asdf",
        "e": copy_app_future(False),
    }
    data_future = dump_json(
        **data,
        outputs=[psiflow.context().new_file("bla_", ".json")],
    ).outputs[0]
    psiflow.wait()
    with open(data_future.filepath, "r") as f:
        data_ = json.loads(f.read())

    new_a = np.array(data_["a"])
    assert len(new_a.shape) == 4
    assert np.allclose(
        data["a"],
        new_a,
    )
    assert data_["e"] is False
    assert type(data_["b"]) is list
    assert type(data_["c"]) is list


def test_mace_function(dataset, mace_model):
    model_path = str(mace_model.model_future.filepath)
    mace_model.model_future.result()
    function = MACEFunction(
        model_path,
        device="cpu",
        dtype="float32",
        ncores=2,
        atomic_energies={},
    )
    output = function.compute(dataset[:1].geometries().result())
    energy = output["energy"]

    function = MACEFunction(
        model_path,
        device="cpu",
        dtype="float32",
        ncores=4,
        atomic_energies={"Cu": 3, "H": 11},
    )
    output = function.compute(dataset[:1].geometries().result())
    energy_ = output["energy"]
    assert np.allclose(
        energy + 11 + 3 * 3,
        energy_,
    )


def test_function_from_json(tmp_path, dataset):
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.5     2.0     70  0
     2.00000     2.6     2.0     70  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        path_hills
    )
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))

    future = hamiltonian.serialize_function()
    future.result()  # ensure file exists
    function = function_from_json(future.filepath)

    energies = hamiltonian.compute(dataset, "energy").result()
    energies_ = function.compute(dataset.geometries().result())["energy"]
    assert np.allclose(energies, energies_)


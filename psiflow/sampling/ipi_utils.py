from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import typeguard

from psiflow.hamiltonians.hamiltonian import Hamiltonian, MixtureHamiltonian, Zero
from psiflow.sampling.walker import Walker, partition


@typeguard.typechecked
class SimulationOutput:
    pass


@typeguard.typechecked
def template(walkers: list[Walker]) -> tuple[dict[str, Hamiltonian], list[tuple]]:
    assert len(partition(walkers)) == 1
    all_hamiltonians = sum([w.hamiltonian for w in walkers], start=Zero())

    # create string names for hamiltonians and sort
    names = []
    counts = {}
    for h in all_hamiltonians.hamiltonians:
        if h.__class__.__name__ not in counts:
            counts[h.__class__.__name__] = 0
        count = counts.get(h.__class__.__name__)
        counts[h.__class__.__name__] += 1
        names.append(h.__class__.__name__ + str(count))
    _, hamiltonians = zip(*sorted(zip(names, all_hamiltonians.hamiltonians)))
    _, coefficients = zip(*sorted(zip(names, all_hamiltonians.coefficients)))
    hamiltonians = list(hamiltonians)
    coefficients = list(coefficients)
    names = sorted(names)
    assert MixtureHamiltonian(hamiltonians, coefficients) == all_hamiltonians
    all_hamiltonians = MixtureHamiltonian(hamiltonians, coefficients)

    weights_header = tuple(names)
    if walkers[0].npt:
        weights_header = ("TEMP", "PRESSURE") + weights_header
    elif walkers[0].nvt:
        weights_header = ("TEMP",) + weights_header
    else:
        pass

    weights_table = [weights_header]
    for walker in walkers:
        coefficients = all_hamiltonians.get_coefficients(1.0 * walker.hamiltonian)
        if walker.npt:
            ensemble = (walker.temperature, walker.pressure)
        elif walker.nvt:
            ensemble = (walker.temperature,)
        else:
            ensemble = ()
        weights_table.append(ensemble + tuple(coefficients))

    hamiltonians_map = {n: h for n, h in zip(names, hamiltonians)}
    return hamiltonians_map, weights_table


@typeguard.typechecked
def setup_motion(walker: Walker, timestep: float) -> ET.Element:
    timestep_element = ET.Element("timestep", units="femtosecond")
    timestep_element.text = str(timestep)

    friction = ET.Element("friction", units="femtosecond")
    friction.text = "100"
    thermostat_pimd = ET.Element("thermostat", mode="pile_g")
    thermostat_pimd.append(friction)
    thermostat = ET.Element("thermostat", mode="langevin")
    thermostat.append(friction)
    if walker.nve:
        dynamics = ET.Element("dynamics", mode="nve")
        dynamics.append(timestep_element)
    elif walker.nvt:
        dynamics = ET.Element("dynamics", mode="nvt")
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
    elif walker.npt:
        dynamics = ET.Element("dynamics", mode="npt")
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
        barostat = ET.Element("barostat", mode="flexible")
        tau = ET.Element("tau", units="femtosecond")
        tau.text = "200"
        barostat.append(tau)
        barostat.append(thermostat)  # never use thermostat_pimd here!
        dynamics.append(barostat)
    else:
        raise ValueError("invalid walker {}".format(walker))

    motion = ET.Element("motion", mode="dynamics")
    motion.append(dynamics)
    return motion


@typeguard.typechecked
def setup_ensemble(weights_header: tuple[str]) -> ET.Element:
    ensemble = ET.Element("ensemble")
    if "TEMP" in weights_header:
        temperature = ET.Element("temperature", units="kelvin")
        temperature.text = "TEMP"
        ensemble.append(temperature)
    if "PRESSURE" in weights_header:
        pressure = ET.Element("pressure", units="megapascal")
        pressure.text = "PRESSURE"
        ensemble.append(pressure)
    bias = ET.Element("bias")
    weights = []
    for i, name in enumerate(weights_header):
        if name in ["TEMP", "PRESSURE"]:
            continue
        force = ET.Element("force", forcefield=name, nbeads="1")
        bias.append(force)
        weights.append("W{}".format(i))
    ensemble.append(bias)
    bias_weights = ET.Element("bias_weights")
    bias_weights.text = " [ {} ] ".format(" ".join(weights))
    ensemble.append(bias_weights)
    return ensemble


@typeguard.typechecked
def setup_forces(
    weights_header: tuple[str],
    hamiltonians: list[Hamiltonian],
) -> ET.Element:
    pass


@typeguard.typechecked
def run_ipi(
    input_xml: ET.Element,
    hamiltonian_names,
    parsl_resource_specification=None,
    inputs=[],
) -> str:
    """

    inputs: [
            dataset with starting configurations,
            *(serialized hamiltonians),

    """


@typeguard.typechecked
def _simulate(
    walkers: list[Walker],
    steps: int,
    step: int,
    timestep: float,
    use_rex: bool = False,
    keep_trajectory: bool = False,
    output_properties: Optional[list[str]] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
) -> list[SimulationOutput]:
    assert len(walkers) > 0
    hamiltonians_map, weights_table = template(walkers)

    if motion_defaults is not None:
        raise NotImplementedError

    # motion = setup_motion(walkers[0], timestep)
    # ensemble = setup_ensemble(weights_table[0])
    # forces = setup_forces(weights_table[0], hamiltonians_map)

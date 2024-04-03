from __future__ import annotations  # necessary for type-guarding class methods

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import Dataset, FlowAtoms, check_equality
from psiflow.hamiltonians.hamiltonian import Hamiltonian, Zero
from psiflow.sampling.coupling import Coupling
from psiflow.sampling.output import SimulationOutput
from psiflow.utils import copy_app_future, unpack_i


@typeguard.typechecked
def _update_walker(
    state: FlowAtoms,
    status: int,
    start: FlowAtoms,
) -> FlowAtoms:
    # success or timeout are OK; see .output.py :: SimulationOutput
    if status in [0, 1]:
        return state
    else:
        return start


update_walker = python_app(_update_walker, executors=["default_threads"])


@typeguard.typechecked
def _conditioned_reset(
    condition: bool,
    state: FlowAtoms,
    start: FlowAtoms,
) -> FlowAtoms:
    from copy import deepcopy  # copy necessary!

    if condition:
        return deepcopy(start)
    else:
        return deepcopy(state)


conditioned_reset = python_app(_conditioned_reset, executors=["default_threads"])


@dataclass
@typeguard.typechecked
class Walker:
    start: Union[FlowAtoms, AppFuture]
    hamiltonian: Hamiltonian
    state: Union[FlowAtoms, AppFuture, None] = None
    temperature: Optional[float] = 300
    pressure: Optional[float] = None
    nbeads: int = 1
    periodic: Union[bool, AppFuture] = True
    timestep: float = 0.5
    coupling: Optional[Coupling] = None

    def __post_init__(self):
        if self.state is None:
            self.state = copy_app_future(self.start)
        if type(self.start) is AppFuture:
            start = self.start.result()  # blocking
        else:
            start = self.start
        periodic = np.all(start.pbc)
        if self.periodic is None:
            self.periodic = periodic
        else:
            assert periodic == self.periodic

    def reset(self, condition: Union[AppFuture, bool] = True):
        self.state = conditioned_reset(condition, self.state, self.start)

    def is_reset(self) -> AppFuture:
        return check_equality(self.start, self.state)

    def update(self, output: SimulationOutput) -> None:
        self.state = update_walker(
            output.state,
            output.status,
            self.start,
        )

    def multiply(self, nreplicas: int) -> list[Walker]:
        if self.coupling is not None:
            raise ValueError("Cannot multiply walkers after they are coupled")
        walkers = []
        for _i in range(nreplicas):
            walker = Walker(
                start=self.start,
                hamiltonian=self.hamiltonian,
                state=self.state,
                temperature=self.temperature,
                pressure=self.pressure,
                nbeads=self.nbeads,
                periodic=self.periodic,
                timestep=self.timestep,
                coupling=self.coupling,
            )
            walkers.append(walker)
        return walkers

    @staticmethod
    def is_similar(w0: Walker, w1: Walker):
        similar_T = (w0.temperature is None) == (w1.temperature is None)
        similar_P = (w0.pressure is None) == (w1.pressure is None)
        similar_pimd = w0.pimd == w1.pimd
        similar_coupling = w0.coupling == w1.coupling
        similar_pbc = w0.periodic == w1.periodic
        similar_timestep = w0.timestep == w1.timestep
        return (
            similar_T
            and similar_P
            and similar_pimd
            and similar_coupling
            and similar_pbc
            and similar_timestep
        )

    @property
    def pimd(self):
        return self.nbeads != 1

    @property
    def nve(self):
        return (self.temperature is None) and (self.pressure is None)

    @property
    def nvt(self):
        return (self.temperature is not None) and (self.pressure is None)

    @property
    def npt(self):
        return (self.temperature is not None) and (self.pressure is not None)


@typeguard.typechecked
def partition(walkers: list[Walker]) -> list[list[Walker]]:
    partitions = []
    for walker in walkers:
        found = False
        for partition in partitions:
            if Walker.is_similar(walker, partition[0]):
                partition.append(walker)
                found = True
        if not found:
            partitions.append([walker])
    return partitions


@typeguard.typechecked
def _get_minimum_energy_states(
    coefficients: np.ndarray,
    inputs: list = [],
) -> tuple[int, ...]:
    import numpy as np

    from psiflow.data import read_dataset

    assert len(coefficients.shape) == 2
    assert len(inputs) == coefficients.shape[1]

    energies = []
    for i in range(len(inputs)):
        data = read_dataset(
            slice(None),
            inputs=[inputs[i]],
        )
        energies.append([a.info["energy"] for a in data])
    energies = np.array(energies)

    indices = []
    for c in coefficients:
        energy = np.sum(c.reshape(-1, 1) * energies, axis=0)
        indices.append(int(np.argmin(energy)))
    return tuple(indices)


get_minimum_energy_states = python_app(
    _get_minimum_energy_states,
    executors=["default_threads"],
)


@typeguard.typechecked
def quench(walkers: list[Walker], dataset: Dataset) -> None:
    all_hamiltonians = sum([w.hamiltonian for w in walkers], start=Zero())
    evaluated = [h.evaluate(dataset) for h in all_hamiltonians.hamiltonians]

    coefficients = []
    for walker in walkers:
        c = all_hamiltonians.get_coefficients(1.0 * walker.hamiltonian)
        coefficients.append(c)
    coefficients = np.array(coefficients)

    indices = get_minimum_energy_states(
        coefficients,
        inputs=[data.data_future for data in evaluated],
    )
    for i, walker in enumerate(walkers):
        walker.start = dataset[unpack_i(indices, i)][0]
        walker.reset()

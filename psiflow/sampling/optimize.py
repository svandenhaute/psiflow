from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import parsl
import typeguard
from parsl.app.app import bash_app, join_app
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.utils.io import dump_json

from ._optimize import ALLOWED_MODES, EVAL_COMMAND


def _execute_ase(
    command_launch: str,
    inputs: list[DataFuture],
    outputs: list[DataFuture],
    env_vars: dict = {},
    stdout: str = "",
    stderr: str = "",
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    tmp_command = "tmpdir=$(mktemp -d)"
    cd_command = "cd $tmpdir"
    env_command = 'export ' + ' '.join([f"{name}={value}" for name, value in env_vars.items()])
    command_start = command_launch + ' run'
    command_start += f" --input_config={inputs[0].filepath}"
    command_start += f" --start_xyz={inputs[1].filepath}"
    for future in inputs[2:]:
        command_start += f" --path_hamiltonian={future.filepath}"
    command_start += "  &"
    command_end = command_launch + ' clean'
    command_end += f" --output_xyz={outputs[0].filepath}"
    if len(outputs) == 2:
        command_end += f" --output_traj={outputs[1].filepath}"

    command_list = [
        tmp_command,
        cd_command,
        env_command,
        command_start,
        "wait;",
        command_end,
    ]
    return "\n".join(command_list)


execute_ase = bash_app(_execute_ase, executors=["ModelEvaluation"])


def setup_forces(hamiltonian: Hamiltonian) -> tuple[list[dict], list[DataFuture]]:
    hamiltonian = 1.0 * hamiltonian  # convert to mixture
    counts, forces, futures = {}, [], []
    for h, c in zip(hamiltonian.hamiltonians, hamiltonian.coefficients):
        name = h.__class__.__name__
        if name not in counts:
            counts[name] = 0
        count = counts.get(name)
        counts[name] += 1
        future = h.serialize_function(dtype="float64")          # double precision for MLPs
        futures.append(future)
        force = dict(forcefield=name + str(count), weight=str(c), file=future.filename)
        forces.append(force)
    return forces, futures


@typeguard.typechecked
def optimize(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    mode: str = 'full',
    steps: int = 500,
    keep_trajectory: bool = False,
    pressure: float = 0,
    f_max: float = 1e-3,
) -> Union[AppFuture, tuple[AppFuture, Dataset]]:

    assert mode in ALLOWED_MODES
    assert steps > 0
    assert f_max > 0

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    command_list = [f"python {EVAL_COMMAND}"]
    if definition.max_simulation_time is not None:
        max_time = 0.9 * (60 * definition.max_simulation_time)
        command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
    command_launch = " ".join(command_list)

    input_geometry = Dataset([state]).extxyz
    forces, input_forces = setup_forces(hamiltonian)

    config = dict(
        task='ASE optimisation',
        forces=forces,
        mode=mode,
        f_max=f_max,
        pressure=pressure,
        max_steps=steps,
        keep_trajectory=keep_trajectory,
    )
    input_future = dump_json(
        outputs=[context.new_file("input_", ".json")],
        **config,
    ).outputs[0]
    inputs = [input_future, input_geometry, *input_forces]

    outputs = [context.new_file("data_", ".xyz")]
    if keep_trajectory:
        outputs.append(context.new_file("opt_", ".xyz"))

    # print(*inputs, sep='\n')
    # print(*outputs, sep='\n')

    result = execute_ase(
        command_launch=command_launch,
        env_vars=definition.env_vars,
        inputs=inputs,
        outputs=outputs,
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        parsl_resource_specification=definition.wq_resources(1),
    )

    # TODO: is this hamiltonian single precision maybe?
    # final = Dataset(None, result.outputs[0]).evaluate(hamiltonian)[-1]
    final = Dataset(None, result.outputs[0])[-1]
    if keep_trajectory:
        trajectory = Dataset(None, result.outputs[1])
        return final, trajectory
    else:
        return final


@join_app
@typeguard.typechecked
def _optimize_dataset(
    geometries: list[Geometry], *args, outputs: list = [], **kwargs
) -> AppFuture:
    assert not kwargs.get("keep_trajectory", False)
    optimized = []
    for geometry in geometries:
        optimized.append(optimize(geometry, *args, **kwargs))
    return write_frames(*optimized, outputs=[outputs[0]])


@typeguard.typechecked
def optimize_dataset(dataset: Dataset, *args, **kwargs) -> Dataset:
    extxyz = _optimize_dataset(
        dataset.geometries(),
        *args,
        outputs=[psiflow.context().new_file("data_", ".xyz")],
        **kwargs,
    ).outputs[0]
    return Dataset(None, extxyz)

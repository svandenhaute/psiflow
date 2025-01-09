from __future__ import annotations  # necessary for type-guarding class methods

import math
import xml.etree.ElementTree as ET
from typing import Optional, Union

import parsl
import typeguard
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian, Zero
from psiflow.sampling.output import (
    DEFAULT_OBSERVABLES,
    SimulationOutput,
    potential_component_names,
)
from psiflow.sampling.walker import Coupling, Walker, partition
from psiflow.utils.io import save_xml


@typeguard.typechecked
def template(
    walkers: list[Walker],
) -> tuple[dict[str, Hamiltonian], list[tuple], list[AppFuture]]:
    # multiply by 1.0 to ensure result is Mixture in case len(walkers) == 1
    total_hamiltonian = 1.0 * sum([w.hamiltonian for w in walkers], start=Zero())
    assert not total_hamiltonian == Zero()

    # create string names for hamiltonians and sort
    names = []
    counts = {}
    for h in total_hamiltonian.hamiltonians:
        if h.__class__.__name__ not in counts:
            counts[h.__class__.__name__] = 0
        count = counts.get(h.__class__.__name__)
        counts[h.__class__.__name__] += 1
        names.append(h.__class__.__name__ + str(count))
    _, hamiltonians = zip(*sorted(zip(names, total_hamiltonian.hamiltonians)))
    _, coefficients = zip(*sorted(zip(names, total_hamiltonian.coefficients)))
    hamiltonians = list(hamiltonians)
    coefficients = list(coefficients)
    names = sorted(names)
    assert MixtureHamiltonian(hamiltonians, coefficients) == total_hamiltonian
    total_hamiltonian = MixtureHamiltonian(hamiltonians, coefficients)

    weights_header = tuple(names)
    if walkers[0].npt:
        weights_header = ("TEMP", "PRESSURE") + weights_header
    elif walkers[0].nvt:
        weights_header = ("TEMP",) + weights_header
    else:
        pass

    weights_table = [weights_header]
    for walker in walkers:
        coefficients = total_hamiltonian.get_coefficients(1.0 * walker.hamiltonian)
        if walker.npt:
            ensemble = (walker.temperature, walker.pressure)
        elif walker.nvt:
            ensemble = (walker.temperature,)
        else:
            ensemble = ()
        weights_table.append(ensemble + tuple(coefficients))

    # inspect metadynamics attributes and allocate additional weights per MTD
    walker_indices = []
    metad_objects = []
    for i, walker in enumerate(walkers):
        mtd = walker.metadynamics
        if mtd is not None:
            assert mtd not in metad_objects, (
                "Metadynamics biases need to be "
                "independent for uncoupled walkers. To perform actual "
                "multiple walker metadynamics, use "
                "psiflow.sampling.multiple_walker_metadynamics"
            )
            walker_indices.append(i)
            metad_objects.append(mtd)
    plumed_list = [mtd.input() for mtd in metad_objects]

    for i, row in enumerate(weights_table):
        if i == 0:
            to_append = ["METAD{}".format(i) for i in range(len(metad_objects))]
        else:
            to_append = [0] * len(metad_objects)
            try:
                index = walker_indices.index(i - 1)
                to_append[index] = 1
            except ValueError:
                pass
        weights_table[i] = row + tuple(to_append)

    hamiltonians_map = {n: h for n, h in zip(names, hamiltonians)}
    return hamiltonians_map, weights_table, plumed_list


@typeguard.typechecked
def setup_sockets(
    hamiltonians_map: dict[str, Hamiltonian],
) -> list[ET.Element]:
    sockets = []
    for name in hamiltonians_map.keys():
        ffsocket = ET.Element("ffsocket", mode="unix", name=name, pbc="False")
        timeout = ET.Element("timeout")
        timeout.text = str(
            60 * psiflow.context().definitions["ModelEvaluation"].timeout
        )
        ffsocket.append(timeout)
        exit_on = ET.Element("exit_on_disconnect")
        exit_on.text = " TRUE "
        ffsocket.append(exit_on)
        address = ET.Element("address")  # placeholder
        address.text = name.lower()
        ffsocket.append(address)

        sockets.append(ffsocket)
    return sockets


@typeguard.typechecked
def setup_motion(walker: Walker, fix_com: bool) -> ET.Element:
    timestep_element = ET.Element("timestep", units="femtosecond")
    timestep_element.text = str(walker.timestep)

    tau = ET.Element("tau", units="femtosecond")
    tau.text = "100"
    thermostat_pimd = ET.Element("thermostat", mode="pile_g")
    thermostat_pimd.append(tau)
    thermostat = ET.Element("thermostat", mode="langevin")
    thermostat.append(tau)
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
        mode = "npt"
        dynamics = ET.Element("dynamics", mode=mode)
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
        mode = "flexible"
        barostat = ET.Element("barostat", mode=mode)
        tau = ET.Element("tau", units="femtosecond")
        tau.text = "200"
        barostat.append(tau)
        barostat.append(thermostat)  # never use thermostat_pimd here!
        dynamics.append(barostat)
    else:
        raise ValueError("invalid walker {}".format(walker))

    motion = ET.Element("motion", mode="dynamics")
    motion.append(dynamics)
    fixcom = ET.Element("fixcom")
    fixcom.text = " {} ".format(fix_com)
    motion.append(fixcom)  # ensure kinetic_md ~ temperature
    return motion


@typeguard.typechecked
def setup_ensemble(weights_header: tuple[str, ...]) -> ET.Element:
    ensemble = ET.Element("ensemble")
    if "TEMP" in weights_header:
        temperature = ET.Element("temperature", units="kelvin")
        temperature.text = "TEMP"
        ensemble.append(temperature)
    else:  # set TEMP in any case to avoid i-PI from throwing weird errors
        temperature = ET.Element("temperature", units="kelvin")
        temperature.text = " 300 "
        ensemble.append(temperature)
    if "PRESSURE" in weights_header:
        pressure = ET.Element("pressure", units="megapascal")
        pressure.text = "PRESSURE"
        ensemble.append(pressure)

    bias = ET.Element("bias")
    bias_weights = ET.Element("bias_weights")
    bias_weights_list = []
    count = 0
    while True:  # metadynamics bias if present
        name = "METAD{}".format(count)
        if name in weights_header:
            force = ET.Element("force", forcefield=name.lower())
            bias.append(force)
            bias_weights_list.append(name)
            count += 1
        else:
            break
    bias_weights.text = " [ " + ", ".join(bias_weights_list) + " ] "

    if count > 0:
        ensemble.append(bias)
        ensemble.append(bias_weights)
    return ensemble


@typeguard.typechecked
def setup_forces(weights_header: tuple[str, ...]) -> ET.Element:
    forces = ET.Element("forces")
    for name in weights_header:
        if name in ["TEMP", "PRESSURE"]:
            continue
        if name.startswith("METAD"):  # added as bias, not as main force
            continue
        force = ET.Element("force", forcefield=name, weight=name.upper())
        forces.append(force)
    return forces


@typeguard.typechecked
def setup_ffplumed(nplumed: int) -> list[ET.Element]:
    ffplumed = []
    for i in range(nplumed):
        input_file = ET.Element("file", mode="xyz", cell_units="angstrom")
        input_file.text = "start_0.xyz"  # always present
        plumeddat = ET.Element("plumeddat")
        plumeddat.text = "metad_input{}.txt".format(i)
        ff = ET.Element("ffplumed", name="metad{}".format(i), pbc="False")
        ff.append(input_file)
        ff.append(plumeddat)
        ffplumed.append(ff)
    return ffplumed


@typeguard.typechecked
def setup_system_template(
    walkers: list[Walker],
    weights_table: list[tuple],
    motion: ET.Element,
    ensemble: ET.Element,
    forces: ET.Element,
) -> ET.Element:
    system_template = ET.Element("system_template")
    labels = ET.Element("labels")
    _labels = [w.upper() for w in weights_table[0]]
    labels.text = "[ INDEX, {} ]".format(", ".join(_labels))
    system_template.append(labels)

    for i, weights in enumerate(weights_table[1:]):
        instance = ET.Element("instance")
        instance.text = "[ {}, {}]".format(i, ", ".join([str(w) for w in weights]))
        system_template.append(instance)

    initialize = ET.Element("initialize", nbeads=str(walkers[0].nbeads))
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_INDEX.xyz "
    initialize.append(start)
    velocities = ET.Element("velocities", mode="thermal", units="kelvin")
    if "TEMP" in weights_table[0]:  # valid template parameter
        velocities.text = " TEMP "
    else:
        velocities.text = " 300 "
    initialize.append(velocities)

    system = ET.Element("system", prefix="walker-INDEX")
    system.append(initialize)
    system.append(motion)
    system.append(ensemble)
    system.append(forces)

    template = ET.Element("template")
    template.append(system)
    system_template.append(template)
    return system_template


@typeguard.typechecked
def setup_output(
    nwalkers: int,
    nhamiltonians: int,
    observables: Optional[list[str]],
    step: Optional[int],
    keep_trajectory: bool,
    checkpoint_step: int,
) -> tuple[ET.Element, list]:
    output = ET.Element("output", prefix="output")

    if observables is None:
        observables = []
    full_list = (
        DEFAULT_OBSERVABLES + potential_component_names(nhamiltonians) + observables
    )
    observables = list(set(full_list))

    if step is None:
        step = checkpoint_step

    checkpoint = ET.Element(
        "checkpoint",
        filename="checkpoint",
        stride=str(checkpoint_step),
        overwrite="True",
    )
    output.append(checkpoint)
    if keep_trajectory:
        trajectory = ET.Element(
            "trajectory",
            filename="trajectory",
            stride=str(step),
            format="ase",
            bead="0",
        )
        trajectory.text = r" positions "
        output.append(trajectory)
    properties = ET.Element(
        "properties",
        filename="properties",
        stride=str(step),
    )
    properties.text = " [ " + ", ".join(observables) + " ] "
    output.append(properties)

    # TODO: check whether observables are valid
    simulation_outputs = [SimulationOutput(observables) for i in range(nwalkers)]
    return output, simulation_outputs


@typeguard.typechecked
def setup_smotion(
    coupling: Optional[Coupling], plumed_list: list[AppFuture]
) -> ET.Element:
    has_metad = len(plumed_list) > 0
    has_coupling = coupling is not None
    if has_coupling:
        smotion = coupling.get_smotion(has_metad)
    else:
        smotion = ET.Element("smotion", mode="dummy")
    if has_metad:
        metaff = ET.Element("metaff")
        bias_names = ["metad{}".format(i) for i in range(len(plumed_list))]
        metaff.text = " [ " + ", ".join(bias_names) + " ] "
        metad = ET.Element("metad")
        metad.append(metaff)
        smotion_metad = ET.Element("smotion", mode="metad")
        smotion_metad.append(metad)
        if has_coupling:
            smotion.append(smotion_metad)
        else:  # overwrite dummy smotion
            smotion = smotion_metad
    return smotion


def _execute_ipi(
    nwalkers: int,
    hamiltonian_names: list[str],
    client_args: list[str],
    keep_trajectory: bool,
    max_force: Optional[float],
    coupling_command: Optional[str],
    command_server: str,
    command_client: str,
    *plumed_list: str,
    env_vars: dict = {},
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = {},
) -> str:
    tmp_command = "tmpdir=$(mktemp -d);"
    cd_command = "cd $tmpdir;"
    write_command = " "
    for i, plumed_str in enumerate(plumed_list):
        write_command += 'echo "{}" > metad_input{}.txt; '.format(plumed_str, i)
    for key, value in env_vars.items():
        write_command += " export {}={}; ".format(key, value)
    command_start = command_server + " --nwalkers={}".format(nwalkers)
    command_start += " --input_xml={}".format(inputs[0].filepath)
    command_start += " --start_xyz={}".format(inputs[1].filepath)
    command_start += "  & \n"
    command_clients = ""
    i = 0
    for i, name in enumerate(hamiltonian_names):
        args = client_args[i]
        for _j, arg in enumerate(args):
            command_ = command_client + " --address={}".format(name.lower())
            command_ += " --path_hamiltonian={}".format(inputs[2 + i].filepath)
            command_ += " --start={}".format(inputs[1].filepath)
            if max_force is not None:
                command_ += " --max_force={}".format(max_force)
            command_ += " " + arg + " "
            command_ += " & \n"
            command_clients += command_

    command_end = command_server
    command_end += " --cleanup"
    command_end += " --output_xyz={};".format(outputs[0].filepath)
    command_copy = ""
    for i in range(nwalkers):
        command_copy += "cp walker-{}_output.properties {}; ".format(
            i,
            outputs[i + 1].filepath,
        )
    if keep_trajectory:
        for i in range(nwalkers):
            command_copy += "cp walker-{}_output.trajectory_0.extxyz {}; ".format(
                i,
                outputs[i + nwalkers + 1].filepath,
            )
    if coupling_command is not None:
        command_copy += coupling_command
    command_list = [
        tmp_command,
        cd_command,
        write_command,
        command_start,
        "sleep 3s;",
        command_clients,
        "wait;",
        command_end,
        command_copy,
    ]
    return " ".join(command_list)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


@typeguard.typechecked
def _sample(
    walkers: list[Walker],
    steps: int,
    step: Optional[int] = None,
    start: int = 0,
    keep_trajectory: bool = True,
    max_force: Optional[float] = None,
    observables: Optional[list[str]] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
    fix_com: bool = True,
    prng_seed: int = 12345,
    checkpoint_step: Optional[int] = None,
    verbosity: str = "medium",
) -> list[SimulationOutput]:
    assert len(walkers) > 0
    hamiltonians_map, weights_table, plumed_list = template(walkers)
    coupling = walkers[0].coupling

    if motion_defaults is not None:
        raise NotImplementedError

    motion = setup_motion(walkers[0], fix_com)
    ensemble = setup_ensemble(weights_table[0])
    forces = setup_forces(weights_table[0])
    system_template = setup_system_template(
        walkers,
        weights_table,
        motion,
        ensemble,
        forces,
    )
    smotion = setup_smotion(coupling, plumed_list)

    # make sure at least one checkpoint is being written
    if checkpoint_step is None:  # default to every 5% of simulation progress
        if step is None:
            checkpoint_step = math.ceil(steps / 20)
        else:
            checkpoint_step = step
    else:
        if steps < checkpoint_step:  # technically a user error
            checkpoint_step = steps
    if step is not None:
        start = math.floor(start / step)  # start is applied on subsampled quantities
    if step is None:
        keep_trajectory = False
    output, simulation_outputs = setup_output(
        len(walkers),
        len(hamiltonians_map),  # for potential components
        observables,
        step,
        keep_trajectory,
        checkpoint_step,
    )
    simulation = ET.Element(
        "simulation",
        verbosity=str(verbosity),
        safe_stride=str(checkpoint_step),
    )
    sockets = setup_sockets(hamiltonians_map)
    for socket in sockets:
        simulation.append(socket)
    ffplumed = setup_ffplumed(len(plumed_list))
    for ff in ffplumed:
        simulation.append(ff)
    simulation.append(output)
    simulation.append(system_template)
    simulation.append(smotion)

    total_steps = ET.Element("total_steps")
    total_steps.text = " {} ".format(steps)
    simulation.append(total_steps)

    prng = ET.Element("prng")
    seed = ET.Element("seed")
    seed.text = " {} ".format(prng_seed)
    prng.append(seed)
    simulation.append(prng)

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    input_future = save_xml(
        simulation,
        outputs=[context.new_file("input_", ".xml")],
    ).outputs[0]
    inputs = [
        input_future,
        Dataset([w.state for w in walkers]).extxyz,
    ]

    # remove any Harmonic instances because they are not implemented with sockets
    hamiltonian_names = list(hamiltonians_map.keys())

    max_nclients = int(sum([w.nbeads for w in walkers]))
    inputs += [h.serialize_function() for h in hamiltonians_map.values()]
    client_args = []
    for name in hamiltonian_names:
        args = definition.get_client_args(name, max_nclients, motion="dynamics")
        client_args.append(args)
    outputs = [context.new_file("data_", ".xyz")]
    outputs += [context.new_file("simulation_", ".txt") for w in walkers]
    if keep_trajectory:
        outputs += [context.new_file("data_", ".xyz") for w in walkers]
        assert len(outputs) == 2 * len(walkers) + 1
    else:
        assert len(outputs) == len(walkers) + 1

    # add coupling inputs after all other ones;
    # these are updated again with the corresponding outputs from execute_ipi
    # the if/else stuff has to happen outside of bash_app because coupling cannot
    # be passed into a bash app as it cannot be serialized
    if coupling is not None:
        inputs += coupling.inputs()
        outputs += [File(f.filepath) for f in coupling.inputs()]
        coupling_copy_command = coupling.copy_command()
    else:
        coupling_copy_command = None

    command_server = definition.server_command()
    command_client = definition.client_command()
    resources = definition.wq_resources(max_nclients)
    result = execute_ipi(
        len(walkers),
        hamiltonian_names,
        client_args,
        keep_trajectory,
        max_force,
        coupling_copy_command,
        command_server,
        command_client,
        *plumed_list,
        env_vars=dict(definition.env_vars),
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )

    final_states = Dataset(None, result.outputs[0])

    for i, simulation_output in enumerate(simulation_outputs):
        state = final_states[i]
        if walkers[i].order_parameter is not None:
            state = walkers[i].order_parameter.evaluate(state)
        simulation_output.parse(result, state)
        simulation_output.parse_data(
            start,
            result.outputs[i + 1],
            hamiltonians=list(hamiltonians_map.values()),
        )
        if keep_trajectory:
            j = len(walkers) + 1 + i
            trajectory = Dataset(None, result.outputs[j])
            if start > 0:
                trajectory = trajectory[start:]
            simulation_output.trajectory = trajectory
        if walkers[i].metadynamics is not None:
            walkers[i].metadynamics.wait_for(result)
        simulation_output.update_walker(walkers[i])

    if coupling is not None:
        coupling.update(result)

    return simulation_outputs


@typeguard.typechecked
def sample(
    walkers: list[Walker],
    steps: int,
    step: Optional[int] = None,
    start: int = 0,
    keep_trajectory: bool = True,
    max_force: Optional[float] = None,
    observables: Optional[list[str]] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
    fix_com: bool = True,
    prng_seed: int = 12345,
    use_unique_seeds: bool = True,
    checkpoint_step: Optional[int] = None,
    verbosity: str = "medium",
) -> list[SimulationOutput]:
    indices = partition(walkers)
    outputs = [None] * len(walkers)
    for i, group in enumerate(indices):
        if not use_unique_seeds:
            seed = prng_seed
        else:
            seed = prng_seed + i
        _walkers = [walkers[index] for index in group]
        _outputs = _sample(
            _walkers,
            steps,
            step=step,
            start=start,
            keep_trajectory=keep_trajectory,
            max_force=max_force,
            observables=observables,
            motion_defaults=motion_defaults,
            fix_com=fix_com,
            prng_seed=seed,
            checkpoint_step=checkpoint_step,
            verbosity=verbosity,
        )
        for i, index in enumerate(group):
            outputs[index] = _outputs[i]
    return outputs

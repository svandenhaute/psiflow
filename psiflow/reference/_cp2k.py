from __future__ import annotations  # necessary for type-guarding class methods

import copy
import io
import logging
from functools import partial
from typing import Optional, Union

import numpy as np
import typeguard
from ase.data import atomic_numbers
from ase.units import Bohr, Ha
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParserSimplified
from parsl.app.app import bash_app, join_app, python_app
from parsl.app.bash import BashApp
from parsl.app.python import PythonApp
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry, NullState
from psiflow.reference.reference import Reference

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def check_input(cp2k_input_str: str):
    pass


@typeguard.typechecked
def str_to_dict(cp2k_input_str: str) -> dict:
    return CP2KInputParserSimplified(
        repeated_section_unpack=True,
        # multi_value_unpack=False,
        # level_reduction_blacklist=['KIND'],
    ).parse(io.StringIO(cp2k_input_str))


@typeguard.typechecked
def dict_to_str(cp2k_input_dict: dict) -> str:
    return "\n".join(list(CP2KInputGenerator().line_iter(cp2k_input_dict)))


@typeguard.typechecked
def insert_atoms_in_input(cp2k_input_dict: dict, geometry: Geometry):
    from ase.data import chemical_symbols

    # get rid of topology if it's there
    cp2k_input_dict["force_eval"]["subsys"].pop("topology", None)

    coord = []
    cell = {}
    numbers = geometry.per_atom.numbers
    positions = geometry.per_atom.positions
    for i in range(len(geometry)):
        coord.append("{} {} {} {}".format(chemical_symbols[numbers[i]], *positions[i]))
    cp2k_input_dict["force_eval"]["subsys"]["coord"] = {"*": coord}

    assert geometry.periodic  # CP2K needs cell info!
    for i, vector in enumerate(["A", "B", "C"]):
        cell[vector] = "{} {} {}".format(*geometry.cell[i])
    cp2k_input_dict["force_eval"]["subsys"]["cell"] = cell


@typeguard.typechecked
def set_global_section(cp2k_input_dict: dict, properties: tuple):
    if "global" not in cp2k_input_dict:
        cp2k_input_dict["global"] = {}
    global_dict = cp2k_input_dict["global"]
    if properties == ("energy",):
        global_dict["run_type"] = "ENERGY"
    elif properties == ("energy", "forces"):
        global_dict["run_type"] = "ENERGY_FORCE"
    else:
        raise ValueError("invalid properties: {}".format(properties))

    if "preferred_diag_library" not in global_dict:
        global_dict["preferred_diag_library"] = "SL"
    if "fm" not in global_dict:
        global_dict["fm"] = {"type_of_matrix_multiplication": "SCALAPACK"}


def parse_cp2k_output(
    cp2k_output_str: str, properties: tuple, geometry: Geometry
) -> Geometry:
    natoms = len(geometry)
    all_lines = cp2k_output_str.split("\n")

    # read coordinates
    lines = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith("MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"):
            skip = 3
            lines = all_lines[i + skip : i + skip + natoms]
    if lines is None:
        return NullState
    assert len(lines) == natoms
    positions = np.zeros((natoms, 3))
    for j, line in enumerate(lines):
        try:
            positions[j, :] = np.array([float(f) for f in line.split()[4:7]])
        except ValueError:  # if positions exploded, CP2K puts *** instead of float
            return NullState
    assert np.allclose(
        geometry.per_atom.positions, positions, atol=1e-2
    )  # accurate up to 0.01 A

    # try and read energy
    energy = None
    for line in all_lines:
        if line.strip().startswith("ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]"):
            energy = float(line.split()[-1]) * Ha
    if energy is None:
        return NullState
    # atoms.reference_status = True
    geometry.energy = energy
    geometry.per_atom.forces[:] = np.nan

    # try and read forces if requested
    if "forces" in properties:
        lines = None
        for i, line in enumerate(all_lines):
            if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
                skip = 3
                lines = all_lines[i + skip : i + skip + natoms]
        if lines is None:
            return NullState
        assert len(lines) == natoms
        forces = np.zeros((natoms, 3))
        for j, line in enumerate(lines):
            forces[j, :] = np.array([float(f) for f in line.split()[3:6]])
        forces *= Ha / Bohr
        geometry.per_atom.forces[:] = forces
    # atoms.info.pop("stress", None)  # remove if present for some reason
    geometry.stress = None
    return geometry


# typeguarding for some reason incompatible with WQ
def cp2k_singlepoint_pre(
    geometry: Geometry,
    cp2k_input_dict: dict,
    properties: tuple,
    cp2k_command: str,
    stdout: str = "",
    stderr: str = "",
    parsl_resource_specification: Optional[dict] = None,
):
    from psiflow.reference._cp2k import (
        dict_to_str,
        insert_atoms_in_input,
        set_global_section,
    )

    set_global_section(cp2k_input_dict, properties)
    insert_atoms_in_input(cp2k_input_dict, geometry)
    if "forces" in properties:
        cp2k_input_dict["force_eval"]["print"] = {"FORCES": {}}
    cp2k_input_str = dict_to_str(cp2k_input_dict)

    # see https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
    tmp_command = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    cd_command = "cd $mytmpdir;"
    write_command = 'echo "{}" > cp2k.inp;'.format(cp2k_input_str)
    command_list = [
        tmp_command,
        cd_command,
        write_command,
        cp2k_command,
    ]
    return " ".join(command_list)


@typeguard.typechecked
def cp2k_singlepoint_post(
    geometry: Geometry,
    properties: tuple,
    inputs: list = [],
) -> Geometry:
    from psiflow.geometry import NullState, new_nullstate
    from psiflow.reference._cp2k import parse_cp2k_output

    if geometry == NullState:
        return NullState.copy()  # copy?

    with open(inputs[0], "r") as f:
        cp2k_output_str = f.read()
    geometry = parse_cp2k_output(cp2k_output_str, properties, geometry)
    if geometry != NullState:
        geometry.stdout = inputs[0]
    else:  # a little hacky
        geometry = new_nullstate()
        geometry.stdout = inputs[0]
    return geometry


@join_app
@typeguard.typechecked
def evaluate_single(
    geometry: Union[Geometry, AppFuture],
    cp2k_input_dict: dict,
    properties: tuple,
    cp2k_command: str,
    wq_resources: dict[str, Union[float, int]],
    app_pre: BashApp,
    app_post: PythonApp,
) -> AppFuture:
    import parsl

    from psiflow.geometry import NullState
    from psiflow.utils.apps import copy_app_future

    if geometry == NullState:
        return copy_app_future(NullState)
    else:
        pre = app_pre(
            geometry,
            cp2k_input_dict,
            properties,
            cp2k_command=cp2k_command,
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            parsl_resource_specification=wq_resources,
        )
        return app_post(
            geometry=geometry,
            properties=properties,
            inputs=[pre.stdout, pre.stderr, pre],  # wait for bash app
        )


@typeguard.typechecked
@psiflow.serializable
class CP2K(Reference):
    properties: list[str]  # json does deserialize(serialize(tuple)) = list
    executor: str
    cp2k_input_str: str
    cp2k_input_dict: dict

    def __init__(
        self,
        cp2k_input_str: str,
        properties: Union[tuple, list] = ("energy", "forces"),
        executor: str = "CP2K",
    ):
        self.properties = list(properties)
        self.executor = executor
        check_input(cp2k_input_str)
        self.cp2k_input_str = cp2k_input_str
        self.cp2k_input_dict = str_to_dict(cp2k_input_str)
        self._create_apps()

    def _create_apps(self):
        definition = psiflow.context().definitions[self.executor]
        cp2k_command = definition.command()
        wq_resources = definition.wq_resources()
        app_pre = bash_app(cp2k_singlepoint_pre, executors=[self.executor])
        app_post = python_app(cp2k_singlepoint_post, executors=["default_threads"])
        self.evaluate_single = partial(
            evaluate_single,
            cp2k_input_dict=self.cp2k_input_dict,
            properties=tuple(self.properties),
            cp2k_command=cp2k_command,
            wq_resources=wq_resources,
            app_pre=app_pre,
            app_post=app_post,
        )

    def get_single_atom_references(self, element):
        number = atomic_numbers[element]
        references = []
        for mult in range(1, 16):
            if number % 2 == 0 and mult % 2 == 0:
                continue  # not 2N + 1 is never even
            if mult - 1 > number:
                continue  # max S = 2 * (N * 1/2) + 1
            cp2k_input_dict = copy.deepcopy(self.cp2k_input_dict)
            cp2k_input_dict["force_eval"]["dft"]["uks"] = "TRUE"
            cp2k_input_dict["force_eval"]["dft"]["multiplicity"] = mult
            cp2k_input_dict["force_eval"]["dft"]["charge"] = 0
            cp2k_input_dict["force_eval"]["dft"]["xc"].pop("vdw_potential", None)
            if "scf" in cp2k_input_dict["force_eval"]["dft"]:
                if "ot" in cp2k_input_dict["force_eval"]["dft"]["scf"]:
                    cp2k_input_dict["force_eval"]["dft"]["scf"]["ot"][
                        "minimizer"
                    ] = "CG"
                else:
                    cp2k_input_dict["force_eval"]["dft"]["scf"]["ot"] = {
                        "minimizer": "CG"
                    }
            else:
                cp2k_input_dict["force_eval"]["dft"]["scf"] = {
                    "ot": {"minimizer": "CG"}
                }

            reference = CP2K(
                dict_to_str(cp2k_input_dict),
                properties=list(self.properties),
                executor=self.executor,
            )
            references.append((mult, reference))
        return references

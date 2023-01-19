from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass
import tempfile
import shutil

import parsl
from parsl.app.app import python_app, bash_app
from parsl.dataflow.memoization import id_for_memo
from parsl.data_provider.files import File

from psiflow.data import FlowAtoms
from .base import BaseReference


#@typeguard.typechecked
def insert_filepaths_in_input(
        cp2k_input: str,
        files: dict[str, Union[str, list[str]]]) -> str:
    from pymatgen.io.cp2k.inputs import Cp2kInput, Keyword, KeywordList
    inp = Cp2kInput.from_string(cp2k_input)
    for name, path in files.items():
        if name == 'basis_set':
            key = 'BASIS_SET_FILE_NAME'
        elif name == 'potential':
            key = 'POTENTIAL_FILE_NAME'
        elif name == 'dftd3':
            key = 'PARAMETER_FILE_NAME'

        if isinstance(path, list): # set as KeywordList
            keywords = []
            for _path in path:
                keywords.append(Keyword(key, _path, repeats=True))
            to_add = KeywordList(keywords)
        else:
            to_add = Keyword(key, path, repeats=False)
        if key == 'BASIS_SET_FILE_NAME':
            inp.update({'FORCE_EVAL': {'DFT': {key: to_add}}}, strict=True)
        elif key == 'POTENTIAL_FILE_NAME':
            inp.update({'FORCE_EVAL': {'DFT': {key: to_add}}}, strict=True)
        elif key == 'PARAMETER_FILE_NAME':
            inp.update(
                    {'FORCE_EVAL': {'DFT': {'XC': {'VDW_POTENTIAL': {'PAIR_POTENTIAL': {key: to_add}}}}}},
                    strict=True,
                    )
        else:
            raise ValueError('File key {} not recognized'.format(key))
    return str(inp)


@typeguard.typechecked
def insert_atoms_in_input(cp2k_input: str, atoms: FlowAtoms) -> str:
    from pymatgen.io.cp2k.inputs import Cp2kInput, Cell, Coord
    from pymatgen.core import Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    structure = AseAtomsAdaptor.get_structure(atoms)
    lattice = Lattice(atoms.get_cell())

    inp = Cp2kInput.from_string(cp2k_input)
    if not 'SUBSYS' in inp['FORCE_EVAL'].subsections.keys():
        raise ValueError('No subsystem present in cp2k input: {}'.format(cp2k_input))
    inp['FORCE_EVAL']['SUBSYS'].insert(Coord(structure))
    inp['FORCE_EVAL']['SUBSYS'].insert(Cell(lattice))
    return str(inp)


@typeguard.typechecked
def regularize_input(cp2k_input: str) -> str:
    """Ensures forces and stress are printed; removes topology/cell info"""
    from pymatgen.io.cp2k.inputs import Cp2kInput
    inp = Cp2kInput.from_string(cp2k_input)
    inp.update({'FORCE_EVAL': {'SUBSYS': {'CELL': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'TOPOLOGY': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'COORD': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'FORCES': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'STRESS_TENSOR': {}}}})
    return str(inp)


@typeguard.typechecked
def set_global_section(cp2k_input: str) -> str:
    from pymatgen.io.cp2k.inputs import Cp2kInput, Global
    inp = Cp2kInput.from_string(cp2k_input)
    inp.subsections['GLOBAL'] = Global(project_name='cp2k_project')
    return str(inp)


@typeguard.typechecked
def cp2k_singlepoint_pre(
        atoms: FlowAtoms,
        parameters: CP2KParameters,
        cp2k_command: str,
        file_names: list[str],
        walltime: int = 0,
        inputs: list = [],
        outputs: list[File] = [],
        stdout: str = '',
        stderr: str = '',
        ):
    import tempfile
    import glob
    from pathlib import Path
    import numpy as np
    from psiflow.reference._cp2k import insert_filepaths_in_input, \
            insert_atoms_in_input, set_global_section
    filepaths = {} # cp2k cannot deal with long filenames; copy into local dir
    for name, file in zip(file_names, inputs):
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        tmp.close()
        shutil.copyfile(file.filepath, tmp.name)
        filepaths[name] = tmp.name
    cp2k_input = insert_filepaths_in_input(
            parameters.cp2k_input,
            filepaths,
            )
    cp2k_input = regularize_input(cp2k_input) # before insert_atoms_in_input
    cp2k_input = insert_atoms_in_input(
            cp2k_input,
            atoms,
            )
    cp2k_input = set_global_section(cp2k_input)
    # see https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_write = 'echo "{}" > cp2k.inp;'.format(cp2k_input)
    command_list = [
            command_tmp,
            command_cd,
            command_write,
            'timeout {}s'.format(max(walltime - 5, 0)), # some time is spent on copying
            cp2k_command,
            '-i cp2k.inp',
            ' || true',
            ]
    return ' '.join(command_list)


def cp2k_singlepoint_post(
        atoms: FlowAtoms,
        inputs: list[File] = [],
        ) -> FlowAtoms:
    import numpy as np
    from ase.units import Hartree, Bohr
    from pymatgen.io.cp2k.outputs import Cp2kOutput
    with open(inputs[0], 'r') as f:
        stdout = f.read()
    with open(inputs[1], 'r') as f:
        stderr = f.read()
    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    try:
        out = Cp2kOutput(inputs[0])
        out.parse_energies()
        out.parse_forces()
        out.parse_stresses()
        energy = out.data['total_energy'][0] # already in eV
        forces = np.array(out.data['forces'][0]) * (Hartree / Bohr) # to eV/A
        stress = np.array(out.data['stress_tensor'][0]) * 1000 # to MPa
        atoms.info['energy'] = energy
        atoms.info['stress'] = stress
        atoms.arrays['forces'] = forces
        atoms.reference_status = True
    except:
        atoms.reference_status = False
    return atoms


@dataclass
class CP2KParameters:
    cp2k_input : str


@id_for_memo.register(CP2KParameters)
def id_for_memo_cp2k_parameters(parameters: CP2KParameters, output_ref=False):
    assert not output_ref
    b1 = id_for_memo(parameters.cp2k_input, output_ref=output_ref)
    return b1


class CP2KReference(BaseReference):
    """CP2K Reference

    Arguments
    ---------

    cp2k_input : str
        string representation of the cp2k input file.

    cp2k_data : dict
        dictionary with data required during the calculation. E.g. basis
        sets, pseudopotentials, ...
        They are written to the local execution directory in order to make
        them available to the cp2k executable.
        The keys of the dictionary correspond to the capitalized keys in
        the cp2k input (e.g. BASIS_SET_FILE_NAME)

    """
    execution_definition = [
            'executor',
            'device',
            'ncores',
            'mpi_command',
            'cp2k_exec',
            'time_per_singlepoint',
            ]
    parameters_cls = CP2KParameters
    required_files = [
            'basis_set',
            'potential',
            'dftd3',
            ]

    @classmethod
    def create_apps(cls, context):
        label  = context[cls]['executor']
        ncores = context[cls]['ncores']
        mpi_command = context[cls]['mpi_command']
        cp2k_exec = context[cls]['cp2k_exec']
        walltime = context[cls]['time_per_singlepoint']

        # parse full command
        command = ''
        if mpi_command is not None:
            command += mpi_command(ncores)
        command += ' '
        command += cp2k_exec

        singlepoint_pre = bash_app(
                cp2k_singlepoint_pre,
                executors=[label],
                cache=False,
                )
        singlepoint_post = python_app(
                cp2k_singlepoint_post,
                executors=[label],
                cache=False,
                )
        def singlepoint_wrapped(
                atoms,
                parameters,
                file_names,
                inputs=[],
                outputs=[],
                ):
            assert len(file_names) == len(inputs)
            for name in cls.required_files:
                assert name in file_names
            pre = singlepoint_pre(
                    atoms,
                    parameters,
                    command,
                    file_names,
                    walltime=walltime,
                    inputs=inputs, # tmp Files
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    )
            return singlepoint_post(
                    atoms=atoms,
                    inputs=[pre.stdout, pre.stderr, pre], # wait for bash app
                    )
        context.register_app(cls, 'evaluate_single', singlepoint_wrapped)
        super(CP2KReference, cls).create_apps(context)

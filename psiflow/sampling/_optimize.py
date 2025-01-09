"""
Structure optimisation through ASE
TODO: do we need to check for very large forces?
TODO: what units are pressure?
TODO: what to do when max_steps is reached before converging?
"""

import os
import json
import warnings
import signal
import argparse
from pathlib import Path
from types import SimpleNamespace

import ase
import ase.io
import numpy as np
from ase.io.extxyz import save_calc_results
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.mixing import LinearCombinationCalculator
from ase.optimize.precon import PreconLBFGS
from ase.filters import FrechetCellFilter
from psiflow.geometry import Geometry
from psiflow.functions import function_from_json, EnergyFunction


EVAL_COMMAND: str = __file__
ALLOWED_MODES: tuple[str, ...] = ('full', 'fix_volume', 'fix_shape', 'fix_cell')
FILE_OUT: str = 'out.xyz'
FILE_TRAJ: str = 'out.traj'


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class FunctionCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, function: EnergyFunction, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def calculate(
        self,
        atoms=None,
        properties=all_properties,
        system_changes=None,
    ):
        super().calculate(atoms, properties, system_changes)
        geometry = Geometry.from_atoms(self.atoms)
        outputs = self.function([geometry])
        self.results = {name: array[0] for name, array in outputs.items()}
        self.results['free_energy'] = self.results['energy']                # required by optimiser


def log_state(atoms: ase.Atoms) -> None:
    """"""
    def make_log(data: list[tuple[str]]):
        """"""
        txt = ['', 'Current atoms state:']
        txt += [f'{_[0]:<15}: {_[1]:<25}[{_[2]}]' for _ in data]
        txt += 'End', ''
        print(*txt, sep='\n')

    data = []
    if atoms.calc:
        energy, max_force = atoms.get_potential_energy(), np.linalg.norm(atoms.get_forces(), axis=0).max()
    else:
        energy, max_force = [np.nan] * 2
    data += ('Energy', f'{energy:.2f}', 'eV'), ('Max. force', f'{max_force:.2E}', 'eV/A')

    if not all(atoms.pbc):
        make_log(data)
        return

    volume, cell = atoms.get_volume(), atoms.get_cell().cellpar().round(3)
    data += ('Cell volume', f'{atoms.get_volume():.2f}', 'A^3'),
    data += ('Box norms', str(cell[:3])[1:-1], 'A'), ('Box angles', str(cell[3:])[1:-1], 'degrees')

    make_log(data)
    return


def get_dof_filter(atoms: ase.Atoms, mode: str, pressure: float) -> ase.Atoms | FrechetCellFilter:
    """"""
    if mode == 'fix_cell':
        if pressure:
            warnings.warn('Ignoring external pressure..')
        return atoms
    kwargs = {'mask': [True] * 6, 'scalar_pressure': pressure}      # enable cell DOFs
    if mode == 'fix_shape':
        kwargs['hydrostatic_strain'] = True
    if mode == 'fix_volume':
        kwargs['constant_volume'] = True
        if pressure:
            warnings.warn('Ignoring applied pressure during fixed volume optimisation..')
    return FrechetCellFilter(atoms, **kwargs)


def run(args: SimpleNamespace):
    """"""
    config = json.load(Path(args.input_config).open('r'))

    atoms = ase.io.read(args.start_xyz)
    if not any(atoms.pbc):
        atoms.center(vacuum=0)              # optimiser mysteriously requires a nonzero unit cell
        if args.mode != 'fix_cell':
            args.mode = 'fix_cell'
            warnings.warn('Molecular structure is not periodic. Ignoring cell..')

    # construct calculator by combining hamiltonians
    assert args.path_hamiltonian is not None
    print('Making calculator from:', *config['forces'], sep='\n')
    functions = [function_from_json(p) for p in args.path_hamiltonian]
    calc = LinearCombinationCalculator(
        [FunctionCalculator(f) for f in functions],
        [float(h['weight']) for h in config['forces']]
    )

    atoms.calc = calc
    dof = get_dof_filter(atoms, config['mode'], config['pressure'])
    opt = PreconLBFGS(dof, trajectory=FILE_TRAJ if config['keep_trajectory'] else None)

    print(f"pid: {os.getpid()}")
    print(f"CPU affinity: {os.sched_getaffinity(os.getpid())}")
    log_state(atoms)
    try:
        opt.run(fmax=config['f_max'], steps=config['max_steps'])
    except TimeoutException:
        print('OPTIMISATION TIMEOUT')
        # TODO: what to do here?
        return

    log_state(atoms)
    save_calc_results(atoms, calc_prefix='', remove_atoms_calc=True)
    if not any(atoms.pbc):
        atoms.cell = None               # remove meaningless cell
    ase.io.write(FILE_OUT, atoms)
    print('OPTIMISATION SUCCESSFUL')
    return


def clean(args: SimpleNamespace):
    """"""
    from psiflow.data.utils import _write_frames

    geometry = Geometry.load(FILE_OUT)
    _write_frames(geometry, outputs=[args.output_xyz])
    if Path(FILE_TRAJ).is_file():
        traj = [at for at in ase.io.trajectory.Trajectory(FILE_TRAJ)]
        geometries = [Geometry.from_atoms(at) for at in traj]
        _write_frames(*geometries, outputs=[args.output_traj])
    print('FILES MOVED')
    return


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, timeout_handler)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='what to do', dest='action')
    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "--path_hamiltonian",
        action='extend',
        nargs='*',
        type=str,
    )
    run_parser.add_argument(
        "--input_config",
        type=str,
        default=None,
    )
    run_parser.add_argument(
        "--start_xyz",
        type=str,
        default=None,
    )
    clean_parser = subparsers.add_parser("clean")
    clean_parser.set_defaults(func=clean)
    clean_parser.add_argument(
        "--output_xyz",
        type=str,
        default=None,
    )
    clean_parser.add_argument(
        "--output_traj",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.func(args)



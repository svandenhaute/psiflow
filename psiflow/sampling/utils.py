from typing import Any, Optional

import numpy as np
import typeguard
from ase.data import chemical_symbols

from psiflow.functions import Function

# do not use psiflow apps; parsl config is not loaded in this process!
from psiflow.geometry import Geometry

# only import stuff which does not issue useless warnings; otherwise
# python -c 'import .client; print(client.__file__)' is going to be polluted
# with those import-related warnings


class ForceMagnitudeException(Exception):
    pass


@typeguard.typechecked
def check_forces(
    forces: np.ndarray,
    geometry: Any,
    max_force: float,
):
    if not isinstance(geometry, Geometry):
        geometry = Geometry.from_atoms(geometry)

    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if np.sum(exceeded):
        indices = np.arange(len(geometry))[exceeded]
        numbers = geometry.per_atom.numbers[exceeded]
        symbols = [chemical_symbols[n] for n in numbers]
        raise ForceMagnitudeException(
            "\nforce exceeded {} eV/A for atoms {}"
            " with chemical elements {}\n".format(
                max_force,
                indices,
                symbols,
            )
        )
    else:
        pass


class FunctionDriver:

    def __init__(
        self,
        template: Geometry,
        function: Function,
        max_force: Optional[float],
        verbose: bool = True,  # used by i-PI internally?
        error_msg="",
    ):
        self.verbose = verbose
        self.template = template
        self.function = function
        self.max_force = max_force

    def check_arguments(self):
        pass

    def __call__(self, cell, pos):
        from ipi.utils.units import unit_to_internal, unit_to_user

        pos = unit_to_user("length", "angstrom", pos)
        cell = unit_to_user("length", "angstrom", cell.T)

        self.template.per_atom.positions[:] = pos
        if self.template.periodic:
            self.template.cell[:] = cell

        outputs = self.function(self.template)
        energy = outputs["energy"]
        forces = outputs["forces"]
        stress = outputs["stress"]

        # check for max_force
        if self.max_force is not None:
            check_forces(forces, self.template, self.max_force)

        # converts to internal quantities
        pot_ipi = np.asarray(
            unit_to_internal("energy", "electronvolt", energy), np.float64
        )
        force_ipi = np.asarray(unit_to_internal("force", "ev/ang", forces), np.float64)
        vir_calc = -stress * self.template.volume
        vir_ipi = np.array(
            unit_to_internal("energy", "electronvolt", vir_calc.T), dtype=np.float64
        )
        extras = ""

        return pot_ipi, force_ipi, vir_ipi, extras

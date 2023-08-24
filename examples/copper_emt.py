import requests
import logging
from pathlib import Path
import numpy as np

from ase.build import bulk, make_supercell

import psiflow
from psiflow.learning import SequentialLearning, load_learning
from psiflow.models import MACEModel, MACEConfig
from psiflow.reference import EMTReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.walkers import DynamicWalker, PlumedBias
from psiflow.state import load_state
from psiflow.metrics import Metrics


def main(path_output):
    assert not path_output.exists()
    reference = EMTReference()     # CP2K; PBE-D3(BJ); TZVP
    atoms     = make_supercell(bulk('Cu', 'fcc', a=3.6, cubic=True), 3 * np.eye(3))

    config = MACEConfig()
    config.r_max = 6.0
    config.hidden_irreps = '8x0e + 8x1o'
    config.batch_size = 4
    model = MACEModel(config)

    model.add_atomic_energy('Cu', reference.compute_atomic_energy('Cu', box_size=6))

    # set learning parameters and do pretraining
    learning = SequentialLearning(
            path_output=path_output,
            niterations=10,
            train_valid_split=0.9,
            train_from_scratch=True,
            metrics=Metrics('copper_EMT', 'examples', 'psiflow'),
            error_thresholds_for_reset=(10, 200), # in meV/atom, meV/angstrom
            initial_temperature=100,
            final_temperature=2000,
            )

    # construct walkers; biased MTD MD in this case
    walkers = DynamicWalker.multiply(
            30,
            data_start=Dataset([atoms]),
            timestep=0.5,
            steps=1000,
            step=50,
            start=0,
            temperature=100,
            temperature_reset_quantile=0.01, # reset if P(temp) < 0.01
            pressure=0,
            )
    data = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            )


if __name__ == '__main__':
    psiflow.load()
    path_output = Path.cwd() / 'output' # stores learning results
    main(path_output)

# top level imports should be lightweight!
import os


def main():
    import argparse
    import time
    from pathlib import Path

    from ase.io import read
    from ipi._driver.driver import run_driver

    from psiflow.functions import function_from_json
    from psiflow.geometry import Geometry
    from psiflow.sampling.utils import ForceMagnitudeException, FunctionDriver

    print("OS environment values:")
    for key, value in os.environ.items():
        print(key, value)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_hamiltonian",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_force",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    assert args.path_hamiltonian is not None
    assert args.address is not None
    assert args.start is not None

    print("pid: {}".format(os.getpid()))
    affinity = os.sched_getaffinity(os.getpid())
    print("CPU affinity before function init: {}".format(affinity))

    template = Geometry.from_atoms(read(args.start))
    function = function_from_json(
        args.path_hamiltonian,
        device=args.device,
        dtype=args.dtype,
    )

    driver = FunctionDriver(
        template=template,
        function=function,
        max_force=args.max_force,
        verbose=True,
    )

    affinity = os.sched_getaffinity(os.getpid())
    print("CPU affinity after function init: {}".format(affinity))

    try:
        t0 = time.time()
        for _ in range(10):
            function(template)  # torch warmp-up before simulation
        print("time for 10 evaluations: {}".format(time.time() - t0))
        run_driver(
            unix=True,
            address=str(Path.cwd() / args.address),
            driver=driver,
            sockets_prefix="",
        )
    except ForceMagnitudeException as e:
        print(e)  # induce timeout in server
    except ConnectionResetError as e:  # some other client induced a timeout
        print(e)

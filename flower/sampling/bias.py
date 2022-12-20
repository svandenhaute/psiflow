import os
import tempfile
import yaff
import molmod
import numpy as np
from collections import OrderedDict

from parsl.app.app import python_app, join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

from flower.execution import Container, ModelExecutionDefinition
from flower.utils import _new_file, copy_data_future


def try_manual_plumed_linking():
    if 'PLUMED_KERNEL' not in os.environ.keys():
        # try linking manually
        if 'CONDA_PREFIX' in os.environ.keys(): # for conda environments
            p = 'CONDA_PREFIX'
        elif 'PREFIX' in os.environ.keys(): # for pip environments
            p = 'PREFIX'
        else:
            print('failed to set plumed .so kernel')
            pass
        path = os.environ[p] + '/lib/libplumedKernel.so'
        if os.path.exists(path):
            os.environ['PLUMED_KERNEL'] = path
            print('plumed kernel manually set at at : {}'.format(path))


def set_path_in_plumed(plumed_input, keyword, path_to_set):
    lines = plumed_input.split('\n')
    for i, line in enumerate(lines):
        if keyword in line.split():
            line_before = line.split('FILE=')[0]
            line_after  = line.split('FILE=')[1].split()[1:]
            lines[i] = line_before + 'FILE={} '.format(path_to_set) + ' '.join(line_after)
    return '\n'.join(lines)


def parse_plumed_input(plumed_input):
    allowed_keywords = ['METAD', 'RESTRAINT', 'EXTERNAL', 'UPPER_WALLS']
    biases = []
    for key in allowed_keywords:
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if key in line.split():
                #assert not found
                cv = line.split('ARG=')[1].split()[0]
                #label = line.split('LABEL=')[1].split()[0]
                biases.append((key, cv))
    return biases


def generate_external_grid(bias_function, cv, cv_label, periodic=False):
    _periodic = 'false' if not periodic else 'true'
    grid = ''
    grid += '#! FIELDS {} external.bias der_{}\n'.format(cv_label, cv_label)
    grid += '#! SET min_{} {}\n'.format(cv_label, np.min(cv))
    grid += '#! SET max_{} {}\n'.format(cv_label, np.max(cv))
    grid += '#! SET nbins_{} {}\n'.format(cv_label, len(cv))
    grid += '#! SET periodic_{} {}\n'.format(cv_label, _periodic)
    for i in range(len(cv)):
        grid += '{} {} {}\n'.format(cv[i], bias_function(cv[i]), 0)
    return grid


def evaluate_bias(plumed_input, cv, inputs=[]):
    import tempfile
    import os
    import numpy as np
    import yaff
    yaff.log.set_level(yaff.log.silent)
    import molmod
    from flower.sampling.utils import ForcePartASE, create_forcefield, \
            ForceThresholdExceededException
    from flower.sampling.bias import try_manual_plumed_linking
    from flower.data import read_dataset
    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    values = np.zeros((len(dataset), 2)) # column 0 for CV, 1 for bias
    system = yaff.System(
            numbers=dataset[0].get_atomic_numbers(),
            pos=dataset[0].get_positions() * molmod.units.angstrom,
            rvecs=dataset[0].get_cell() * molmod.units.angstrom,
            )
    try_manual_plumed_linking()
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    colvar_log = tmp.name # dummy log file
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    plumed_log = tmp.name # dummy log file

    # prepare input; modify METAD pace if necessary
    lines = plumed_input.split('\n')
    for i, line in enumerate(lines):
        if 'METAD' in line.split():
            line_before = line.split('PACE=')[0]
            line_after  = line.split('PACE=')[1].split()[1:]
            pace = 2147483647 # some random high prime number
            lines[i] = line_before + 'PACE={} '.format(pace) + ' '.join(line_after)
    plumed_input = '\n'.join(lines)
    plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
    plumed_input += '\nPRINT STRIDE=1 ARG={} FILE={}'.format(cv, colvar_log)
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write(plumed_input) # write input
        path_input = f.name
    part_plumed = yaff.external.ForcePartPlumed(
            system,
            timestep=1*molmod.units.femtosecond, # does not matter
            restart=1,
            fn=path_input,
            fn_log=plumed_log,
            )
    ff = yaff.pes.ForceField(system, [part_plumed])
    for i, atoms in enumerate(dataset):
        ff.update_pos(atoms.get_positions() * molmod.units.angstrom)
        ff.update_rvecs(atoms.get_cell() * molmod.units.angstrom)
        values[i, 1] = ff.compute() / molmod.units.kjmol
        part_plumed.plumed.cmd('update')
        part_plumed.plumedstep = 3 # can be anything except zero; pick a prime
    part_plumed.plumed.cmd('update') # flush last
    values[:, 0] = np.loadtxt(colvar_log)[:, 1]
    os.unlink(plumed_log)
    os.unlink(colvar_log)
    os.unlink(path_input)
    return values


class PlumedBias(Container):
    """Represents a PLUMED bias potential"""

    def __init__(self, context, plumed_input, data={}):
        super().__init__(context)
        assert 'PRINT' not in plumed_input
        components = parse_plumed_input(plumed_input)
        assert len(components) > 0
        for c in components:
            assert ',' not in c[1] # require 1D bias
        assert len(set([c[1] for c in components])) == 1 # single CV
        self.components   = components
        self.cv           = components[0][1]
        self.plumed_input = plumed_input

        # initialize data future for each component
        self.data_futures = OrderedDict()
        for key, value in data.items():
            assert key in self.keys
            if type(value) == str:
                path_new = _new_file(context.path, key + '_', '.txt')
                with open(path_new, 'w') as f:
                    f.write(value)
                self.data_futures[key] = File(path_new)
            else:
                assert (isinstance(value, DataFuture) or isinstance(value, File))
                self.data_futures[key] = value
        for key in self.keys:
            if key not in self.data_futures.keys():
                assert key != 'EXTERNAL' # has to be initialized by user
                self.data_futures[key] = File(_new_file(context.path, key + '_', '.txt'))
        if 'METAD' in self.keys:
            self.data_futures.move_to_end('METAD', last=False)


    def evaluate(self, dataset):
        plumed_input = self.prepare_input()
        return self.context.apps(PlumedBias, 'evaluate')(
                plumed_input,
                self.cv,
                inputs=[dataset.data_future] + self.futures,
                )

    def prepare_input(self):
        plumed_input = str(self.plumed_input)
        for key in self.keys:
            if key in ['METAD', 'EXTERNAL']: # keys for which path needs to be set
                plumed_input = set_path_in_plumed(
                        plumed_input,
                        key,
                        self.data_futures[key].filepath,
                        )
        if 'METAD' in self.keys: # necessary to print hills properly
            plumed_input = 'RESTART\n' + plumed_input
            plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
        return plumed_input

    def copy(self):
        new_futures = OrderedDict()
        for key, future in self.data_futures.items():
            new_futures[key] = copy_data_future(
                    inputs=[future],
                    outputs=[File(_new_file(self.context.path, 'bias_', '.txt'))],
                    ).outputs[0]
        return PlumedBias(
                self.context,
                self.plumed_input,
                data_futures=new_futures,
                )

    @property
    def keys(self):
        keys = sorted([c[0] for c in self.components])
        assert len(set(keys)) == len(keys) # keys should be unique!
        if 'METAD' in keys:
            keys.remove('METAD')
            return ['METAD'] + keys
        else:
            return keys

    @property
    def futures(self):
        return [value for _, value in self.data_futures.items()] # MTD first

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ModelExecutionDefinition].executor_label
        app_evaluate = python_app(evaluate_bias, executors=[executor_label])
        context.register_app(cls, 'evaluate', app_evaluate)


#class MetadynamicsBias(PlumedBias):
#
#    def __init__(self, context, plumed_input, data_futures=None):
#        super().__init__(context, plumed_input, data_futures)
#        assert self.keyword == 'METAD'
#
#    def prepare_input(self):
#        plumed_input = str(self.plumed_input)
#        plumed_input = set_path_in_plumed(plumed_input, 'METAD', self.data_futures[0].filepath)
#        plumed_input = 'RESTART\n' + plumed_input
#        plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
#        return plumed_input
#
#
#class ExternalBias(PlumedBias):
#
#    def __init__(self, context, plumed_input, data_futures=None):
#        super().__init__(context, plumed_input, data_futures)
#        assert self.keyword == 'EXTERNAL'
#
#    def prepare_input(self):
#        plumed_input = str(self.plumed_input)
#        plumed_input = set_path_in_plumed(plumed_input, 'EXTERNAL', self.data_futures[0].filepath)
#        return plumed_input
#
#
#class AggregateBias(PlumedBias):
#
#    def __init__(self, context, plumed_input, data_futures):
#        #assert bias0.cv == bias1.cv
#        #assert bias0.keyword != bias1.keyword
#        #assert len(bias0.data_futures) == 1
#        #assert len(bias1.data_futures) == 1
#        #self.context = bias0.context
#        #self.bias0 = bias0.copy()
#        #self.bias1 = bias1.copy()
#        #self.data_futures = [
#        #        self.bias0.data_futures[0],
#        #        self.bias1.data_futures[0],
#        #        ]
#
#    def prepare_input(self):
#        #plumed_input = bias0.prepare_input() # base input file
#        #lines = bias1.prepare_input().split('\n')
#        #found = False
#        #for i, line in enumerate(lines):
#        #    if bias1.keyword in line.split():
#        #        found = True
#        #        plumed_input += '\n' + line
#        #assert found
#        #return plumed_input


#def create_bias(context, plumed_input, path_data=None, data=None):
#    keyword, cv = parse_plumed_input(plumed_input)
#    if isinstance(path_data, str):
#        assert data is None
#        assert os.path.exists(path_data)
#        data_future = File(path_data) # convert to File before passing it as future
#    elif isinstance(data, str):
#        assert path_data is None
#        path_data = _new_file(context.path, 'bias_', '.txt')
#        with open(path_data, 'w') as f:
#            f.write(data)
#        data_future = File(path_data)
#    else:
#        data_future = None
#    if (keyword == 'RESTRAINT' or keyword == 'UPPER_WALLS'):
#        return PlumedBias(context, plumed_input, data_futures=[])
#    elif keyword == 'METAD':
#        return MetadynamicsBias(context, plumed_input, data_futures=[data_future])
#    elif keyword == 'EXTERNAL':
#        return ExternalBias(context, plumed_input, data_futures=[data_future])
#    else:
#        raise ValueError('plumed keyword {} unrecognized'.format(keyword))
# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd;
import hoomd.md;
from hoomd import deprecated
hoomd.context.initialize()
import unittest
import os
import tempfile
import numpy

# unit tests for analyze.log
class analyze_log_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.05);

        hoomd.context.current.sorter.set_params(grid=8)

        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the analyzer
    def test(self):
        hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        hoomd.run(100);

    # tests with phase
    def test_phase(self):
        hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file, phase=0);
        hoomd.run(100);

    # test set_params
    def test_set_params(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.set_params(quantities = ['test1']);
        hoomd.run(100);
        ana.set_params(delimiter = ' ');
        hoomd.run(100);
        ana.set_params(quantities = ['test2', 'test3'], delimiter=',')
        hoomd.run(100);
        ana.set_params(quantities = [u'test4', u'test5'], delimiter=',')
        hoomd.run(100);

    # test variable period
    def test_variable(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = lambda n: n*10, filename=self.tmp_file);
        hoomd.run(100);

    # test the initialization checks
    def test_init_checks(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.cpp_analyzer = None;

        self.assertRaises(RuntimeError, ana.enable);
        self.assertRaises(RuntimeError, ana.disable);

    def tearDown(self):
        hoomd.context.initialize();
        if (hoomd.comm.get_rank()==0):
            os.remove(self.tmp_file);


# test analyze.log with query
class analyze_log_query_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.005);
        nl = hoomd.md.nlist.cell()
        self.pair = hoomd.md.pair.lj(r_cut=2.5, nlist = nl)
        self.pair.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005);
        hoomd.md.integrate.langevin(hoomd.group.all(), seed=1, kT=1.0);

        hoomd.context.current.sorter.set_params(grid=8)

    # tests query with no output file
    def test(self):
        log = hoomd.analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=None);
        hoomd.run(102);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        hoomd.run(2);
        t1 = log.query('timestep');
        U1 = log.query('potential_energy');
        K1 = log.query('kinetic_energy');

        self.assertEqual(int(t0), 102)
        self.assertNotEqual(K0, 0);
        self.assertNotEqual(U0, 0);
        self.assertEqual(int(t1), 104)
        self.assertNotEqual(U0, U1);
        self.assertNotEqual(K0, K1);

    # tests basic creation of the analyzer
    def test_with_file(self):
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        log = hoomd.analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=self.tmp_file);
        hoomd.run(11);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        hoomd.run(2);
        t1 = log.query('timestep');
        U1 = log.query('potential_energy');
        K1 = log.query('kinetic_energy');

        self.assertEqual(int(t0), 10)
        self.assertNotEqual(K0, 0);
        self.assertNotEqual(U0, 0);
        self.assertEqual(int(t1), 10)
        self.assertEqual(U0, U1);
        self.assertEqual(K0, K1);

    def tearDown(self):
        self.pair = None;
        hoomd.context.initialize();


# test hdf5.log with query
class analyze_log_hdf5_query_tests (unittest.TestCase):
    def setUp(self):
        print
        self.enable_hdf5 = False
        try:
            import hoomd.hdf5
            self.enable_hdf5 = True
        except:
            print("Warning no h5py available, so hdf5 is classes are unchecked.")
            return
        deprecated.init.create_random(N=100, phi_p=0.005);
        nl = hoomd.md.nlist.cell()
        self.pair = hoomd.md.pair.lj(r_cut=2.5, nlist = nl)
        self.pair.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005);
        hoomd.md.integrate.langevin(hoomd.group.all(), seed=1, kT=1.0);

        hoomd.context.current.sorter.set_params(grid=8)

    # tests basic creation of the analyzer
    def test_with_file(self):
        if not self.enable_hdf5:
            return

        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.h5');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        log = hoomd.hdf5.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=self.tmp_file);
        hoomd.run(11);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        hoomd.run(2);
        t1 = log.query('timestep');
        U1 = log.query('potential_energy');
        K1 = log.query('kinetic_energy');

        self.assertEqual(int(t0), 10)
        self.assertNotEqual(K0, 0);
        self.assertNotEqual(U0, 0);
        self.assertEqual(int(t1), 10)
        self.assertEqual(U0, U1);
        self.assertEqual(K0, K1);

    def tearDown(self):
        if not self.enable_hdf5:
            return

        self.pair = None;
        hoomd.context.initialize();
        if (hoomd.comm.get_rank()==0):
            os.remove(self.tmp_file);

# unit tests for analyze.log_hdf5
class analyze_log_hdf5_tests (unittest.TestCase):
    def setUp(self):
        print
        self.enable_hdf5 = False
        try:
            import hoomd.hdf5
            self.enable_hdf5 = True
        except:
            print("Warning no h5py available, so hdf5 is classes are unchecked.")
            return


        deprecated.init.create_random(N=100, phi_p=0.05);

        hoomd.context.current.sorter.set_params(grid=8)

        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the analyzer
    def test(self):
        if not self.enable_hdf5:
            return

        hoomd.hdf5.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        hoomd.run(100);

    def test_matrix(self):
        if not self.enable_hdf5:
            return

        def callback(timestep):
            return numpy.random.rand(2,3)
        ana = hoomd.hdf5.log(quantities = ['test1', 'test2', 'test3'],matrix_quantities=["mtest1","mtest2"], period = 10, filename=self.tmp_file);
        ana.register_callback("mtest1",callback,matrix=True)
        ana.register_callback("mtest2",callback,matrix=True)

        hoomd.run(100);

    # tests with phase
    def test_phase(self):
        if not self.enable_hdf5:
            return

        def callback(timestep):
            return numpy.random.rand(2,3)
        ana = hoomd.hdf5.log(quantities = ['test1', 'test2', 'test3'],matrix_quantities=["mtest1","mtest2"], period = 10, filename=self.tmp_file,phase=0);
        ana.register_callback("mtest1",callback,matrix=True)
        ana.register_callback("mtest2",callback,matrix=True)

        hoomd.run(100);

    # test set_params
    def test_set_params(self):
        if not self.enable_hdf5:
            return

        def callback(timestep):
            return numpy.random.rand(2,3)

        ana = hoomd.hdf5.log(quantities = ['test1', 'test2', 'test3'],matrix_quantities=["mtest1","mtest2"], period = 10, filename=self.tmp_file);
        ana.register_callback("mtest1",callback,matrix=True)
        ana.register_callback("mtest2",callback,matrix=True)

        #hdf5 logger does not support changing the number of logged quantities on the fly.
        # ana.set_params(quantities = ['test1']);
        # hoomd.run(100);
        # ana.set_params(quantities = ['test2', 'test3'])
        # hoomd.run(100);
        # ana.set_params(quantities = [u'test4', u'test5'])
        # hoomd.run(100);
        # ana.set_params(matrix_quantities = [])
        # hoomd.run(100);
        ana.set_params(matrix_quantities = ["mtest1"])
        hoomd.run(100);
        ana.set_params(matrix_quantities = ["mtest1"])
        hoomd.run(100);

    # test the initialization checks
    def test_init_checks(self):
        if not self.enable_hdf5:
            return

        ana = hoomd.hdf5.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.cpp_analyzer = None;

        self.assertRaises(RuntimeError, ana.enable);
        self.assertRaises(RuntimeError, ana.disable);

    def tearDown(self):
        if not self.enable_hdf5:
            return

        hoomd.context.initialize();
        if (hoomd.comm.get_rank()==0):
            os.remove(self.tmp_file);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

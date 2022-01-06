
import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd sphere streaming geometry
class mpcd_stream_sphere_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=2)
        snap.particles.position[:] = [[3.85,0.,0.],[0.,0.,0.]]
        snap.particles.velocity[:] = [[1.,0.,0.],[-1.,-1.,-1.]]
        self.s = mpcd.init.read_snapshot(snap)

        mpcd.integrator(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.stream.sphere(R=4., boundary="no_slip", period=2)

    # test for setting parameters
    def test_set_params(self):
        sphere = mpcd.stream.sphere(R=4.)
        self.assertAlmostEqual(sphere.R, 4.)
        self.assertEqual(sphere.boundary, "no_slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 4.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change R and also ensure other parameters stay the same
        sphere.set_params(R=2.)
        self.assertAlmostEqual(sphere.R, 2.)
        self.assertEqual(sphere.boundary, "no_slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 2.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change BCs
        sphere.set_params(boundary="slip")
        self.assertAlmostEqual(sphere.R, 2.)
        self.assertEqual(sphere.boundary, "slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 2.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        sphere = mpcd.stream.sphere(R=4.)
        sphere.set_params(boundary="no_slip")
        sphere.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            sphere.set_params(boundary="invalid")

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
        mpcd.stream.sphere(R=4.)

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.95,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-0.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step where one particle will now reflect from the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.95,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-0.2])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step where both particles are streaming only
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.85,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.3,-0.3,-0.3])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

    # test basic stepping behaviour with slip boundary conditions
    def test_step_slip(self):
        mpcd.stream.sphere(R=4., boundary="slip")

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.95, 0., 0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1., 0., 0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-0.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1., -1., -1.])

        # take another step where one particle will now hit the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.95, 0., 0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1., 0., 0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-0.2])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1., -1., -1.])

        # take another step where both particles are streaming only
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [3.85,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,0.,0.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.3,-0.3,-0.3])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

    # test that setting the sphere size too large raises an error
    def test_validate_box(self):
        # initial configuration is invalid
        sphere = mpcd.stream.sphere(R=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        # now it should be valid
        sphere.set_params(R=4.)
        hoomd.run(2)

        # make sure we can invalidate it again
        sphere.set_params(R=4.1)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
        sphere = mpcd.stream.sphere(R=3.8)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        sphere.set_params(R=3.95)
        hoomd.run(1)

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

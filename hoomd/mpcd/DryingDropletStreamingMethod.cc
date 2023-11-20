// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreamingMethod.cc
 * \brief Definition of mpcd::DryingDropletStreamingMethod
 */
#include "DryingDropletStreamingMethod.h"
//! Constructor
/*!
 * \param sysdata MPCD system data
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param R is the radius of sphere
 * \param bc is boundary conditions(slip or no-slip)
 * \param density is solvent number density inside the droplet
 * \param seed is Seed to random number Generator
 */
mpcd::DryingDropletStreamingMethod::DryingDropletStreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                                                 unsigned int cur_timestep,
                                                                 unsigned int period,
                                                                 int phase,
                                                                 std::shared_ptr<::Variant> R,
                                                                 mpcd::detail::boundary bc,
                                                                 Scalar density,
                                                                 unsigned int seed)
    : mpcd::ConfinedStreamingMethod<mpcd::detail::SphereGeometry>(sysdata, cur_timestep, period, phase, std::shared_ptr<mpcd::detail::SphereGeometry>()),
      m_R(R), m_bc(bc), m_density(density), m_seed(seed), m_picks(m_exec_conf), m_picker(m_sysdef, seed)
    {}

/*!
 * \param timestep Current time to stream
 */
void mpcd::DryingDropletStreamingMethod::stream(unsigned int timestep)
    {
    // use peekStream since shouldStream will be called by parent class
    if(!peekStream(timestep)) return;

    // compute final Radius and Velocity of surface
    const Scalar start_R = m_R->getValue(timestep);
    const Scalar end_R = m_R->getValue(timestep + m_period);
    const Scalar V = (end_R - start_R)/(m_mpcd_dt);
    // make sure that V < 0, since size of droplet must decrease
    if (V > 0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }

    /*
     * If the initial geometry was not set. Set the geometry and validate the geometry
     * Because the interface is shrinking, it is sufficient to validate only the first time the geometry
     * is set.
     */
    if (!m_geom)
        {
        m_geom = std::make_shared<mpcd::detail::SphereGeometry>(start_R, V , m_bc );
        validate();
        m_validate_geom = false;
        }

    /*
     * Update the geometry radius to the size at the end of the streaming step.
     * This needs to be done every time.
     */
    m_geom = std::make_shared<mpcd::detail::SphereGeometry>(end_R, V, m_bc );

    // stream according to base class rules
    ConfinedStreamingMethod<mpcd::detail::SphereGeometry>::stream(timestep);

    // calculating number of particles to evaporate(such that solvent density remain constant)
    const unsigned int N_end = std::round((4.*M_PI/3.)*end_R*end_R*end_R*m_density);
    const unsigned int N_global = m_mpcd_pdata->getNGlobal();
    const unsigned int N_evap = (N_end < N_global) ? N_global - N_end : 0;

    /*
     * Picking N_evap particles out of total number of bounced particles using RandomPicker,
     * m_Npick is the total number of particles picked on this rank, m_picks will contain the indices of
     * picked particles in \a m_bounced array.
     */
    m_Npick = 0;
    m_picker(m_picks, m_Npick, m_bounced, N_evap, timestep, m_mpcd_pdata->getNGlobal());

    /*
     * applying the picks, In m_bounced array, the particles which were picked are marked 
     * by setting an additional bit.
     */
    applyPicks();

    // finally removing the picked particles
    m_mpcd_pdata->removeParticles(m_removed,
                                  m_bounced,
                                  m_mask,
                                  timestep);

    // calculating density after removing particles if it's changed alot, print a warning
    Scalar V_end = (4.*M_PI/3.)*end_R*end_R*end_R;
    Scalar currentdensity = this->m_mpcd_pdata->getNGlobal()/V_end;
    if (std::fabs(currentdensity - m_density) > Scalar(0.1))
        {
        this->m_exec_conf->msg->warning() << "Solvent density changed to: " << currentdensity << std::endl;
        }
    }

void mpcd::DryingDropletStreamingMethod::applyPicks()
    {
    /*
     * In m_bounced array, the particles which were picked are marked 
     * by setting an additional bit (e.g., 3 (11) is stored in m_bounced),
     * m_picks has indices of picked particles in a \m_bounced array
     * m_Npick is number of particles picked
     */
    ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_bounced(m_bounced, access_location::host, access_mode::readwrite);
    for (unsigned int i=0; i < m_Npick; ++i)
        {
        h_bounced.data[h_picks.data[i]] |= m_mask;
        }
    }

void mpcd::detail::export_DryingDropletStreamingMethod(pybind11::module& m)
    {
    //! Export mpcd::DryingDropletStreamingMethod to python
    /*!
     * \param m Python module to export to
     */
    namespace py = pybind11;
    py::class_<mpcd::DryingDropletStreamingMethod, mpcd::ConfinedStreamingMethod<mpcd::detail::SphereGeometry>, std::shared_ptr<mpcd::DryingDropletStreamingMethod>>(m, "DryingDropletStreamingMethod")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, boundary, Scalar, unsigned int>());
    }
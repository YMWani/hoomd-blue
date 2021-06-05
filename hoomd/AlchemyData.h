// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

/*! \file AlchemyData.h
    \brief Contains declarations for AlchemyData.
 */

#ifndef __ALCHEMYDATA_H__
#define __ALCHEMYDATA_H__

#include "hoomd/ExecutionConfiguration.h"
#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <string>

#include "HOOMDMPI.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"

class AlchemicalParticle
    {
    public:
    AlchemicalParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : value(Scalar(1.0)), m_exec_conf(exec_conf) {};

    Scalar value; //!< Alpha space dimensionless position of the particle
    uint64_t m_nextTimestep;

    protected:
    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf;                 //!< Stored shared ptr to the execution configuration
    std::shared_ptr<Compute> m_base; //!< the associated Alchemical Compute
    };
class AlchemicalMDParticle : public AlchemicalParticle
    {
    public:
    AlchemicalMDParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : AlchemicalParticle(exec_conf) {};

    void inline zeroForces()
        {
        ArrayHandle<Scalar> h_forces(m_alchemical_derivatives,
                                     access_location::host,
                                     access_mode::overwrite);
        memset((void*)h_forces.data, 0, sizeof(Scalar) * m_alchemical_derivatives.getNumElements());
        }

    void resizeForces(unsigned int N)
        {
        GlobalArray<Scalar> new_forces(N, m_exec_conf);
        m_alchemical_derivatives.swap(new_forces);
        }

    void setNetForce(uint64_t timestep)
        {
        // TODO: remove this sanity check after we're done making sure timing works
        zeroForces();
        m_timestepNetForce.first = timestep;
        }

    void setNetForce()
        {
        Scalar netForce(0.0);
        ArrayHandle<Scalar> h_forces(m_alchemical_derivatives,
                                     access_location::host,
                                     access_mode::read);
        for (unsigned int i = 0; i < m_alchemical_derivatives.getNumElements(); i++)
            netForce += h_forces.data[i];
        m_exec_conf->msg->notice(10) << "alchForceSum:" << std::to_string(netForce) << std::endl;
        netForce /= Scalar(m_alchemical_derivatives.getNumElements());
        m_timestepNetForce.second = netForce;
        }

    Scalar getNetForce(uint64_t timestep)
        {
        // TODO: remove this sanity check after we're done making sure timing works
        assert(m_timestepNetForce.first == timestep);
        return m_timestepNetForce.second;
        }

    void setMass(Scalar new_mass)
        {
        mass.x = new_mass;
        mass.y = Scalar(1.) / new_mass;
        }

    Scalar getMass()
        {
        return mass.x;
        }

    Scalar getValue()
        {
        return value;
        }

    Scalar momentum = 0.; // the momentum of the particle
    Scalar2 mass;         // mass (x) and it's inverse (y) (don't have to recompute constantly)
    Scalar mu = 0.;       //!< the alchemical potential of the particle
    GlobalArray<Scalar> m_alchemical_derivatives; //!< Per particle alchemical forces
    protected:
    // the timestep the net force was computed and the netforce
    std::pair<uint64_t, Scalar> m_timestepNetForce;
    };

class AlchemicalPairParticle : public AlchemicalMDParticle
    {
    public:
    AlchemicalPairParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                           int3 type_pair_param)
        : AlchemicalMDParticle(exec_conf), m_type_pair_param(type_pair_param) {};
    int3 m_type_pair_param;
    };

inline void export_AlchemicalMDParticle(pybind11::module& m)
    {
    pybind11::class_<AlchemicalMDParticle, std::shared_ptr<AlchemicalMDParticle>>(
        m,
        "AlchemicalMDParticle")
        .def("setMass", &AlchemicalMDParticle::setMass)
        .def_property_readonly("getMass", &AlchemicalMDParticle::getMass)
        .def_property_readonly("alpha", &AlchemicalMDParticle::getValue);
    }

inline void export_AlchemicalPairParticle(pybind11::module& m)
    {
    pybind11::class_<AlchemicalPairParticle,
                     AlchemicalMDParticle,
                     std::shared_ptr<AlchemicalPairParticle>>(m, "AlchemicalPairParticle")
        // .def("setMass", &AlchemicalPairParticle::setMass)
        // .def_property_readonly("getMass", &AlchemicalPairParticle::getMass)
        // .def_property_readonly("alpha", &AlchemicalPairParticle::getValue)
        ;
    }

#endif
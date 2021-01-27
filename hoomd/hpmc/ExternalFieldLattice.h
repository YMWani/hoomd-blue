// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_FIELD_LATTICE_H_
#define _EXTERNAL_FIELD_LATTICE_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/

#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/AABBTree.h"

#include "ExternalField.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{
/*
For simplicity and consistency both the positional and orientational versions of
the external field will take in a list of either positions or orientations that
are the reference values. the i-th reference point will correspond to the particle
with tag i.
*/
inline void python_list_to_vector_scalar3(const pybind11::list& r0, std::vector<Scalar3>& ret, unsigned int ndim)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    for ( pybind11::ssize_t i=0; i<n; i++)
        {
        pybind11::ssize_t d = pybind11::len(r0[i]);
        pybind11::list r0_tuple = pybind11::cast<pybind11::list >(r0[i]);
        if( d < ndim )
            {
            throw std::runtime_error("dimension of the list does not match the dimension of the simulation.");
            }
        Scalar x = pybind11::cast<Scalar>(r0_tuple[0]), y = pybind11::cast<Scalar>(r0_tuple[1]), z = 0.0;
        if(d == 3)
            {
            z = pybind11::cast<Scalar>(r0_tuple[2]);
            }
        ret[i] = make_scalar3(x, y, z);
        }
    }

inline void python_list_to_vector_scalar4(const pybind11::list& r0, std::vector<Scalar4>& ret)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    for ( pybind11::ssize_t i=0; i<n; i++)
        {
        pybind11::list r0_tuple = pybind11::cast<pybind11::list >(r0[i]);

        ret[i] = make_scalar4(  pybind11::cast<Scalar>(r0_tuple[0]),
                                pybind11::cast<Scalar>(r0_tuple[1]),
                                pybind11::cast<Scalar>(r0_tuple[2]),
                                pybind11::cast<Scalar>(r0_tuple[3]));
        }
    }

inline void python_list_to_vector_int(const pybind11::list& r0, std::vector<unsigned int>& ret)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    int zahl = 0;
    for ( pybind11::ssize_t i=0; i<n; i++)
        {
        ret[i] = zahl;
        zahl++;
        }
    }

inline void python_list_to_vector_bool(const pybind11::list& r0, std::vector<bool>& ret, unsigned int N)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    for ( pybind11::ssize_t i=0; i<N; i++)
        {
        ret[i] = true;
        }
    for ( pybind11::ssize_t i=N; i<n; i++)
        {
        ret[i] = false;
        }
    }


template< class ScalarType >
class LatticeReferenceList
    {
    public:
        LatticeReferenceList() : m_N(0) {}

        template<class InputIterator >
        LatticeReferenceList(InputIterator first, InputIterator last, const std::shared_ptr<ParticleData> pdata, std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            initialize(first, last, pdata, exec_conf);
            }

        ~LatticeReferenceList() {
            }

        template <class InputIterator>
        void initialize(InputIterator first, InputIterator last, const std::shared_ptr<ParticleData> pdata, std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            m_N = std::distance(first, last);
            if( m_N > 0 )
                {
                setReferences(first, last, pdata, exec_conf);
                }
            }

        const ScalarType& getReference( const unsigned int& tag ) { ArrayHandle<ScalarType> h_ref(m_reference, access_location::host, access_mode::read); return h_ref.data[tag]; }

        const GPUArray< ScalarType >& getReferenceArray() { return m_reference; }

        const unsigned int getSize() { return m_N; }

        template <class InputIterator>
        void setReferences(InputIterator first, InputIterator last, const std::shared_ptr<ParticleData> pdata, std::shared_ptr<const ExecutionConfiguration> exec_conf)
        {
            size_t numPoints = std::distance(first, last);
            if(!numPoints)
                {
                clear();
                return;
                }

            if(!exec_conf || !pdata )//|| pdata->getNGlobal() != numPoints)
                {
                if(exec_conf) exec_conf->msg->error() << "Check pointers and initialization list" << std::endl;
                throw std::runtime_error("Error setting LatticeReferenceList");
                }
            m_N = numPoints;
            GPUArray<ScalarType> temp(numPoints, exec_conf);
            { // scope the copy.
            ArrayHandle<ScalarType> h_temp(temp, access_location::host, access_mode::overwrite);
            // now copy and swap the data.
            std::copy(first, last, h_temp.data);
            }
            m_reference.swap(temp);
        }

        void setReference(unsigned int i, ScalarType a)
            {
            ArrayHandle<ScalarType> h_ref(m_reference, access_location::host, access_mode::readwrite);
            h_ref.data[i] = a;
            }

        void scale(const Scalar& s)
            {
            ArrayHandle<ScalarType> h_ref(m_reference, access_location::host, access_mode::readwrite);
            for(unsigned int i = 0; i < m_N; i++)
                {
                h_ref.data[i].x *= s;
                h_ref.data[i].y *= s;
                h_ref.data[i].z *= s;
                }
            }

        void clear()
            {
            m_N = 0;
            GPUArray<ScalarType> nullArray;
            m_reference.swap(nullArray);
            }

        bool isValid() { return m_N != 0 && !m_reference.isNull(); }

    private:
        GPUArray<ScalarType> m_reference;
        unsigned int         m_N;
    };
    

#define LATTICE_ENERGY_LOG_NAME                 "lattice_energy"
#define LATTICE_ENERGY_TRANS_LOG_NAME           "lattice_energy_trans"
#define LATTICE_ENERGY_ROT_LOG_NAME             "lattice_energy_rot"
#define LATTICE_ENERGY_AVG_LOG_NAME             "lattice_energy_pp_avg"
#define LATTICE_ENERGY_SIGMA_LOG_NAME           "lattice_energy_pp_sigma"
#define LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME  "lattice_translational_spring_constant"
#define LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME  "lattice_rotational_spring_constant"
#define LATTICE_NUM_SAMPLES_LOG_NAME            "lattice_num_samples"

template< class Shape>
class ExternalFieldLatticeHypersphere : public ExternalFieldMono<Shape>
    {
    using ExternalFieldMono<Shape>::m_pdata;
    using ExternalFieldMono<Shape>::m_exec_conf;
    using ExternalFieldMono<Shape>::m_sysdef;
    public:
        ExternalFieldLatticeHypersphere(  std::shared_ptr<SystemDefinition> sysdef,
                                        pybind11::list quat_l,
                                        Scalar k,
                                        pybind11::list quat_r,
                                        Scalar q,
                                        pybind11::list symRotations
                                    ) : ExternalFieldMono<Shape>(sysdef), m_k(k), m_q(q), m_Energy(0)
            {
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_TRANS_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_ROT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_AVG_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_SIGMA_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_NUM_SAMPLES_LOG_NAME);
            m_aabbs=NULL;
            // Connect to the BoxChange signal
            m_hypersphere = m_pdata->getHypersphere();
            m_pdata->getBoxChangeSignal().template connect<ExternalFieldLatticeHypersphere<Shape>, &ExternalFieldLatticeHypersphere<Shape>::scaleReferencePoints>(this);
            setReferences(quat_l, quat_r, m_pdata->getN());
            setLatticeDist();
            buildAABBTree();

            std::vector<Scalar4> rots;
            python_list_to_vector_scalar4(symRotations, rots);
            bool identityFound = false;
            quat<Scalar> identity(1, vec3<Scalar>(0, 0, 0));
            Scalar tol = 1e-5;
            for(size_t i = 0; i < rots.size(); i++)
                {
                quat<Scalar> qi(rots[i]);
                identityFound = !identityFound ? norm2(qi-identity) < tol : identityFound;
                m_symmetry.push_back(qi);
                }
            if(!identityFound) // ensure that the identity rotation is provided.
                {
                m_symmetry.push_back(identity);
                }
            reset(0); // initializes all of the energy logging parameters.
            }

        ~ExternalFieldLatticeHypersphere()
        {
            if (m_aabbs != NULL)
                free(m_aabbs);
            m_pdata->getBoxChangeSignal().template disconnect<ExternalFieldLatticeHypersphere<Shape>, &ExternalFieldLatticeHypersphere<Shape>::scaleReferencePoints>(this);
        }

        //! Build the AABB tree (if needed)
        const detail::AABBTree& buildAABBTree()
            {
                // grow the AABB list to the needed size
                unsigned int n_aabb = m_latticeQuat_l.getSize();
                if (n_aabb > 0)
                    {
                    growAABBList(n_aabb);
        
                    const Hypersphere& hypersphere = m_pdata->getHypersphere();
                    for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
                        {
                        unsigned int i = cur_particle;
        
                        Scalar radius = m_refdist;
                        m_aabbs[i] = detail::AABB(hypersphere.hypersphericalToCartesian(quat<Scalar>(m_latticeQuat_l.getReference(i)), quat<Scalar>(m_latticeQuat_r.getReference(i))), radius);
                        }
        
                    // build the tree
                    m_aabb_tree.buildTree(m_aabbs, n_aabb);
                    }
        
                if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

            return m_aabb_tree;
            }

        Scalar calculateBoltzmannWeight(unsigned int timestep) { return 0.0; }

        double calculateDeltaEHypersphere(const Scalar4 * const quat_l_old_arg,
                                        const Scalar4 * const quat_r_old_arg,
                                        const Hypersphere * const hypersphere_old_arg
                                        )
            {
            // TODO: rethink the formatting a bit.
            ArrayHandle<Scalar4> h_quat_l(m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_quat_r(m_pdata->getRightQuaternionArray(), access_location::host, access_mode::readwrite);
            const Scalar4 * const quat_l_new = h_quat_l.data;
            const Scalar4 * const quat_r_new = h_quat_r.data;
            const Hypersphere * const hypersphere_new = &m_pdata->getHypersphere();
            const Scalar4 * quat_l_old=quat_l_old_arg, * quat_r_old=quat_r_old_arg;
            const Hypersphere * hypersphere_old = hypersphere_old_arg;
            if( !quat_l_old )
                quat_l_old = quat_l_new;
            if( !quat_r_old )
                quat_r_old = quat_r_new;
            if( !hypersphere_old )
                hypersphere_old = hypersphere_new;

            Scalar curVolume = m_hypersphere.getVolume();
            Scalar newVolume = hypersphere_new->getVolume();
            Scalar oldVolume = hypersphere_old->getVolume();
            Scalar scaleOld = pow((oldVolume/curVolume), Scalar(1.0/3.0));
            Scalar scaleNew = pow((newVolume/curVolume), Scalar(1.0/3.0));

            double dE = 0.0;
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {

                Scalar old_E = calcE(i, quat<Scalar>(*(quat_l_old+i)), quat<Scalar>(*(quat_r_old+i)), scaleOld);
                Scalar new_E = calcE(i, quat<Scalar>(*(quat_l_new+i)), quat<Scalar>(*(quat_r_new+i)), scaleNew);
                dE += new_E - old_E;
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(MPI_IN_PLACE, &dE, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                }
            #endif

            return dE;
            }

        void compute(unsigned int timestep)
            {
            if(!this->shouldCompute(timestep))
                {
                return;
                }
            m_Energy_trans = Scalar(0.0);
            m_Energy_rot = Scalar(0.0);
            // access particle data and system box
            ArrayHandle<Scalar4> h_quat_l(m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_quat_r(m_pdata->getRightQuaternionArray(), access_location::host, access_mode::read);
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {
                quat<Scalar> quat_l(h_quat_l.data[i]);
                quat<Scalar> quat_r(h_quat_r.data[i]);
                m_Energy_trans += calcE_trans(i, quat_l, quat_r, 1);
                m_Energy_rot += calcE_rot(i, quat_l, quat_r);
                }

            m_Energy = m_Energy_trans + m_Energy_rot;

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(MPI_IN_PLACE, &m_Energy, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                }
            #endif

            Scalar energy_per = m_Energy / Scalar(m_pdata->getNGlobal());
            m_EnergySum_y    = energy_per - m_EnergySum_c;
            m_EnergySum_t    = m_EnergySum + m_EnergySum_y;
            m_EnergySum_c    = (m_EnergySum_t-m_EnergySum) - m_EnergySum_y;
            m_EnergySum      = m_EnergySum_t;

            Scalar energy_sq_per = energy_per*energy_per;
            m_EnergySqSum_y    = energy_sq_per - m_EnergySqSum_c;
            m_EnergySqSum_t    = m_EnergySqSum + m_EnergySqSum_y;
            m_EnergySqSum_c    = (m_EnergySqSum_t-m_EnergySqSum) - m_EnergySqSum_y;
            m_EnergySqSum      = m_EnergySqSum_t;
            m_num_samples++;
            }

        double energydiffHypersphere(const unsigned int& index, const quat<Scalar>& quat_l_old, const quat<Scalar>& quat_r_old, const Shape& shape_old, const quat<Scalar>& quat_l_new, const quat<Scalar>& quat_r_new, const Shape& shape_new)
            {
            double old_U = calcE(index, quat_l_old, quat_r_old, shape_old);
            double new_U = calcE(index, quat_l_new, quat_r_new, shape_new);
            return new_U - old_U;
            }

        void setReferences(const pybind11::list& ql, const pybind11::list& qr, const unsigned int N)
            {
            std::vector<Scalar4> lattice_quat_l;
            std::vector<Scalar> qlbuffer;
            std::vector<Scalar4> lattice_quat_r;
            std::vector<Scalar> qrbuffer;
            std::vector<unsigned int> lattice_index;
            std::vector<bool> lattice_bool;
            #ifdef ENABLE_MPI
            unsigned int qlsz = 0, qrsz = 0, isz = 0;

            if ( this->m_exec_conf->isRoot() )
                {
                python_list_to_vector_scalar4(ql, lattice_quat_l);
                python_list_to_vector_scalar4(qr, lattice_quat_r);
                python_list_to_vector_int(ql,lattice_index);
                python_list_to_vector_bool(ql,lattice_bool,N);
                qlsz = lattice_quat_l.size();
                qrsz = lattice_quat_r.size();
                isz = lattice_index.size();
                }
            if( this->m_pdata->getDomainDecomposition())
                {
                if(qlsz)
                    {
                    qlbuffer.resize(4*qlsz, 0.0);
                    for(size_t i = 0; i < qlsz; i++)
                        {
                        qlbuffer[4*i] = lattice_quat_l[i].x;
                        qlbuffer[4*i+1] = lattice_quat_l[i].y;
                        qlbuffer[4*i+2] = lattice_quat_l[i].z;
                        qlbuffer[4*i+3] = lattice_quat_l[i].w;
                        }
                    }
                if(qrsz)
                    {
                    qrbuffer.resize(4*qrsz, 0.0);
                    for(size_t i = 0; i < qrsz; i++)
                        {
                        qrbuffer[4*i] = lattice_quat_r[i].x;
                        qrbuffer[4*i+1] = lattice_quat_r[i].y;
                        qrbuffer[4*i+2] = lattice_quat_r[i].z;
                        qrbuffer[4*i+3] = lattice_quat_r[i].w;
                        }
                    }
                MPI_Bcast(&qlsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(qlsz)
                    {
                    if(!qlbuffer.size())
                        qlbuffer.resize(4*qlsz, 0.0);
                    MPI_Bcast(&qlbuffer.front(), 4*qlsz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_quat_l.size())
                        {
                        lattice_quat_l.resize(qlsz, make_scalar4(0.0, 0.0, 0.0, 0.0));
                        for(size_t i = 0; i < qlsz; i++)
                            {
                            lattice_quat_l[i].x = qlbuffer[4*i];
                            lattice_quat_l[i].y = qlbuffer[4*i+1];
                            lattice_quat_l[i].z = qlbuffer[4*i+2];
                            lattice_quat_l[i].w = qlbuffer[4*i+3];
                            }
                        }
                    }
                MPI_Bcast(&qrsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(qrsz)
                    {
                    if(!qrbuffer.size())
                        qrbuffer.resize(4*qrsz, 0.0);
                    MPI_Bcast(&qrbuffer.front(), 4*qrsz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_quat_r.size())
                        {
                        lattice_quat_r.resize(qrsz, make_scalar4(0, 0, 0, 0));
                        for(size_t i = 0; i < qrsz; i++)
                            {
                            lattice_quat_r[i].x = qrbuffer[4*i];
                            lattice_quat_r[i].y = qrbuffer[4*i+1];
                            lattice_quat_r[i].z = qrbuffer[4*i+2];
                            lattice_quat_r[i].w = qrbuffer[4*i+3];
                            }
                        }
                    }
                }

            #else
            python_list_to_vector_scalar4(ql, lattice_quat_l);
            python_list_to_vector_scalar4(qr, lattice_quat_r);
            python_list_to_vector_int(qr, lattice_index);
            python_list_to_vector_bool(qr, lattice_bool, N);
            #endif

            if( lattice_quat_l.size() )
                m_latticeQuat_l.setReferences(lattice_quat_l.begin(), lattice_quat_l.end(), m_pdata, m_exec_conf);

            if( lattice_quat_r.size() )
                m_latticeQuat_r.setReferences(lattice_quat_r.begin(), lattice_quat_r.end(), m_pdata, m_exec_conf);

            if( lattice_index.size() )
                m_latticeIndex.setReferences(lattice_index.begin(), lattice_index.end(), m_pdata, m_exec_conf);

            if( lattice_bool.size() )
                m_latticeBool.setReferences(lattice_bool.begin(), lattice_bool.end(), m_pdata, m_exec_conf);
            }

        void clearQuat_l() { m_latticeQuat_l.clear(); }

        void clearQuat_r() { m_latticeQuat_r.clear(); }

        void clearIndex() { m_latticeIndex.clear(); }

        void clearBool() { m_latticeBool.clear(); }

        void scaleReferencePoints()
            {
                Hypersphere newHypersphere = m_pdata->getHypersphere();
                m_hypersphere = newHypersphere;
            }

        //! Returns a list of log quantities this compute calculates
        std::vector< std::string > getProvidedLogQuantities()
            {
            return m_ProvidedQuantities;
            }

        //! Calculates the requested log value and returns it
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            compute(timestep);

            if( quantity == LATTICE_ENERGY_LOG_NAME )
                {
                return m_Energy;
                }
            else if( quantity == LATTICE_ENERGY_TRANS_LOG_NAME )
                {
                return m_Energy_trans;
                }
            else if( quantity == LATTICE_ENERGY_ROT_LOG_NAME )
                {
                return m_Energy_rot;
                }
            else if( quantity == LATTICE_ENERGY_AVG_LOG_NAME )
                {
                if( !m_num_samples )
                    return 0.0;
                return m_EnergySum/double(m_num_samples);
                }
            else if ( quantity == LATTICE_ENERGY_SIGMA_LOG_NAME )
                {
                if( !m_num_samples )
                    return 0.0;
                Scalar first_moment = m_EnergySum/double(m_num_samples);
                Scalar second_moment = m_EnergySqSum/double(m_num_samples);
                return sqrt(second_moment - (first_moment*first_moment));
                }
            else if ( quantity == LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME )
                {
                return m_k;
                }
            else if ( quantity == LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME )
                {
                return m_q;
                }
            else if ( quantity == LATTICE_NUM_SAMPLES_LOG_NAME )
                {
                return m_num_samples;
                }
            else
                {
                m_exec_conf->msg->error() << "field.lattice_field: " << quantity << " is not a valid log quantity" << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }

        void setParams(Scalar k, Scalar q)
            {
            m_k = k;
            m_q = q;
            }

        const GPUArray< Scalar4 >& getReferenceLatticeQuat_l()
            {
            return m_latticeQuat_l.getReferenceArray();
            }

        const GPUArray< Scalar4 >& getReferenceLatticeQuat_r()
            {
            return m_latticeQuat_r.getReferenceArray();
            }

        const GPUArray< unsigned int >& getReferenceLatticeIndex()
            {
            return m_latticeIndex.getReferenceArray();
            }

        const GPUArray< bool >& getReferenceLatticeBool()
            {
            return m_latticeBool.getReferenceArray();
            }

        void reset( unsigned int ) // TODO: remove the timestep
            {
            m_EnergySum = m_EnergySum_y = m_EnergySum_t = m_EnergySum_c = Scalar(0.0);
            m_EnergySqSum = m_EnergySqSum_y = m_EnergySqSum_t = m_EnergySqSum_c = Scalar(0.0);
            m_num_samples = 0;
            }

        Scalar getEnergy(unsigned int timestep)
        {
            compute(timestep);
            return m_Energy;
        }
        
        Scalar getAvgEnergy(unsigned int timestep)
        {
            compute(timestep);
            if( !m_num_samples )
                return 0.0;
            return m_EnergySum/double(m_num_samples);
        }

        Scalar getSigma(unsigned int timestep)
        {
            compute(timestep);
            if( !m_num_samples )
                return 0.0;
            Scalar first_moment = m_EnergySum/double(m_num_samples);
            Scalar second_moment = m_EnergySqSum/double(m_num_samples);
            return sqrt(second_moment - (first_moment*first_moment));
        }

        unsigned int testIndex(const unsigned int& index, const quat<Scalar>& quat_l, const quat<Scalar>& quat_r)
        {
            const Hypersphere& hypersphere = this->m_pdata->getHypersphere();

            detail::AABB aabb_i = detail::AABB(hypersphere.hypersphericalToCartesian(quat_l, quat_r),m_refdist);

            OverlapReal dr = 1000;
            unsigned int k=0;

            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
              {
              if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb_i))
                  {
                  if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                      {
                      for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                          {
                          // read in its position and orientation
                          unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                          quat<Scalar> ql(m_latticeQuat_l.getReference(j));
                          quat<Scalar> qr(m_latticeQuat_r.getReference(j));

                          OverlapReal arc_length = detail::get_arclength_hypersphere(ql, qr, quat_l, quat_r, hypersphere);

                          if( arc_length < dr)
			      {
                              dr = arc_length;
                              k = j;
                              if(dr < m_refdist)
                                  break;
                              }
                          }
                      }
                  }
              else
                  {
                   //skip ahead
                  cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                  }

              if(dr < m_refdist)
                  break;

              }  // end loop over AABB nodes

              return k;
        }

        void changeIndex(const unsigned int& index, unsigned int& k, unsigned int& kk)
        {
              ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);
              m_latticeIndex.setReference(h_tags.data[index],k);
              m_latticeBool.setReference(k,true);
              m_latticeBool.setReference(kk,false);
        }


    protected:

        void setLatticeDist()
        {

            const Hypersphere& hypersphere = this->m_pdata->getHypersphere();
            quat<Scalar> ql(m_latticeQuat_l.getReference(0));
            quat<Scalar> qr(m_latticeQuat_r.getReference(0));

            Scalar dr = 1000;
            for( unsigned int i =1; i < m_latticeQuat_l.getSize(); i++){
                quat<Scalar> ql_ref(m_latticeQuat_l.getReference(i));
                quat<Scalar> qr_ref(m_latticeQuat_r.getReference(i));
                
                OverlapReal arc_length = detail::get_arclength_hypersphere(ql, qr, ql_ref, qr_ref, hypersphere);

                if(arc_length < dr)
                    dr = arc_length;
            }
            m_refdist = dr/2;

        }

        //! Grow the m_aabbs list
        void growAABBList(unsigned int N)
        {
            if (m_aabbs != NULL)
                free(m_aabbs);


            int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
            if (retval != 0)
                {
                m_exec_conf->msg->errorAllRanks() << "Error allocating aligned memory" << std::endl;
                throw std::runtime_error("Error allocating AABB memory");
                }
        }


        // These could be a little redundant. think about this more later.
        Scalar calcE_trans(const unsigned int& index, const quat<Scalar>& quat_l, const quat<Scalar>& quat_r, const Scalar& scale = 1.0)
            {
            const Hypersphere& hypersphere = this->m_pdata->getHypersphere();
            ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);

            unsigned int kk = m_latticeIndex.getReference(h_tags.data[index]);

            quat<Scalar> ql(m_latticeQuat_l.getReference(kk));
            quat<Scalar> qr(m_latticeQuat_r.getReference(kk));

            Scalar dr = detail::get_arclength_hypersphere(ql, qr, quat_l, quat_r, hypersphere);

            if(dr > m_refdist){
                unsigned int k = testIndex( index, quat_l, quat_r);
                changeIndex(index,k,kk);

                 ql = quat<Scalar>(m_latticeQuat_l.getReference(k));
                 qr = quat<Scalar>(m_latticeQuat_r.getReference(k));
                 dr = detail::get_arclength_hypersphere(ql, qr, quat_l, quat_r, hypersphere);
            }

            return m_k*dr*dr;
            }

        Scalar calcE_rot(const unsigned int& index, const quat<Scalar>& quat_l, const quat<Scalar>& quat_r)
            {
            assert(m_symmetry.size());
            ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);

            unsigned int k = m_latticeIndex.getReference(h_tags.data[index]);

            quat<Scalar> ql(m_latticeQuat_l.getReference(k));
            quat<Scalar> qr(m_latticeQuat_r.getReference(k));

            quat<Scalar> equiv_orientation1 = quat_l*quat<Scalar>(0,vec3<Scalar>(1,0,0))*quat_r;
            quat<Scalar> equiv_orientation2 = quat_l*quat<Scalar>(0,vec3<Scalar>(0,1,0))*quat_r;

            quat<Scalar> ref_pos = ql*qr;
            quat<Scalar> equiv_pos = quat_l*quat_r;

            Scalar dpos = 1/(1+dot(equiv_pos,ref_pos));
            Scalar dqmin = 0.0;
            for(size_t i = 0; i < m_symmetry.size(); i++)
                {
                quat<Scalar> ref_quat_l = ql*m_symmetry[i];
                quat<Scalar> ref_quat_r = conj(m_symmetry[i])*qr;
                quat<Scalar> ref_orientation1 = ref_quat_l*quat<Scalar>(0,vec3<Scalar>(1,0,0))*ref_quat_r;
                quat<Scalar> ref_orientation2 = ref_quat_l*quat<Scalar>(0,vec3<Scalar>(0,1,0))*ref_quat_r;

                Scalar dq1 = dot(equiv_orientation1,ref_orientation1) - dot(equiv_pos,ref_orientation1)*dot(equiv_orientation1,ref_pos)*dpos;
                Scalar dq2 = dot(equiv_orientation2,ref_orientation2) - dot(equiv_pos,ref_orientation2)*dot(equiv_orientation2,ref_pos)*dpos;
                Scalar dq = 2 - dq1*dq1 - dq2*dq2;

                dqmin = (i == 0) ? dq : fmin(dqmin, dq);
                }

            return m_q*dqmin;
            }


        Scalar calcE(const unsigned int& index, const quat<Scalar>& quat_l, const quat<Scalar>& quat_r, const Scalar& scale = 1.0)
            {
            Scalar energy = 0.0;
            if(m_latticeQuat_l.isValid() && m_latticeQuat_r.isValid())
                {
                energy += calcE_trans(index, quat_l, quat_r, scale);
                energy += calcE_rot(index, quat_l, quat_r);
                }
            return energy;
            }



        Scalar calcE_rot(const unsigned int& index, const Shape& shape)
            {
            return calcE_rot(index, shape.quat_l,shape.quat_r);
            }
        Scalar calcE(const unsigned int& index, const quat<Scalar>& quat_l, const quat<Scalar>& quat_r, const Shape& shape, const Scalar& scale = 1.0)
            {
            return calcE(index, quat_l, quat_r, scale);
            }
    private:
        LatticeReferenceList<Scalar4>   m_latticeQuat_l;         // positions of the lattice.
        Scalar                          m_k;                        // spring constant

        LatticeReferenceList<Scalar4>   m_latticeQuat_r;      // orientation of the lattice particles.
        Scalar                          m_q;                        // spring constant

        LatticeReferenceList<unsigned int>   m_latticeIndex;         // index of the lattice.
        LatticeReferenceList<bool>      m_latticeBool;         // bool of the lattice.
        Scalar                          m_refdist;

        std::vector< quat<Scalar> >     m_symmetry;       // quaternions in the symmetry group of the shape.

        Scalar                          m_Energy;                   // Store the total energy of the last computed timestep
        Scalar                          m_Energy_trans;             // Store the total energy of the last computed timestep
        Scalar                          m_Energy_rot;               // Store the total energy of the last computed timestep

        // All of these are on a per particle basis
        Scalar                          m_EnergySum;
        Scalar                          m_EnergySum_y;
        Scalar                          m_EnergySum_t;
        Scalar                          m_EnergySum_c;

        Scalar                          m_EnergySqSum;
        Scalar                          m_EnergySqSum_y;
        Scalar                          m_EnergySqSum_t;
        Scalar                          m_EnergySqSum_c;

        unsigned int                    m_num_samples;

        std::vector<std::string>        m_ProvidedQuantities;
        Hypersphere                     m_hypersphere;              //!< Save the last known box;

        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for lattice checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
    };


template< class Shape>
class ExternalFieldLattice : public ExternalFieldMono<Shape>
    {
    using ExternalFieldMono<Shape>::m_pdata;
    using ExternalFieldMono<Shape>::m_exec_conf;
    using ExternalFieldMono<Shape>::m_sysdef;
    public:
        ExternalFieldLattice(  std::shared_ptr<SystemDefinition> sysdef,
                                        pybind11::list r0,
                                        Scalar k,
                                        pybind11::list q0,
                                        Scalar q,
                                        pybind11::list symRotations
                                    ) : ExternalFieldMono<Shape>(sysdef), m_k(k), m_q(q), m_Energy(0)
            {
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_AVG_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_SIGMA_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_NUM_SAMPLES_LOG_NAME);
            m_aabbs=NULL;
            // Connect to the BoxChange signal
            m_box = m_pdata->getBox();
            m_pdata->getBoxChangeSignal().template connect<ExternalFieldLattice<Shape>, &ExternalFieldLattice<Shape>::scaleReferencePoints>(this);
            setReferences(r0, q0, m_pdata->getN());
            setLatticeDist();
            buildAABBTree();

            std::vector<Scalar4> rots;
            python_list_to_vector_scalar4(symRotations, rots);
            bool identityFound = false;
            quat<Scalar> identity(1, vec3<Scalar>(0, 0, 0));
            Scalar tol = 1e-5;
            for(size_t i = 0; i < rots.size(); i++)
                {
                quat<Scalar> qi(rots[i]);
                identityFound = !identityFound ? norm2(qi-identity) < tol : identityFound;
                m_symmetry.push_back(qi);
                }
            if(!identityFound) // ensure that the identity rotation is provided.
                {
                m_symmetry.push_back(identity);
                }
            reset(0); // initializes all of the energy logging parameters.
            }

        ~ExternalFieldLattice()
        {
            if (m_aabbs != NULL)
                free(m_aabbs);
            m_pdata->getBoxChangeSignal().template disconnect<ExternalFieldLattice<Shape>, &ExternalFieldLattice<Shape>::scaleReferencePoints>(this);
        }

        const detail::AABBTree& buildAABBTree()
    	{

    	// grow the AABB list to the needed size
    	unsigned int n_aabb = m_latticePositions.getSize();
    	if (n_aabb > 0)
    	    {
    	    growAABBList(n_aabb);

    	    for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
    	        {
    	        unsigned int i = cur_particle;

    	        Scalar radius = m_refdist;
    	        m_aabbs[i] = detail::AABB(vec3<Scalar>(m_latticePositions.getReference(i)), radius);
    	        }

    	    // build the tree
    	    m_aabb_tree.buildTree(m_aabbs, n_aabb);
    	    }

    	if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    	return m_aabb_tree;
    	}

        Scalar calculateBoltzmannWeight(unsigned int timestep) { return 0.0; }

        double calculateDeltaE(const Scalar4 * const position_old_arg,
                                        const Scalar4 * const orientation_old_arg,
                                        const BoxDim * const box_old_arg
                                        )
            {
            // TODO: rethink the formatting a bit.
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            const Scalar4 * const position_new = h_pos.data;
            const Scalar4 * const orientation_new = h_orient.data;
            const BoxDim * const box_new = &m_pdata->getGlobalBox();
            const Scalar4 * position_old=position_old_arg, * orientation_old=orientation_old_arg;
            const BoxDim * box_old = box_old_arg;
            if( !position_old )
                position_old = position_new;
            if( !orientation_old )
                orientation_old = orientation_new;
            if( !box_old )
                box_old = box_new;

            Scalar curVolume = m_box.getVolume();
            Scalar newVolume = box_new->getVolume();
            Scalar oldVolume = box_old->getVolume();
            Scalar scaleOld = pow((oldVolume/curVolume), Scalar(1.0/3.0));
            Scalar scaleNew = pow((newVolume/curVolume), Scalar(1.0/3.0));

            double dE = 0.0;
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {
                Scalar old_E = calcE(i, vec3<Scalar>(*(position_old+i)), quat<Scalar>(*(orientation_old+i)), scaleOld);
                Scalar new_E = calcE(i, vec3<Scalar>(*(position_new+i)), quat<Scalar>(*(orientation_new+i)), scaleNew);
                dE += new_E - old_E;
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(MPI_IN_PLACE, &dE, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                }
            #endif

            return dE;
            }

        void compute(unsigned int timestep)
            {
            if(!this->shouldCompute(timestep))
                {
                return;
                }
            m_Energy = Scalar(0.0);
            // access particle data and system box
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {
                vec3<Scalar> position(h_postype.data[i]);
                quat<Scalar> orientation(h_orient.data[i]);
                m_Energy += calcE(i, position, orientation);
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(MPI_IN_PLACE, &m_Energy, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                }
            #endif

            Scalar energy_per = m_Energy / Scalar(m_pdata->getNGlobal());
            m_EnergySum_y    = energy_per - m_EnergySum_c;
            m_EnergySum_t    = m_EnergySum + m_EnergySum_y;
            m_EnergySum_c    = (m_EnergySum_t-m_EnergySum) - m_EnergySum_y;
            m_EnergySum      = m_EnergySum_t;

            Scalar energy_sq_per = energy_per*energy_per;
            m_EnergySqSum_y    = energy_sq_per - m_EnergySqSum_c;
            m_EnergySqSum_t    = m_EnergySqSum + m_EnergySqSum_y;
            m_EnergySqSum_c    = (m_EnergySqSum_t-m_EnergySqSum) - m_EnergySqSum_y;
            m_EnergySqSum      = m_EnergySqSum_t;
            m_num_samples++;
            }

        double energydiff(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            double old_U = calcE(index, position_old, shape_old), new_U = calcE(index, position_new, shape_new);
            return new_U - old_U;
            }

        void setReferences(const pybind11::list& r0, const pybind11::list& q0, const unsigned int N)
            {
            unsigned int ndim = m_sysdef->getNDimensions();
            std::vector<Scalar3> lattice_positions;
            std::vector<Scalar> pbuffer;
            std::vector<Scalar4> lattice_orientations;
            std::vector<Scalar> qbuffer;
            std::vector<unsigned int> lattice_index;
            std::vector<bool> lattice_bool;
            #ifdef ENABLE_MPI
            unsigned int psz = 0, qsz = 0, isz = 0;

            if ( this->m_exec_conf->isRoot() )
                {
                python_list_to_vector_scalar3(r0, lattice_positions, ndim);
                python_list_to_vector_scalar4(q0, lattice_orientations);
                python_list_to_vector_int(r0,lattice_index);
                python_list_to_vector_bool(r0,lattice_bool,N);
                psz = lattice_positions.size();
                qsz = lattice_orientations.size();
                isz = lattice_index.size();
                }
            if( this->m_pdata->getDomainDecomposition())
                {
                if(psz)
                    {
                    pbuffer.resize(3*psz, 0.0);
                    for(size_t i = 0; i < psz; i++)
                        {
                        pbuffer[3*i] = lattice_positions[i].x;
                        pbuffer[3*i+1] = lattice_positions[i].y;
                        pbuffer[3*i+2] = lattice_positions[i].z;
                        }
                    }
                if(qsz)
                    {
                    qbuffer.resize(4*qsz, 0.0);
                    for(size_t i = 0; i < qsz; i++)
                        {
                        qbuffer[4*i] = lattice_orientations[i].x;
                        qbuffer[4*i+1] = lattice_orientations[i].y;
                        qbuffer[4*i+2] = lattice_orientations[i].z;
                        qbuffer[4*i+3] = lattice_orientations[i].w;
                        }
                    }
                MPI_Bcast(&psz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(psz)
                    {
                    if(!pbuffer.size())
                        pbuffer.resize(3*psz, 0.0);
                    MPI_Bcast(&pbuffer.front(), 3*psz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_positions.size())
                        {
                        lattice_positions.resize(psz, make_scalar3(0.0, 0.0, 0.0));
                        for(size_t i = 0; i < psz; i++)
                            {
                            lattice_positions[i].x = pbuffer[3*i];
                            lattice_positions[i].y = pbuffer[3*i+1];
                            lattice_positions[i].z = pbuffer[3*i+2];
                            }
                        }
                    }
                MPI_Bcast(&qsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(qsz)
                    {
                    if(!qbuffer.size())
                        qbuffer.resize(4*qsz, 0.0);
                    MPI_Bcast(&qbuffer.front(), 4*qsz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_orientations.size())
                        {
                        lattice_orientations.resize(qsz, make_scalar4(0, 0, 0, 0));
                        for(size_t i = 0; i < qsz; i++)
                            {
                            lattice_orientations[i].x = qbuffer[4*i];
                            lattice_orientations[i].y = qbuffer[4*i+1];
                            lattice_orientations[i].z = qbuffer[4*i+2];
                            lattice_orientations[i].w = qbuffer[4*i+3];
                            }
                        }
                    }
                }

            #else
            python_list_to_vector_scalar3(r0, lattice_positions, ndim);
            python_list_to_vector_scalar4(q0, lattice_orientations);
            python_list_to_vector_int(r0, lattice_index);
            python_list_to_vector_bool(r0,lattice_bool,N);
            #endif

            if( lattice_positions.size() )
                m_latticePositions.setReferences(lattice_positions.begin(), lattice_positions.end(), m_pdata, m_exec_conf);

            if( lattice_orientations.size() )
                m_latticeOrientations.setReferences(lattice_orientations.begin(), lattice_orientations.end(), m_pdata, m_exec_conf);

            if( lattice_index.size() )
                m_latticeIndex.setReferences(lattice_index.begin(), lattice_index.end(), m_pdata, m_exec_conf);

            if( lattice_bool.size() )
                m_latticeBool.setReferences(lattice_bool.begin(), lattice_bool.end(), m_pdata, m_exec_conf);
            }

        void clearPositions() { m_latticePositions.clear(); }

        void clearOrientations() { m_latticeOrientations.clear(); }

        void clearIndex() { m_latticeIndex.clear(); }

        void scaleReferencePoints()
            {
                BoxDim newBox = m_pdata->getBox();
                Scalar newVol = newBox.getVolume();
                Scalar lastVol = m_box.getVolume();
                Scalar scale;
                if (this->m_sysdef->getNDimensions() == 2)
                    scale = pow((newVol/lastVol), Scalar(1.0/2.0));
                else
                    scale = pow((newVol/lastVol), Scalar(1.0/3.0));
                m_latticePositions.scale(scale);
                m_box = newBox;
            }

        //! Returns a list of log quantities this compute calculates
        std::vector< std::string > getProvidedLogQuantities()
            {
            return m_ProvidedQuantities;
            }

        //! Calculates the requested log value and returns it
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            compute(timestep);

            if( quantity == LATTICE_ENERGY_LOG_NAME )
                {
                return m_Energy;
                }
            else if( quantity == LATTICE_ENERGY_AVG_LOG_NAME )
                {
                if( !m_num_samples )
                    return 0.0;
                return m_EnergySum/double(m_num_samples);
                }
            else if ( quantity == LATTICE_ENERGY_SIGMA_LOG_NAME )
                {
                if( !m_num_samples )
                    return 0.0;
                Scalar first_moment = m_EnergySum/double(m_num_samples);
                Scalar second_moment = m_EnergySqSum/double(m_num_samples);
                return sqrt(second_moment - (first_moment*first_moment));
                }
            else if ( quantity == LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME )
                {
                return m_k;
                }
            else if ( quantity == LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME )
                {
                return m_q;
                }
            else if ( quantity == LATTICE_NUM_SAMPLES_LOG_NAME )
                {
                return m_num_samples;
                }
            else
                {
                m_exec_conf->msg->error() << "field.lattice_field: " << quantity << " is not a valid log quantity" << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }

        void setParams(Scalar k, Scalar q)
            {
            m_k = k;
            m_q = q;
            }

        const GPUArray< Scalar3 >& getReferenceLatticePositions()
            {
            return m_latticePositions.getReferenceArray();
            }

        const GPUArray< Scalar4 >& getReferenceLatticeOrientations()
            {
            return m_latticeOrientations.getReferenceArray();
            }

        const GPUArray< unsigned int >& getReferenceLatticeIndex()
            {
            return m_latticeIndex.getReferenceArray();
            }

        const GPUArray< bool >& getReferenceLatticeBool()
            {
            return m_latticeBool.getReferenceArray();
            }

        void reset( unsigned int ) // TODO: remove the timestep
            {
            m_EnergySum = m_EnergySum_y = m_EnergySum_t = m_EnergySum_c = Scalar(0.0);
            m_EnergySqSum = m_EnergySqSum_y = m_EnergySqSum_t = m_EnergySqSum_c = Scalar(0.0);
            m_num_samples = 0;
            }

        Scalar getEnergy(unsigned int timestep)
        {
            compute(timestep);
            return m_Energy;
        }
        Scalar getAvgEnergy(unsigned int timestep)
        {
            compute(timestep);
            if( !m_num_samples )
                return 0.0;
            return m_EnergySum/double(m_num_samples);
        }
        Scalar getSigma(unsigned int timestep)
        {
            compute(timestep);
            if( !m_num_samples )
                return 0.0;
            Scalar first_moment = m_EnergySum/double(m_num_samples);
            Scalar second_moment = m_EnergySqSum/double(m_num_samples);
            return sqrt(second_moment - (first_moment*first_moment));
        }

        unsigned int testIndex(const unsigned int& index, const vec3<Scalar>& position)
        {
            vec3<Scalar> origin(m_pdata->getOrigin());
            const BoxDim& box = this->m_pdata->getGlobalBox();

            detail::AABB aabb_i = detail::AABB(position,m_refdist);

            OverlapReal dr_min = 1000;
            unsigned int k=0;

            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
              {
              if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb_i))
                  {
                  if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                      {
                      for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                          {
                          // read in its position and orientation
                          unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                          vec3<Scalar> r0(m_latticePositions.getReference(j));

            		  vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));

                          OverlapReal dist = fast::sqrt(dot(dr,dr));

                          if( dist < dr_min)
			      {
                              dr_min = dist;
                              k = j;
                              if(dr_min < m_refdist)
                                  break;
                              }
                          }
                      }
                  }
              else
                  {
                   //skip ahead
                  cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                  }

              if(dr_min < m_refdist)
                  break;

              }  // end loop over AABB nodes


              return k;
        }

        void changeIndex(const unsigned int& index, unsigned int& k, unsigned int& kk)
        {
              ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);
              m_latticeIndex.setReference(h_tags.data[index],k);
              m_latticeBool.setReference(k,true);
              m_latticeBool.setReference(kk,false);
        }


    protected:

        void setLatticeDist()
        {

            vec3<Scalar> origin(m_pdata->getOrigin());
            const BoxDim& box = this->m_pdata->getGlobalBox();
            vec3<Scalar> r0(m_latticePositions.getReference(0));

            Scalar dr_min = 1000;
            for( unsigned int i =1; i < m_latticePositions.getSize(); i++){
                vec3<Scalar> r1(m_latticePositions.getReference(i));

            	vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r1 - r0 + origin)));

                OverlapReal dist = fast::sqrt(dot(dr,dr));
                
                if(dist < dr_min && dist > 1e-6)
                    dr_min = dist;
            }
            m_refdist = dr_min/2;

        }

        //! Grow the m_aabbs list
        void growAABBList(unsigned int N)
        {
            if (m_aabbs != NULL)
                free(m_aabbs);


            int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
            if (retval != 0)
                {
                m_exec_conf->msg->errorAllRanks() << "Error allocating aligned memory" << std::endl;
                throw std::runtime_error("Error allocating AABB memory");
                }
        }

        // These could be a little redundant. think about this more later.
        Scalar calcE_trans(const unsigned int& index, const vec3<Scalar>& position, const Scalar& scale = 1.0)
            {
            ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);

            unsigned int kk = m_latticeIndex.getReference(h_tags.data[index]);

            vec3<Scalar> origin(m_pdata->getOrigin());
            const BoxDim& box = this->m_pdata->getGlobalBox();
            vec3<Scalar> r0(m_latticePositions.getReference(kk));
            r0 *= scale;
            vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));
	    
            Scalar dist = dot(dr,dr);

	    if(dist > m_refdist){
		unsigned int k = testIndex(index, position); 
		changeIndex(index,k,kk);

            	r0 = vec3<Scalar>(m_latticePositions.getReference(k));
            	r0 *= scale;
            	dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));
            	dist = dot(dr,dr);
	    }

            return m_k*dist;
            }


        Scalar calcE_rot(const unsigned int& index, const quat<Scalar>& orientation)
            {
            assert(m_symmetry.size());
            ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);
            unsigned int k = m_latticeIndex.getReference(h_tags.data[index]);
            quat<Scalar> q0(m_latticeOrientations.getReference(k));
            Scalar dqmin = 0.0;

            for(size_t i = 0; i < m_symmetry.size(); i++)
                {
                quat<Scalar> equiv_orientation = orientation*m_symmetry[i];
                quat<Scalar> dq = q0 - equiv_orientation;
                dqmin = (i == 0) ? norm2(dq) : fmin(dqmin, norm2(dq));
                }
            return m_q*dqmin;
            }
        Scalar calcE_rot(const unsigned int& index, const Shape& shape)
            {
            if(!shape.hasOrientation())
                return Scalar(0.0);

            return calcE_rot(index, shape.orientation);
            }
        Scalar calcE(const unsigned int& index, const vec3<Scalar>& position, const quat<Scalar>& orientation, const Scalar& scale = 1.0)
            {
            Scalar energy = 0.0;
            if(m_latticePositions.isValid())
                {
                energy += calcE_trans(index, position, scale);
                }
            if(m_latticeOrientations.isValid())
                {
                energy += calcE_rot(index, orientation);
                }
            return energy;
            }
        Scalar calcE(const unsigned int& index, const vec3<Scalar>& position, const Shape& shape, const Scalar& scale = 1.0)
            {
            return calcE(index, position, shape.orientation, scale);
            }
    private:
        LatticeReferenceList<Scalar3>   m_latticePositions;         // positions of the lattice.
        Scalar                          m_k;                        // spring constant

        LatticeReferenceList<Scalar4>   m_latticeOrientations;      // orientation of the lattice particles.
        Scalar                          m_q;                        // spring constant

        LatticeReferenceList<unsigned int>   m_latticeIndex;         // positions of the lattice.
        LatticeReferenceList<bool>      m_latticeBool;         // positions of the lattice.
        Scalar                          m_refdist;

        std::vector< quat<Scalar> >     m_symmetry;       // quaternions in the symmetry group of the shape.

        Scalar                          m_Energy;                   // Store the total energy of the last computed timestep

        // All of these are on a per particle basis
        Scalar                          m_EnergySum;
        Scalar                          m_EnergySum_y;
        Scalar                          m_EnergySum_t;
        Scalar                          m_EnergySum_c;

        Scalar                          m_EnergySqSum;
        Scalar                          m_EnergySqSum_y;
        Scalar                          m_EnergySqSum_t;
        Scalar                          m_EnergySqSum_c;

        unsigned int                    m_num_samples;

        std::vector<std::string>        m_ProvidedQuantities;
        BoxDim                          m_box;              //!< Save the last known box;

        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for lattice checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
    };

template<class Shape>
void export_LatticeField(pybind11::module& m, std::string name)
    {
   pybind11::class_<ExternalFieldLattice<Shape>, std::shared_ptr< ExternalFieldLattice<Shape> > >(m, name.c_str(), pybind11::base< ExternalFieldMono<Shape> >())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, pybind11::list, Scalar, pybind11::list, Scalar, pybind11::list>())
    .def("setReferences", &ExternalFieldLattice<Shape>::setReferences)
    .def("setParams", &ExternalFieldLattice<Shape>::setParams)
    .def("reset", &ExternalFieldLattice<Shape>::reset)
    .def("clearPositions", &ExternalFieldLattice<Shape>::clearPositions)
    .def("clearOrientations", &ExternalFieldLattice<Shape>::clearOrientations)
    .def("getEnergy", &ExternalFieldLattice<Shape>::getEnergy)
    .def("getAvgEnergy", &ExternalFieldLattice<Shape>::getAvgEnergy)
    .def("getSigma", &ExternalFieldLattice<Shape>::getSigma)
    ;
    }

void export_LatticeFields(pybind11::module& m);

template<class Shape>
void export_LatticeFieldHypersphere(pybind11::module& m, std::string name)
    {
   pybind11::class_<ExternalFieldLatticeHypersphere<Shape>, std::shared_ptr< ExternalFieldLatticeHypersphere<Shape> > >(m, name.c_str(), pybind11::base< ExternalFieldMono<Shape> >())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, pybind11::list, Scalar, pybind11::list, Scalar, pybind11::list>())
    .def("setReferences", &ExternalFieldLatticeHypersphere<Shape>::setReferences)
    .def("setParams", &ExternalFieldLatticeHypersphere<Shape>::setParams)
    .def("reset", &ExternalFieldLatticeHypersphere<Shape>::reset)
    .def("clearQuat_l", &ExternalFieldLatticeHypersphere<Shape>::clearQuat_l)
    .def("clearQuat_r", &ExternalFieldLatticeHypersphere<Shape>::clearQuat_r)
    .def("getEnergy", &ExternalFieldLatticeHypersphere<Shape>::getEnergy)
    .def("getAvgEnergy", &ExternalFieldLatticeHypersphere<Shape>::getAvgEnergy)
    .def("getSigma", &ExternalFieldLatticeHypersphere<Shape>::getSigma)
    ;
    }

void export_LatticeFieldsHypersphere(pybind11::module& m);

} // namespace hpmc

#endif // _EXTERNAL_FIELD_LATTICE_H_

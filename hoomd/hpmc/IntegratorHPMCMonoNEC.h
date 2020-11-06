// Copyright (c) 2009-2017 The Regents of the University of Michigan
// Copyright (c) 2017-2019 Marco Klement, Michael Engel
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HPMC_MONO_NEC__H__
#define __HPMC_MONO_NEC__H__

#include "IntegratorHPMCMono.h"
#include "hoomd/Autotuner.h"

//#include <random>
//#include <cfloat>

/*! \file IntegratorHPMCMonoNEC.h
    \brief Defines the template class for HPMC with Newtonian event chains
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// use PYBIND11 included from integrator !!
// #include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc
{

//! Template class for HPMC update with Newtonian event chains
/*!
    
    
    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoNEC : public IntegratorHPMCMono<Shape>
    {
    protected:
        Scalar m_chain_time;        //!< the length of a chain, given in units of time
        unsigned int m_chain_probability; //!< how often we do a chain. Replaces translation_move_probability
        Scalar m_update_fraction;   //!< if we perform chains we update several particles as one 
        
        // GlobalArray< hpmc_counters_t >     m_count_total;      // Inherited from base class
        GlobalArray< hpmc_nec_counters_t > m_nec_count_total;  //!< counters for chain statistics

//         unsigned long int count_moved_particles; // Counts translations that result in a collision (including the last particle)
//         unsigned long int count_moved_again;     // Counts translations that do not result in a collision (or end of chain)
//         unsigned long int count_move_attempts;   // Counts chains
//         unsigned long int count_events;          // Counts everything (translations, repeated translations, and rotations [if applicable])
//         
//         unsigned long int count_tuner_chains;  
//         unsigned long int count_tuner_collisions;  
        
        // statistics - pressure
        // We follow the equations of Isobe and Krauth, Journal of Chemical Physics 143, 084509 (2015)
        Scalar count_pressurevirial; 
        Scalar count_movelength;     

    private: // in line with IntegratorHPMC
        hpmc_nec_counters_t m_nec_count_run_start;             //!< Count saved at run() start
        hpmc_nec_counters_t m_nec_count_step_start;            //!< Count saved at the start of the last step

    public:
        //! Construct the integrator
        IntegratorHPMCMonoNEC(std::shared_ptr<SystemDefinition> sysdef,
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoNEC();

        //! Reset statistics counters
        virtual void resetStats()
            {
            this->m_exec_conf->msg->notice(10) << "IntegratorHPMCMonoNEC<Shape>::resetStats" << std::endl;

            IntegratorHPMCMono<Shape>::resetStats();

            // Taking IntegratorHPMC.h as reference
            ArrayHandle<hpmc_nec_counters_t> h_nec_counters(m_nec_count_total, access_location::host, access_mode::read);
            m_nec_count_run_start = h_nec_counters.data[0];
            }

        //! Change move ratio
        /*! \param m_chain_time set duration of a chain
        */
        void setChainTime(Scalar chain_time)
            {
            this->m_exec_conf->msg->notice(10) << "IntegratorHPMCMonoNEC<Shape>::setChainTime(" << chain_time << ")" << std::endl;
            m_chain_time = chain_time;
            }

        //! Get move ratio
        //! \returns ratio of translation versus rotation move attempts
        inline Scalar getChainTime()
            {
            return m_chain_time;
            }            

        //! Change move ratio
        /*! \param m_chain_time set duration of a chain
        */
        void setChainProbability(Scalar chain_probability)
            {
            this->m_exec_conf->msg->notice(10) << "IntegratorHPMCMonoNEC<Shape>::setChainProbability(" << chain_probability << ")" << std::endl;
            m_chain_probability = chain_probability * 65536.0;
            }

        //! Get move ratio
        //! \returns ratio of translation versus rotation move attempts
        inline Scalar getChainProbability()
            {
            return m_chain_probability / 65536.0;
            }            

        //! Change update_fraction
        /*! \param update_fraction new update_fraction to set
        */
        void setUpdateFraction(Scalar update_fraction)
            {
            if ( update_fraction > 0 and update_fraction <= 1.0 )
                {
                this->m_exec_conf->msg->notice(10) << "IntegratorHPMCMonoNEC<Shape>::setUpdateFraction(" << update_fraction << ")" << std::endl;
                m_update_fraction = update_fraction;
                }
            }

        //! Get update_fraction
        //! \returns number of updates as fraction of N
        inline double getUpdateFraction()
            {
            return m_update_fraction;
            }            
            
            
        /** \returns a list of provided quantities
        */
        std::vector< std::string > getProvidedLogQuantities()
            {
            // start with the integrator provided quantities
            std::vector< std::string > result = IntegratorHPMCMono<Shape>::getProvidedLogQuantities();

            // then add ours
            result.push_back("hpmc_chain_time");
//             result.push_back("hpmc_ec_move_size");
//             result.push_back("hpmc_ec_sweepequivalent");
//             result.push_back("hpmc_ec_raw_events");
//             result.push_back("hpmc_ec_raw_mvd_ptcl");
//             result.push_back("hpmc_ec_raw_mvd_agin");
//             result.push_back("hpmc_ec_raw_mv_atmpt");
            result.push_back("hpmc_ec_pressure");
            return result;
            }
            
        //! Get pressure from virial expression
        //! We follow the equations of Isobe and Krauth, Journal of Chemical Physics 143, 084509 (2015)
        //! \returns pressure
        inline double getPressure()
            {
            return (1+count_pressurevirial/count_movelength)*this->m_pdata->getN()/this->m_pdata->getBox().getVolume();
            }  
            
             

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
        Scalar getTunerParticlesPerChain()
        {
            hpmc_nec_counters_t nec_counters = getNECCounters(2);
            return nec_counters.chain_at_collision_count*Scalar(1.0)/nec_counters.chain_start_count;
        }
        
        //! Get the current counter values for NEC
        hpmc_nec_counters_t getNECCounters(unsigned int mode=0);
        
    private:
        /*!
         This function is an extracted overlap check from the precursor
         IntegratorHPMCMono. It was extracted to enhance readability.
         \param timestep
         \param i
         \param next
         \param typ_i
         \param pos_i
         \param postype_i
         \param shape_i
         \param h_overlaps
         \param h_postype
         \param h_orientation
         \param counters 
         */
        bool checkForOverlap(unsigned int timestep, 
                             unsigned int i,
                             int typ_i,
                             vec3<Scalar>& pos_i,
                             Scalar4 postype_i,
                             Shape& shape_i,
                             ArrayHandle<unsigned int>& h_overlaps,
                             ArrayHandle<Scalar4>& h_postype,
                             ArrayHandle<Scalar4>& h_orientation,
                             hpmc_counters_t& counters
                            );

        /*!
         This function measures the distance sphere 'i' could move in
         'direction' until it would hit another particle.
         
         To enhance logic and speed the distance is limited to the parameter
         maxSweep.
         
         Much of the layout is based on the checkForOverlap function.
         \param timestep
         \param direction Where are we going?
         \param maxSweep  How far will we go maximal
         \param i
         \param next
         \param typ_i
         \param pos_i
         \param postype_i
         \param shape_i
         \param h_overlaps
         \param h_postype
         \param h_orientation
         \param counters 
         \param collisionPlaneVector
         */
        double sweepDistance(unsigned int timestep,
                             vec3<Scalar>& direction,
                             double maxSweep,
                             unsigned int i,
                             int& next,
                             int typ_i,
                             vec3<Scalar>& pos_i,
                             Scalar4 postype_i,
                             ShapeSphere& shape_i,
                             ArrayHandle<unsigned int>& h_overlaps,
                             ArrayHandle<Scalar4>& h_postype,
                             ArrayHandle<Scalar4>& h_orientation,
                             hpmc_nec_counters_t& nec_counters,
                             vec3<Scalar>& collisionPlaneVector
                            );
        /*!
         This function measures the distance ConvexPolyhedron 'i' could move in
         'direction' until it would hit another particle.
         
         To enhance logic and speed the distance is limited to the parameter
         maxSweep.
         
         Much of the layout is based on the checkForOverlap function.
         \param timestep
         \param direction Where are we going?
         \param maxSweep  How far will we go maximal
         \param i
         \param next
         \param typ_i
         \param pos_i
         \param postype_i
         \param shape_i
         \param h_overlaps
         \param h_postype
         \param h_orientation
         \param counters 
         \param collisionPlaneVector
         */
        double sweepDistance(unsigned int timestep,
                             vec3<Scalar>& direction,
                             double maxSweep,
                             unsigned int i,
                             int& next,
                             int typ_i,
                             vec3<Scalar>& pos_i,
                             Scalar4 postype_i,
                             ShapeConvexPolyhedron& shape_i,
                             ArrayHandle<unsigned int>& h_overlaps,
                             ArrayHandle<Scalar4>& h_postype,
                             ArrayHandle<Scalar4>& h_orientation,
                             hpmc_nec_counters_t& nec_counters,
                             vec3<Scalar>& collisionPlaneVector
                            );

    protected:

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    };

/*! \param sysdef System definition
    \param cl Cell list
    \param seed Random number generator seed

    NOTE: only 3d supported at this time
    */

template< class Shape >
IntegratorHPMCMonoNEC< Shape >::IntegratorHPMCMonoNEC(std::shared_ptr<SystemDefinition> sysdef,
                                                                   unsigned int seed)
    : IntegratorHPMCMono<Shape>(sysdef, seed)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMCMonoNEC" << std::endl;
    count_pressurevirial  = 0.0;
    count_movelength      = 0.0;
    
    m_update_fraction     = 1.0;
    m_chain_probability   = 0.01;
    m_chain_time          = 1.0;
    
    GlobalArray<hpmc_nec_counters_t> nec_counters(1, this->m_exec_conf);
    m_nec_count_total.swap(nec_counters);
    }

//! Destructor
template< class Shape >
IntegratorHPMCMonoNEC< Shape >::~IntegratorHPMCMonoNEC()
    {
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    IntegratorHPMC maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
template< class Shape >
hpmc_nec_counters_t IntegratorHPMCMonoNEC< Shape >::getNECCounters(unsigned int mode)
    {
    ArrayHandle<hpmc_nec_counters_t> h_nec_counters(m_nec_count_total, access_location::host, access_mode::read);
    hpmc_nec_counters_t result;

    if (mode == 0)
        result = h_nec_counters.data[0];
    else if (mode == 1)
        result = h_nec_counters.data[0] - m_nec_count_run_start;
    else
        result = h_nec_counters.data[0] - m_nec_count_step_start;

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // MPI Reduction to total result values on all nodes.
        MPI_Allreduce(MPI_IN_PLACE, &result.chain_start_count,        1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.chain_at_collision_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.chain_no_collision_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.distance_queries,         1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.overlap_err_count,        1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }
    
    
template< class Shape >
void IntegratorHPMCMonoNEC< Shape >::update(unsigned int timestep)
    {
    this->m_exec_conf->msg->notice(10) << "HPMCMonoEC update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    // get needed vars
    ArrayHandle<hpmc_counters_t> h_counters(this->m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];
    // m_count_step_start = h_counters.data[0]; // in IntegratorHPMC
    
    ArrayHandle<hpmc_nec_counters_t> h_nec_counters(m_nec_count_total, access_location::host, access_mode::readwrite);
    hpmc_nec_counters_t& nec_counters = h_nec_counters.data[0];
    m_nec_count_step_start = h_nec_counters.data[0];


    const BoxDim& box = this->m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = this->m_nominal_width / npd;
    
    vec3<Scalar> lattice_x = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> lattice_y = vec3<Scalar>(box.getLatticeVector(1));
    vec3<Scalar> lattice_z = vec3<Scalar>(box.getLatticeVector(2));

    vec3<Scalar> normal_x = cross(lattice_y,lattice_z);
    normal_x /= sqrt(dot(normal_x,normal_x));
    vec3<Scalar> normal_y = cross(lattice_z,lattice_x);
    normal_y /= sqrt(dot(normal_y,normal_y));
    vec3<Scalar> normal_z = cross(lattice_x,lattice_y);
    normal_z /= sqrt(dot(normal_z,normal_z));

    Scalar latticeNormal_x = dot(normal_x,lattice_x) * (1 - ghost_fraction.x);
    Scalar latticeNormal_y = dot(normal_y,lattice_y) * (1 - ghost_fraction.y);
    Scalar latticeNormal_z = dot(normal_z,lattice_z) * (1 - ghost_fraction.z);

    uchar3 periodic = box.getPeriodic();
    #endif

    // reset pressure statistics
    count_pressurevirial = 0.0;
    count_movelength = 0.0;

//     count_tuner_chains = 0;
//     count_tuner_collisions = 0;

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC EC update");

    if( this->m_external ) // I think we need this here otherwise I don't think it will get called.
        {
        this->m_external->compute(timestep);
        }

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(this->m_overlaps, access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

	// access particle data
    ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_velocities(this->m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);

    //access move sizes
    ArrayHandle<Scalar> h_d(this->m_d, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_a(this->m_a, access_location::host, access_mode::read);

    // loop over local particles nselect times
    for (unsigned int i_nselect = 0; i_nselect < this->m_nselect; i_nselect++)
        {

        // With chains particles move way more, so we need to update the AABB-Tree more often.
        // Previously n_select = 1 was fine. To avoid confusion
    
        // update the AABB Tree
        this->buildAABBTree();
        // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
        this->limitMoveDistances();
        // update the image list
        this->updateImageList();

        // loop through N particles in a shuffled order
        for (unsigned int cur_chain = 0; cur_chain < this->m_pdata->getN() * m_update_fraction; cur_chain++)
            {
            // Get the RNG for chain cur_chain.
            hoomd::RandomGenerator rng_chain_i(hoomd::RNGIdentifier::HPMCMonoChainMove, this->m_seed, cur_chain, this->m_exec_conf->getRank()*this->m_nselect + i_nselect, timestep);

            // this->m_update_order.shuffle(...) wants to update the particles in forward or reverse order.
            // For chains this is an invalid behavior. Instead we have to pick a starting particle randomly.
            unsigned int i = hoomd::UniformIntDistribution( this->m_pdata->getN()-1 )(rng_chain_i);

            Scalar4 postype_i     = h_postype.data[i];
            Scalar4 orientation_i = h_orientation.data[i];
            Scalar4 velocity_i    = h_velocities.data[i];
            int typ_i             = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<Scalar>(orientation_i), this->m_params[typ_i]);
  
            #ifdef ENABLE_MPI
            if (this->m_comm)
                {
                // only move particle if active
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                    continue;
                }
            #endif

            unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng_chain_i);
            bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < m_chain_probability);

            if (move_type_translate)
                {
                // start a chain
                // -> increment chain counter
                nec_counters.chain_start_count++;
                //count_tuner_chains++;
            
                // take the particle's velocity as direction and normalize the direction vector
                vec3<Scalar> direction = vec3<Scalar>(velocity_i);
                Scalar       velocity  = sqrt( dot(direction,direction));
            
                if( velocity == 0.0 )
                    {
                    this->m_exec_conf->msg->error() << "Trying to start a chain with exactly zero velocity. Were velocities initialized propperly?" << std::endl;
                    break;
                    }
                    
                direction /= velocity;
                    
                double chain_time = m_chain_time;
                double sweep;
                
                int debug_max_chain = 1e5;
                int count_chain = 0;
                
                // perform the chain in a loop.
                // next denotes the next particle, where -1 means there is no further particle.
                //int prev = -1;
                int next = i;
                while( next > -1 )
                    {
                    
                    count_chain++;
                    if( count_chain == debug_max_chain )
                    {
                        this->m_exec_conf->msg->error() << "The number of chain elements exceeded safe-guard limit of "<<debug_max_chain<<".\n";
                        this->m_exec_conf->msg->error() << "Shorten chain_time if this message appears regularly." << std::endl;
                        break;
                    }
                        
                    // k is the current particle, which is to be moved
                    int k = next;
                    
                    // read in the current position and orientation
                    Scalar4 orientation_k = h_orientation.data[k];
                    
                    Scalar4 postype_k     = h_postype.data[k];
                    int     typ_k         = __scalar_as_int(postype_k.w);
                    vec3<Scalar> pos_k    = vec3<Scalar>(postype_k);

                    Shape   shape_k(  quat<Scalar>(orientation_k), this->m_params[typ_k]);

                    
                    // we use the parameter 'd' as search radius for potential collisions. 
                    // if we can not find a collision partner within this distance we will
                    // move the particle again in the next step. 
                    // This is neccessary to not check the whole simulation volume (or a 
                    // huge fraction of it) for potential collisions.
                    double maxSweep  = h_d.data[typ_k];
                    
                    
                    // the collisionPlaneVector will be set to the direction of the separating axis as two particles collide.
                    // for spheres this is r_ij (or delta_pos later), for polyhedron it is different.
                    // to make sure that the collided particle will not run into the one we are currently moving we use this
                    // vector instead of r_ij
                    vec3<Scalar> collisionPlaneVector;
                    
                    
                    // measure the distance to the next particle
                    // updates:
                    //   sweep                 - return value (units of length)
                    //   next                  - given as reference
                    //   collisionPlaneVector  - given as reference
                    sweep = sweepDistance(timestep,
                                        direction,
                                        maxSweep,
                                        k,
                                        next,
                                        typ_k,
                                        pos_k,
                                        postype_k,
                                        shape_k,
                                        h_overlaps,
                                        h_postype,
                                        h_orientation,
                                        nec_counters,
                                        collisionPlaneVector
                                        );

                    #ifdef ENABLE_MPI
                    if (this->m_comm)
                        {
                        // Collide with walls of the active domain for non-periodic dimensions.
                    
                        vec3<Scalar> pos_rel = pos_k - vec3<Scalar>(box.getLo());
                        // x
                        if( not periodic.x )
                            {
                            Scalar directionNormal = dot(normal_x,direction);
                            Scalar posNormal       = dot(normal_x,pos_rel);
                            Scalar planeCollisionDistance = 0.0;
                            if( directionNormal < 0.0 )
                                {
                                planeCollisionDistance = - posNormal / directionNormal;
                                }
                            else
                                {
                                planeCollisionDistance = (latticeNormal_x - posNormal) / directionNormal;
                                }
                            
                            if( planeCollisionDistance < sweep )
                                {
                                sweep = planeCollisionDistance;
                                collisionPlaneVector = normal_x;
                                next = -2;
                                }
                            }
                        // y
                        if( not periodic.y )
                            {
                            Scalar directionNormal = dot(normal_y,direction);
                            Scalar posNormal       = dot(normal_y,pos_rel);
                            Scalar planeCollisionDistance = 0.0;
                            if( directionNormal < 0.0 )
                                {
                                planeCollisionDistance = - posNormal / directionNormal;
                                }
                            else
                                {
                                planeCollisionDistance = (latticeNormal_y - posNormal) / directionNormal;
                                }
                            
                            if( planeCollisionDistance < sweep )
                                {
                                sweep = planeCollisionDistance;
                                collisionPlaneVector = normal_y;
                                next = -2;
                                }
                            }
                        // z
                        if( not periodic.z )
                            {
                            Scalar directionNormal = dot(normal_z,direction);
                            Scalar posNormal       = dot(normal_z,pos_rel);
                            Scalar planeCollisionDistance = 0.0;
                            if( directionNormal < 0.0 )
                                {
                                planeCollisionDistance = - posNormal / directionNormal;
                                }
                            else
                                {
                                planeCollisionDistance = (latticeNormal_z - posNormal) / directionNormal;
                                }
                            
                            if( planeCollisionDistance < sweep )
                                {
                                sweep = planeCollisionDistance;
                                collisionPlaneVector = normal_z;
                                next = -2;
                                }
                            }
                        }
                    #endif
            
                    // Error handling
                    // If the next collision is further than we looked for a collision 
                    // limit the possible update and try again in the next iteration
                    if( sweep > maxSweep )
                        {
                        sweep = maxSweep;
                        next  = k;
                        }

                    // if we go further than what is left: stop
                    if( sweep > chain_time * velocity )
                        {
                        sweep = chain_time * velocity;
                        next  = -1;
                        }

                    //statistics for pressure  -1-
                    count_movelength  += sweep;

                    pos_k             += sweep * direction;
                    chain_time        -= sweep / velocity ;

                    // increment accept counter
                    if (!shape_i.ignoreStatistics())
                        {
                    
                        // Note:
                        // counters.translate_* can still be used by tuners. i.e. MoveSize
                        // we want a very low "acceptance" like 3% though, to avoid too many
                        // updates to be calculated twice, yet not much lower, as considering
                        // possible neighbors in a far range costs time itself.
                        if( next != k and next > -1)
                            {
                            counters.translate_reject_count++;
                            nec_counters.chain_at_collision_count++;
                            }
                            else
                            {
                            if( next != -1 )
                                {
                                counters.translate_accept_count++;
                                }
                            nec_counters.chain_no_collision_count++;
                            }
                        }

                    // update position of particle
                    h_postype.data[k] = make_scalar4(pos_k.x,pos_k.y,pos_k.z,postype_k.w);
                    box.wrap(h_postype.data[k], h_image.data[k]);
                    
                    // update the position of the particle in the tree for future updates
                    detail::AABB aabb_k_local = shape_k.getAABB(vec3<Scalar>(0,0,0));
                    detail::AABB aabb = aabb_k_local;
                    //aabb.translate(pos_k);
                    aabb.translate( vec3<Scalar>(h_postype.data[k]));
                    this->m_aabb_tree.update(k, aabb);


                    
                    #ifdef ENABLE_MPI
                    if (this->m_comm and next == -2)
                        {
                        // collide with walls
                        vec3<Scalar> vel_k = vec3<Scalar>(h_velocities.data[k]);

                        vec3<Scalar> vel_change = collisionPlaneVector * (dot(vel_k,collisionPlaneVector) / dot(collisionPlaneVector,collisionPlaneVector));
                        vel_k = vel_k - 2.0 * vel_change;

                        h_velocities.data[k]    = make_scalar4( vel_k.x, vel_k.y, vel_k.z, h_velocities.data[k].w);
                        next = k;
                    
                        velocity = sqrt( dot(vel_k,vel_k));
                                
                        direction = vel_k / velocity;
                        }
                    else
                    #endif
                    // Update the velocities of 'k' and 'next'
                    // unless there was no collision
                    if ( next != k and next > -1 )
                        {
                        vec3<Scalar> pos_n = vec3<Scalar>(h_postype.data[next]);

                        vec3<Scalar> vel_n = vec3<Scalar>(h_velocities.data[next]);
                        vec3<Scalar> vel_k = vec3<Scalar>(h_velocities.data[k]);
                        
                        int3 null_image;
                        vec3<Scalar> delta_pos = pos_n-pos_k;
                        box.wrap(delta_pos, null_image );
                        
                        //statistics for pressure  -2-
                        count_pressurevirial   += dot(delta_pos, direction);
                        //count_tuner_collisions++;

                        #ifdef ENABLE_MPI
                        if (! this->m_comm || isActive( vec_to_scalar3(pos_n),box,ghost_fraction) )
                            {
                        #endif

                            // Update Velocities (fully elastic)
                            vec3<Scalar> delta_vel  = vel_n-vel_k;
                            vec3<Scalar> vel_change = collisionPlaneVector * (dot(delta_vel,collisionPlaneVector) / dot(collisionPlaneVector,collisionPlaneVector));
                            
                            //
                            //  Update Velocities when actually colliding
                            //  otherwise they will collide again in the next step.
                            //
                            vel_n -= vel_change;
                            vel_k += vel_change;

                            h_velocities.data[next] = make_scalar4( vel_n.x, vel_n.y, vel_n.z, h_velocities.data[next].w);
                            h_velocities.data[k]    = make_scalar4( vel_k.x, vel_k.y, vel_k.z, h_velocities.data[k].w);

                            velocity = sqrt( dot(vel_n,vel_n));
                            
                            direction = vel_n / velocity;
                        #ifdef ENABLE_MPI
                            }
                        else
                            { // if colliding with an inactive particle.
                                vec3<Scalar> vel_change = collisionPlaneVector * (dot(vel_k,collisionPlaneVector) / dot(collisionPlaneVector,collisionPlaneVector));
                                vel_k = vel_k - 2.0 * vel_change;

                                h_velocities.data[k]    = make_scalar4( vel_k.x, vel_k.y, vel_k.z, h_velocities.data[k].w);
                                next = k;
                                
                                velocity = sqrt( dot(vel_k,vel_k));
                                
                                direction = vel_k / velocity;
                            }
                        #endif
                        
                        if( velocity == 0.0 )
                            {
                            this->m_exec_conf->msg->warning() << "Cannot continue a chain without moving.\n";
                            next = -1;
                            }

                            // store previous particle.
                            //prev = k;
                        }
                    } // end loop over totalDist.

//                 counters.translate_accept_count++;

                }
            else
                {
                // Get the RNG for current particle
                hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::HPMCMonoTrialMove, this->m_seed, i, this->m_exec_conf->getRank()*this->m_nselect + i_nselect, timestep);

                bool overlap = false;
                Scalar4 postype_i     = h_postype.data[i];
                Scalar4 orientation_i = h_orientation.data[i];
                vec3<Scalar> pos_i    = vec3<Scalar>(postype_i);
                Shape shape_old(quat<Scalar>(orientation_i), this->m_params[typ_i]);

            
                //move_rotate(shape_i.orientation, rng_i, h_a.data[typ_i], ndim);
                if (ndim == 2)
                    move_rotate<2>(shape_i.orientation, rng_i, h_a.data[typ_i]);
                else
                    move_rotate<3>(shape_i.orientation, rng_i, h_a.data[typ_i]);
                
                
                detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

                overlap = checkForOverlap(timestep,
                                            i,
                                            typ_i,
                                            pos_i,
                                            postype_i,
                                            shape_i,
                                            h_overlaps,
                                            h_postype,
                                            h_orientation,
                                            counters
                                            );

                // if the move is accepted
                if (!overlap)
                    {
                    // increment accept counter and assign new position
                    if (!shape_i.ignoreStatistics())
                        {
                        if (move_type_translate)
                            counters.translate_accept_count++;
                        else
                            counters.rotate_accept_count++;
                        }
                    
                    
                    // update the position of the particle in the tree for future updates
                    detail::AABB aabb = aabb_i_local;
                    aabb.translate(pos_i);
                    this->m_aabb_tree.update(i, aabb);

                    // update position of particle
                    h_postype.data[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);

                    if (shape_i.hasOrientation())
                        {
                        h_orientation.data[i] = quat_to_scalar4(shape_i.orientation);
                        }
                    }

                else
                    {
                    if (!shape_i.ignoreStatistics())
                        {
                        // increment reject counter
                        if (move_type_translate)
                            counters.translate_reject_count++;
                        else
                            counters.rotate_reject_count++;
                        }
                    }
                } // end loop over totalDist.
            } // end loop over all particles
        } // end loop over nselect

        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // wrap particles back into box
        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        }

    // perform the grid shift
    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // precalculate the grid shift
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);
        Scalar3 shift = make_scalar3(0,0,0);
        hoomd::UniformDistribution<Scalar> uniform(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        shift.x = uniform(rng);
        shift.y = uniform(rng);
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = uniform(rng);
            }
        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
            r_i += vec3<Scalar>(shift);
            h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        this->m_pdata->translateOrigin(shift);
        }
    #endif
    
    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    // migrate and exchange particles
    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    
    hpmc_counters_t     run_counters     = this->getCounters(1);
    hpmc_nec_counters_t run_nec_counters = getNECCounters(1);
    double cur_time = double(this->m_clock.getTime()) / Scalar(1e9);
    unsigned long long sum_of_moves = run_counters.rotate_accept_count + run_counters.rotate_reject_count + run_nec_counters.chain_at_collision_count + run_nec_counters.chain_no_collision_count;
    this->m_mps = double(sum_of_moves) / cur_time;
    }

    
template< class Shape >
bool IntegratorHPMCMonoNEC< Shape >::checkForOverlap(unsigned int timestep, 
                                                    unsigned int i,
                                                    int typ_i,
                                                    vec3<Scalar>& pos_i,
                                                    Scalar4 postype_i,
                                                    Shape& shape_i,
                                                    ArrayHandle<unsigned int>& h_overlaps,
                                                    ArrayHandle<Scalar4>& h_postype,
                                                    ArrayHandle<Scalar4>& h_orientation,
                                                    hpmc_counters_t& counters
                                                   )
    {
    bool overlap=false;
    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));
    
    // All image boxes (including the primary)
    const unsigned int n_images = this->m_image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
        detail::AABB aabb = aabb_i_local;
        aabb.translate(pos_i_image);

        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j;
                        Scalar4 orientation_j;

                        // handle j==i situations
                        if ( j != i )
                            {
                            // load the position and orientation of the j particle
                            postype_j = h_postype.data[j];
                            orientation_j = h_orientation.data[j];
                            }
                        else
                            {
                            if (cur_image == 0)
                                {
                                // in the first image, skip i == j
                                continue;
                                }
                            else
                                {
                                // If this is particle i and we are in an outside image, use the translated position and orientation
                                postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                orientation_j = quat_to_scalar4(shape_i.orientation);
                                }
                            }

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                        counters.overlap_checks++;
                        if (h_overlaps.data[this->m_overlap_idx(typ_i, typ_j)]
                            && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                            && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                            {
                            overlap = true;
                            break;
                            }
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                }

            if (overlap)
                break;
            }  // end loop over AABB nodes

        if (overlap)
            break;
        } // end loop over images
        
    return overlap;
    }
   
// sweepableDistance for spheres
template< class Shape >
double IntegratorHPMCMonoNEC< Shape >::sweepDistance(unsigned int timestep,
                                                    vec3<Scalar>& direction,
                                                    double maxSweep,
                                                    unsigned int i,
                                                    int& next,
                                                    int typ_i,
                                                    vec3<Scalar>& pos_i,
                                                    Scalar4 postype_i,
                                                    ShapeSphere& shape_i,
                                                    ArrayHandle<unsigned int>& h_overlaps,
                                                    ArrayHandle<Scalar4>& h_postype,
                                                    ArrayHandle<Scalar4>& h_orientation,
                                                    hpmc_nec_counters_t& nec_counters, 
                                                    vec3<Scalar>& collisionPlaneVector
                                                   )
    {
    double sweepableDistance = maxSweep;

    direction /= sqrt(dot(direction,direction));
    
    detail::AABB aabb_i_current = shape_i.getAABB(vec3<Scalar>(0,0,0));
    detail::AABB aabb_i_future  = aabb_i_current;
    aabb_i_future.translate(maxSweep * direction);
    
    detail::AABB aabb_i_test    = detail::merge( aabb_i_current, aabb_i_future );

    
    // All image boxes (including the primary)
    const unsigned int n_images = this->m_image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
        detail::AABB aabb = aabb_i_test;
        aabb.translate(pos_i_image);

        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j;
                        Scalar4 orientation_j;

                        // handle j==i situations
                        if ( j != i )
                            {
                            // load the position and orientation of the j particle
                            postype_j = h_postype.data[j];
                            orientation_j = h_orientation.data[j];
                            }
                        else
                            {
                            if (cur_image == 0)
                                {
                                // in the first image, skip i == j
                                continue;
                                }
                            else
                                {
                                // If this is particle i and we are in an outside image, use the translated position and orientation
                                postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                orientation_j = quat_to_scalar4(shape_i.orientation);
                                }
                            }

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                        nec_counters.distance_queries++;
                        
                        
                        if ( h_overlaps.data[this->m_overlap_idx(typ_i, typ_j)])
                            {
                            double sumR   =   shape_i.params.radius
                                            + shape_j.params.radius;
                            double maxR   = sumR + sweepableDistance;
                            double distSQ = dot(r_ij,r_ij);
                        
                            if( distSQ < maxR*maxR )
                                {
                                double d_parallel =  dot(r_ij, direction);
                                if( d_parallel   <= 0 ) continue; // Moving apart
                                
                                double discriminant = sumR*sumR - distSQ + d_parallel*d_parallel;
                                if( discriminant < 0 )
                                    {
                                    // orthogonal distance larger than sum of radii
                                    continue;
                                    }
                                    
                                double newDist = d_parallel - sqrt( discriminant );
                            
                                if( newDist > 0)
                                    {
                                    if( newDist < sweepableDistance )
                                        {
                                        sweepableDistance = newDist;
                                        next = j;
                                        // calculate delta_pos on touch
                                        collisionPlaneVector = r_ij - direction * newDist ;
                                        }
                                    else
                                        {
                                        if( newDist == sweepableDistance )
                                            {
                                            this->m_exec_conf->msg->error() << "Two particles with the same distance" << std::endl;
                                            }
                                        }
                                    }
                                else
                                    {
                                    this->m_exec_conf->msg->error() << "Two particles overlapping [with negative sweepable distance]." << i << " and " << j  << std::endl;
                                    this->m_exec_conf->msg->error() << "Proceeding with the new particle without moving the initial one" << std::endl;
                                
                                    sweepableDistance = 0.0;
                                    next = j;
                                    collisionPlaneVector = r_ij;
                                    
                                    return sweepableDistance;
                                    }
                                }
                            }
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                }
            }  // end loop over AABB nodes
        } // end loop over images
        
    return sweepableDistance;
    }



    
// sweepableDistance for convex polyhedron
template< class Shape >
double IntegratorHPMCMonoNEC< Shape >::sweepDistance(unsigned int timestep,
                                                    vec3<Scalar>& direction,
                                                    double maxSweep,
                                                    unsigned int i,
                                                    int& next,
                                                    int typ_i,
                                                    vec3<Scalar>& pos_i,
                                                    Scalar4 postype_i,
                                                    ShapeConvexPolyhedron& shape_i,
                                                    ArrayHandle<unsigned int>& h_overlaps,
                                                    ArrayHandle<Scalar4>& h_postype,
                                                    ArrayHandle<Scalar4>& h_orientation,
                                                    hpmc_nec_counters_t& nec_counters,
                                                    vec3<Scalar>& collisionPlaneVector
                                                   )
    {
    double sweepableDistance = maxSweep;

    detail::AABB aabb_i_current = shape_i.getAABB(vec3<Scalar>(0,0,0));
    detail::AABB aabb_i_future  = aabb_i_current;
    aabb_i_future.translate(maxSweep * direction);
    
    detail::AABB aabb_i_test    = detail::merge( aabb_i_current, aabb_i_future );

    vec3<Scalar> newCollisionPlaneVector;

    
    // All image boxes (including the primary)
    const unsigned int n_images = this->m_image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
        detail::AABB aabb = aabb_i_test;
        aabb.translate(pos_i_image);
    
        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j;
                        Scalar4 orientation_j;

                        // handle j==i situations
                        if ( j != i )
                            {
                            // load the position and orientation of the j particle
                            postype_j = h_postype.data[j];
                            orientation_j = h_orientation.data[j];
                            }
                        else
                            {
                            if (cur_image == 0)
                                {
                                // in the first image, skip i == j
                                continue;
                                }
                            else
                                {
                                // If this is particle i and we are in an outside image, use the translated position and orientation
                                postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                orientation_j = quat_to_scalar4(shape_i.orientation);
                                }
                            }

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                        nec_counters.distance_queries++;
                        
                        
                        if ( h_overlaps.data[this->m_overlap_idx(typ_i, typ_j)])
                            {
                            double maxR =   shape_i.getCircumsphereDiameter()
                                          + shape_j.getCircumsphereDiameter();
                            maxR /= 2;
                            maxR += sweepableDistance;//maxSweep;
                        
                            if( dot(r_ij,r_ij) < maxR*maxR )
                                {
                                double newDist = sweep_distance(r_ij, shape_i, shape_j, direction, nec_counters.overlap_err_count, newCollisionPlaneVector);
                            
                                if( newDist >= 0 and newDist < sweepableDistance )
                                    {
                                    collisionPlaneVector = newCollisionPlaneVector;
                                    sweepableDistance = newDist;
                                    next = j;
                                    }
                                else
                                    {
                                    if( newDist < -3.5 ) // resultOverlapping = -3.0;
                                        {
                                        
                                        if( dot(r_ij,direction) > 0 )
                                            {
                                            collisionPlaneVector = newCollisionPlaneVector;
                                            next = j;
                                            sweepableDistance = 0.0;
                                            }
                                        }
                                    
                                        
                                        
                                    if( newDist == sweepableDistance )
                                        {
                                        this->m_exec_conf->msg->error() << "Two particles with the same distance\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                }
            }  // end loop over AABB nodes
        } // end loop over images
        
    return sweepableDistance;
    }


/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \return the requested log quantity.
*/
template<class Shape>
Scalar IntegratorHPMCMonoNEC<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "hpmc_chain_time")
        {
        return m_chain_time;
        }
//     if (quantity == "hpmc_ec_move_size")
//         {
//         if( count_move_attempts == 0 ) return 0;
//         return (Scalar)count_moved_particles / (Scalar)count_move_attempts;
//         }
//     if (quantity == "hpmc_ec_sweepequivalent")
//         {
//         return (Scalar)count_events / (Scalar)this->m_pdata->getN();
//         }
//     if (quantity == "hpmc_ec_raw_events")
//         {
//         return (Scalar)count_events;
//         }
//     if (quantity == "hpmc_ec_raw_mvd_ptcl")
//         {
//         return (Scalar)count_moved_particles;
//         }
//     if (quantity == "hpmc_ec_raw_mvd_agin")
//         {
//         return (Scalar)count_moved_again;
//         }
//     if (quantity == "hpmc_ec_raw_mv_atmpt")
//         {
//         return (Scalar)count_move_attempts;
//         }
    if (quantity == "hpmc_ec_pressure")
        {
        return (1+count_pressurevirial/count_movelength)*this->m_pdata->getN()/this->m_pdata->getBox().getVolume();
        }


    //nothing found -> pass on to base class
    return IntegratorHPMCMono<Shape>::getLogValue(quantity, timestep);
    }


//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
    \todo !!!
*/
template < class Shape > void export_IntegratorHPMCMonoNEC(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< IntegratorHPMCMonoNEC<Shape>, IntegratorHPMCMono<Shape>, IntegratorHPMC,
            std::shared_ptr< IntegratorHPMCMonoNEC<Shape> > >(m, name.c_str())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, unsigned int >())
        .def("setChainTime", &IntegratorHPMCMonoNEC<Shape>::setChainTime)
        .def("getChainTime", &IntegratorHPMCMonoNEC<Shape>::getChainTime)
        .def_property("chain_time", &IntegratorHPMCMonoNEC<Shape>::getChainTime, &IntegratorHPMCMonoNEC<Shape>::setChainTime)
        .def("setChainProbability", &IntegratorHPMCMonoNEC<Shape>::setChainProbability)
        .def("getChainProbability", &IntegratorHPMCMonoNEC<Shape>::getChainProbability)
        .def_property("chain_probability", &IntegratorHPMCMonoNEC<Shape>::getChainProbability, &IntegratorHPMCMonoNEC<Shape>::setChainProbability)
        .def("setUpdateFraction", &IntegratorHPMCMonoNEC<Shape>::setUpdateFraction)
        .def("getUpdateFraction", &IntegratorHPMCMonoNEC<Shape>::getUpdateFraction)
        .def_property("update_fraction", &IntegratorHPMCMonoNEC<Shape>::getUpdateFraction, &IntegratorHPMCMonoNEC<Shape>::setUpdateFraction)
        .def("getTunerParticlesPerChain", &IntegratorHPMCMonoNEC<Shape>::getTunerParticlesPerChain)
        .def("getPressure", &IntegratorHPMCMonoNEC<Shape>::getPressure)
        .def("getNECCounters", &IntegratorHPMCMonoNEC<Shape>::getNECCounters)
        ;

    }
    
inline void export_hpmc_nec_counters(pybind11::module& m)
    {
    pybind11::class_< hpmc_nec_counters_t >(m, "hpmc_nec_counters_t")
    .def_readonly("chain_start_count",        &hpmc_nec_counters_t::chain_start_count)
    .def_readonly("chain_at_collision_count", &hpmc_nec_counters_t::chain_at_collision_count)
    .def_readonly("chain_no_collision_count", &hpmc_nec_counters_t::chain_no_collision_count)
    .def_readonly("distance_queries",         &hpmc_nec_counters_t::distance_queries)
    .def_readonly("overlap_errors",           &hpmc_nec_counters_t::overlap_err_count)
    ;
    }
} // end namespace hpmc

#endif // __HPMC_MONO_EC__H__
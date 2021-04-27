// Maintainer: jproc
#ifndef __PAIR_EVALUATOR_LJGAUSS_H__
#define __PAIR_EVALUATOR_LJGAUSS_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairLJGauss.h
    \brief Defines the pair evaluator class for Lennard Jones Gaussian potentials
    \details .....
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the LJGauss pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>LJ Gauss specifics</b>

    EvaluatorPairLJGauss evaluates the function:
    \f V_{\mathrm{gauss}}(r) = \frac{1}{r^{12}} - \frac{2}{r^{6}} - \epsilon e^{- \frac{\left(r - r_{0}\right)^{2}}{2 \sigma^{2}}} \f]

*/
class EvaluatorPairLJGauss
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            Scalar epsilon;
            Scalar sigma2;
            Scalar r0;
            
            #ifdef ENABLE_HIP
            //! Set CUDA memory hints
            void set_memory_hint() const
                {
                // default implementation does nothing
                }
            #endif
            
            #ifndef __HIPCC__
            param_type() :
                epsilon(0), sigma2(0), r0(0) {}

            param_type(pybind11::dict v)
                {
                epsilon = v["epsilon"].cast<Scalar>();
                sigma2 = v["sigma2"].cast<Scalar>();
                r0 = v["r0"].cast<Scalar>();
                }

            pybind11::dict asDict()
                {
                pybind11::dict v;
                v["epsilon"] = epsilon;
                v["sigma2"] = sigma2;
                v["r0"] = r0;
                return v;
                }
            #endif
            };
 
        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairLJGauss(Scalar _rsq, 
                                    Scalar _rcutsq, 
                                    const param_type& _params)
            : rsq(_rsq), 
              rcutsq(_rcutsq),
              epsilon(_params.epsilon),
              sigma2(_params.sigma2),
              r0(_params.r0)
            {
            }

        //! LJGauss doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJGauss doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        
        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Get the number of alchemical parameters
        static const unsigned int num_alchemical_parameters = 3;

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff

            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                // const Scalar sqrt_2pi= Scalar(2.0) * M_SQRT2 / M_2_SQRTPI;
                Scalar r = fast::sqrt(rsq);
                Scalar rdiff = r - r0;
                Scalar rdiff_sigma2 = rdiff / sigma2;
                Scalar exp_val = fast::exp(-Scalar(0.5) * rdiff_sigma2 * rdiff);
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;

                force_divr = (r2inv * r6inv * Scalar(12.0) * (r6inv - Scalar(1.0))) - (exp_val * epsilon * rdiff_sigma2 / r);
                pair_eng = r6inv * (r6inv - Scalar(2.0)) - exp_val * epsilon;

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= rcut6inv * (rcut6inv - Scalar(2.0)) - (epsilon * fast::exp(-Scalar(1.0)/Scalar(2.0) * (rcutsq - r0) / sigma2));
                    }
                return true;
                }
            else
                return false;
            }

        DEVICE void AlchemParams(Scalar* alphas)
            {
            epsilon *= alphas[0];
            sigma2 *= alphas[1]*alphas[1];
            r0 *= alphas[2];
            }

        DEVICE void evalDAlphaEnergy(Scalar* dalchem_pair_energy, Scalar* alphas)
            {
                {
                    
                Scalar r = fast::sqrt(rsq);
                // Scalar sigma = fast::sqrt(sigma2);
                Scalar inva1  = 1.0/alphas[1];
                Scalar invsiga1sq  = inva1*inva1*(1/sigma2);
                Scalar rdiff  = r - alphas[2]*r0;
                Scalar rdiffsq  = rdiff*rdiff;
                Scalar exp_term  = fast::exp(-Scalar(0.5)*rdiffsq*invsiga1sq);
                Scalar c = alphas[0] * epsilon * exp_term * invsiga1sq;
                dalchem_pair_energy[0]=-epsilon * exp_term;
                dalchem_pair_energy[1]=c*rdiffsq*inva1;
                dalchem_pair_energy[2]=c*r0*rdiff;
                }
            }


        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("lj_gauss");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar epsilon; //!< epsilon parameter extracted from the params passed to the constructor
        Scalar sigma2;   //!< sigma^2 parameter extracted from the params passed to the constructor
        Scalar r0;       //!< r0 prarameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_LJGAUSS_H__
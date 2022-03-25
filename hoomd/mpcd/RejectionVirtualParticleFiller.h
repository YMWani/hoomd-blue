
#ifndef MPCD_REJECTION_FILLER_H_
#define MPCD_REJECTION_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to MPCD particle data for "sphere" geometry
/*!
 * <detailed description needed>
*/

class PYBIND11_EXPORT RejectionVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
        RejectionVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                       Scalar density,
                                       unsigned int type,
                                       std::shared_ptr<::Variant> T,
                                       unsigned int seed,
                                       std::shared_ptr<const Geometry> geom);

        virtual ~RejectionVirtualParticleFiller();

        //! Get the streaming geometry
        std::shared_ptr<const Geometry> getGeometry()
            {
            return m_geom;
            }

        //! Set the streaming geometry
        void setGeometry(std::shared_ptr<const Geometry> geom)
            {
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const Geometry> m_geom;

        //! Fill the particles outside the confinement
        virtual void fill();
    };

namespace detail
{
//! Export RejectionVirtualParticleFiller to python
void export_RejectionVirtualParticleFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_FILLER_H_
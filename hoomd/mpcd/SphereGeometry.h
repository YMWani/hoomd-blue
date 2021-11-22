
#ifndef MPCD_SPHERE_GEOMETRY_H_
#define MPCD_SPHERE_GEOMETRY_H_

#include "BoundaryCondition.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#include <cmath>

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // NVCC

namespace mpcd
{
namespace detail
{
//! Spherical droplet geometry
/*!
 * TODO: write proper description of what is being done in the class below
 */
class __attribute__((visibility("default"))) SphereGeometry
    {
    public:
        //! Constructor
        /*!
         * \param R confinement radius
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SphereGeometry(Scalar R, boundary bc)
            : m_R(R), m_R2(R*R), m_bc(bc)
            { }

        //! Detect collision between the particle and the boundary
        /*!
         * \param pos Proposed particle position
         * \param vel Proposed particle velocity
         * \param dt Integration time remaining
         *
         * \returns True if a collision occurred, and false otherwise
         *
         * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel is updated
         *       according to the appropriate bounce back rule, and the integration time \a dt is decreased to the
         *       amount of time remaining.
         */
        HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
            {

            /*
             * Detect if particle has left the spherical confinement.
             * comparison: 'r2 <= R*R' used in the following calculations is equal to 0(FALSE) if the particle is outside
             * a sphere of radius R, 1(TRUE) if inside.
             */
            Scalar r2 = dot(pos,pos);
            // exit immediately if no collision is found.
            /*
            To avoid division by zero error later on, we exit immediately. (no collision could have occurred
            if the particle speed is equal to zero)
            */
            Scalar v2 = dot(vel,vel);
            if (r2 <= R2 || v2 == Scalar(0))
               {
               dt = Scalar(0);
               return false;
               }

            // Find the point of contact when the particle is just leaving the surface
            /*
            * Calculating the time spent outside-bounds requires the knowledge of the point of contact on the boundary.
            * The point of contact can be calculated using geometrical considerations.
            *
            * We know the following quantities,
            *    1. r(t+del_t)      (the point outside the sphere)
            *    2. v(t)            (previous velocity)
            *
            * Assuming 'r*' is the point of contact on the spherical shell and dt is the time particle travelled
            * outside the boundary.
            *
            * r* = r(t+del_t) - dx* * v(t)/|v|
            * and |r*|**2 = R**2
            *
            * Solving the above equations,
            * dx* = (r.v^) - sqrt((r.v^)**2 - r2 + R2)
            * consequently, dt = dx* / |v|
            *
            * --> dt = ((r.v) - sqrt((r.v)**2 - v2*(r2-R2)))/v2
            */

            Scalar rv = dot(pos,vel);
            dt = (rv - fast::sqrt(rv*rv-v2*(r2-R2)))/v2;

            // backtrack the particle for time dt to get to point of contact
            pos -= vel*dt;

            // update velocity according to boundary conditions
            /*
             * Let n^ be the normal unit vector ar the point of contact.
             * Therefore,
             * v_perp = (v.n^)n^        -->         ((r.v)/R2)r         [R2, since the particle is on the surface]
             * v_para = v-v_perp
             */
            if (m_bc == boundary::no_slip)
                {
                /* no-slip requires reflection of the tangential components.
                 * Radial component reflected since no penetration of the surface is necessary.
                 * This results in just flipping of all the velocity components.
                 */
                vel = -vel;
                }
            else if (m_bc == boundary::slip)
                {
                // tangential component of the velocity is unchanged.
                // Radial component reflected since no penetration of the surface is necessary.
                /*
                 * v' = -v_perp + v_para = v - 2*v_perp
                */
                const Scalar3 vperp = dot(vel,pos)*pos/R2;
                vel -= Scalar(2)*vperp;
                }
            return true;
            }

        //! Check if a particle is out of bounds
        /*!
         * \param pos Current particle position
         * \returns True if particle is out of bounds, and false otherwise
         */
        HOSTDEVICE bool isOutside(const Scalar3& pos) const
            {
            return dot(pos,pos) > R2;
            }

        //! Validate that the simulation box is large enough for the geometry
        /*!
         * \param box Global simulation box
         * \param cell_size Size of MPCD cell
         *
         * The box is large enough if the shell is padded along the radial direction, so that cells at the boundary
         * would not interact with each other via PBC.
         *
         * It would be enough to check the padding along the x,y,z directions individually as the box boundaries are
         * closest to the sphere boundary along these axes.
         */
        HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
            {
            const Scalar3 hi;
            const Scalar3 lo;
            hi.x = box.getHi().x; hi.y = box.getHi().y; hi.z = box.getHi().z;
            lo.x = box.getLo().x; lo.y = box.getLo().y; lo.z = box.getLo().z;

            return ((hi.x-m_R) >= cell_size && (-lo.x-m_R) >= cell_size &&
                    (hi.y-m_R) >= cell_size && (-lo.y-m_R) >= cell_size &&
                    (hi.y-m_R) >= cell_size && (-lo.y-m_R) >= cell_size );
            }

        //! Get Sphere radius
        /*!
         * \returns confinement radius
         */
        HOSTDEVICE Scalar getR() const
            {
            return m_R;
            }

        //! Get the wall boundary condition
        /*!
         * \returns Boundary condition at wall
         */
        HOSTDEVICE boundary getBoundaryCondition() const
            {
            return m_bc;
            }

        #ifndef NVCC
        //! Get the unique name of this geometry
        static std::string getName()
            {
            return std::string("Sphere");
            }
        #endif // NVCC

    private:
        const Scalar m_R;       //!< Spherical confinement radius
        const boundary m_bc;    //!< Boundary condition
    };

} // end namespace detail
} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_SPHERE_GEOMETRY_H_

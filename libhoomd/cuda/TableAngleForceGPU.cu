/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: phillicl

#include "TableAngleForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// SMALL a relatively small number
#define SMALL 0.001f

/*! \file TableAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the table angle forces. Used by TableAngleForceComputeGPU.
*/


//! Texture for reading table values
scalar2_tex_t tables_tex;

/*!  This kernel is called to calculate the table angle forces on all triples this is defined or

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param alist List of angles stored on the GPU
    \param pitch Pitch of 2D angle list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param n_angle_type number of angle types
    \param d_tables Tables of the potential and force
    \param table_value index helper function
    \param delta_th angle delta of the table

    See TableAngleForceCompute for information on the memory layout.

    \b Details:
    * Table entries are read from tables_tex. Note that currently this is bound to a 1D memory region. Performance tests
      at a later date may result in this changing.
*/
__global__ void gpu_compute_table_angle_forces_kernel(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim box,
                                     const uint4 *alist,
                                     const unsigned int pitch,
                                     const unsigned int *n_angles_list,
                                     const Scalar2 *d_tables,
                                     const Index2D table_value,
                                     const Scalar delta_th)
    {


    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles =n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet
   

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
    
    Scalar fab[3], fcb[3];    
    
    // initialize the virial tensor to 0
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        uint4 cur_angle = alist[pitch*angle_idx + idx];

        int cur_angle_x_idx = cur_angle.x;
        int cur_angle_y_idx = cur_angle.y;
        int cur_angle_type = cur_angle.z;
        int cur_angle_abc = cur_angle.w;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = a_pos - b_pos;
        Scalar3 dcb = c_pos - b_pos;
        Scalar3 dac = a_pos - c_pos;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        Scalar rsqab = dot(dab, dab);
        Scalar rab = sqrtf(rsqab);
        Scalar rsqcb = dot(dcb, dcb);
        Scalar rcb = sqrtf(rsqcb);

        Scalar c_abbc = dot(dab, dcb);
        c_abbc /= rab*rcb;

        if (c_abbc > 1.0f) c_abbc = 1.0f;
        if (c_abbc < -1.0f) c_abbc = -1.0f;

        Scalar s_abbc = sqrtf(1.0f - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0f/s_abbc;

        // actually calculate the force
        Scalar theta = acosf(c_abbc);


        // precomputed term
        Scalar value_f = theta / delta_th;

        // compute index into the table and read in values
        unsigned int value_i = floor(value_f);
        Scalar2 VT0 = texFetchScalar2(d_tables, tables_tex, table_value(value_i, cur_angle_type));
        Scalar2 VT1 = texFetchScalar2(d_tables, tables_tex, table_value(value_i+1, cur_angle_type));
        // unpack the data
        Scalar V0 = VT0.x;
        Scalar V1 = VT1.x;
        Scalar T0 = VT0.y;
        Scalar T1 = VT1.y;

        // compute the linear interpolation coefficient
        Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and T;
        Scalar V = V0 + f * (V1 - V0);
        Scalar T = T0 + f * (T1 - T0);
        
        
        Scalar a = T * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;

        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;

        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = V*Scalar(1.0f/3.0f);

        // symmetrized version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1./3.)*(dab.x*fab[0] + dcb.x*fcb[0]);
        angle_virial[1] = Scalar(1./6.)*(dab.x*fab[1] + dcb.x*fcb[1]
                                      + dab.y*fab[0] + dcb.y*fcb[0]);
        angle_virial[2] = Scalar(1./6.)*(dab.x*fab[2] + dcb.x*fcb[2]
                                      + dab.z*fab[0] + dcb.z*fcb[0]);
        angle_virial[3] = Scalar(1./3.)*(dab.y*fab[1] + dcb.y*fcb[1]);
        angle_virial[4] = Scalar(1./6.)*(dab.y*fab[2] + dcb.y*fcb[2]
                                      + dab.z*fab[1] + dcb.z*fcb[1]);
        angle_virial[5] = Scalar(1./3.)*(dab.z*fab[2] + dcb.z*fcb[2]);


        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force_idx;
    for (unsigned int i = 0; i < 6 ; i++)
        d_virial[i*virial_pitch + idx] = virial[i];
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param alist List of angles stored on the GPU
    \param pitch Pitch of 2D angle list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param n_angle_type number of angle types
    \param d_tables Tables of the potential and force
    \param table_width Number of points in each table    
    \param table_value indexer helper
    \param block_size Block size at which to run the kernel

    \note This is just a kernel driver. See gpu_compute_table_angle_forces_kernel for full documentation.
*/
cudaError_t gpu_compute_table_angle_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim &box,
                                     const uint4 *alist,
                                     const unsigned int pitch,
                                     const unsigned int *n_angles_list,
                                     const Scalar2 *d_tables,
                                     const unsigned int table_width,
                                     const Index2D &table_value,
                                     const unsigned int block_size)
    {
    assert(d_tables);
    assert(table_width > 1);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the tables texture
    tables_tex.normalized = false;
    tables_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, tables_tex, d_tables, sizeof(Scalar2) * table_value.getNumElements());
    if (error != cudaSuccess)
        return error;

    Scalar delta_th = M_PI/(table_width - 1.0f);
    
    gpu_compute_table_angle_forces_kernel<<< grid, threads >>>
            (d_force,
             d_virial,
             virial_pitch,
             N,
             d_pos,
             box,
             alist,
             pitch,
             n_angles_list,
             d_tables,
             table_value,
             delta_th);

    return cudaSuccess;
    }

// vim:syntax=cpp

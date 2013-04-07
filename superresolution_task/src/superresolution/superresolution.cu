/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    superresolution.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/

/*
 * superresolution.cu
 *
 *  Created on: May 16, 2012
 *      Author: steinbrf
 */
#include "superresolution.cuh"
#include <stdio.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include <auxiliary/cuda_basic.cuh>
#include <vector>
#include <list>
#include <sys/time.h>

timeval startFirstGauss, endFirstGauss;

#include <sstream>
#include <string.h>

//#include <linearoperations.cuh>
#include <linearoperations/linearoperations.cuh>
#include "superresolution_definitions.h"
#include <auxiliary/debug.hpp>

#ifdef DGT400
#define SR_BW 32
#define SR_BH 16
#else
#define SR_BW 16
#define SR_BH 16
#endif

//shared mem flags
#define SHARED_MEM 0
#define BACKWARDSWARPING_VALUE_TEXTURE_MEM 1
#define GAUSS_MEMORY 0 // 0 = global memory, 1 = shared memory, 2 = texture memory

#define TIME_DEBUG 0
#define IMAGE_DEBUG 0

#include <linearoperations/linearoperations.h>

#if IMAGE_DEBUG
	#define IMAGE_DEBUG_IMAGE 20
	int IMAGE_DEBUG_CALL_NUM = 0;
	char cudaDebug[128];
#endif

extern __shared__ float smem[];

//TODO write comment
// global memory version of dualL1Difference
__global__ void dualL1Difference_gm
(
	const float *primal,
	const float *constant,
	float *dual,
	int nx,
	int ny,
	int pitch,
	float factor_update,
	float factor_clipping,
	float huber_denom,
	float tau_d
)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < nx && y < ny)
	{
		int idx = x + pitch * y;
		
		float dualTemp = (dual[idx] + tau_d * factor_update * (primal[idx] - constant[idx])) / huber_denom;
		
		if( dualTemp < -factor_clipping)
		{
			dual[idx] = -factor_clipping;
		}
		else if( dualTemp > factor_clipping)
		{
			dual[idx] = factor_clipping;
		}
		else
		{
			dual[idx] = dualTemp;
		}
	}
}

//TODO write comment
// shared memory version of dualL1Difference
__global__ void dualL1Difference_sm
	(
		const float *primal,
		const float *constant,
		float *dual,
		int nx,
		int ny,
		int pitch,
		float factor_update,
		float factor_clipping,
		float huber_denom,
		float tau_d
    )
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	__shared__ float s_dual[SR_BW][SR_BH];
	
	int idx = y * pitch + x;
	
	// loading the data in shared memory
	// NOTE: not using shared memory for primal & constant reduces the execution time 
	// from 127 micro sec to 108 micro sec on 48 core GPU
	if (x < nx && y < ny)//guards
	{
		s_dual[tx][ty] = dual[idx];
	}
	
	__syncthreads();
	
	if( x < nx && y < ny )//guards
	{
		// compute
		s_dual[tx][ty] =( s_dual[tx][ty] + tau_d * factor_update * 
						( primal[idx] - constant[idx])) / huber_denom;		
		
		if( s_dual[tx][ty] < -factor_clipping)
		{
			dual[idx] = -factor_clipping;
		}
		else if ( s_dual[tx][ty] > factor_clipping)
		{
			dual[idx] = factor_clipping;
		}
		else
		{
			dual[idx] = s_dual[tx][ty];
		}
	}
}

//TODO write comment
//global memory version of primal1N
__global__ void primal1N_gm
	(
		const float *xi1,
		const float *xi2,
		const float *degraded,
		float *u,
		float *uor,
		int nx,
		int ny,
		int pitch,
		float factor_tv_update,
		float factor_degrade_update,
		float tau_p,
		float overrelaxation
	)
{
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

	if( x < nx && y < ny )
	{
		const int idx = y * pitch + x;

		float u_old = u[idx];

		float u_new = u_old + tau_p * 
			(
				factor_tv_update *
				(xi1[idx] - ( x == 0 ? 0.0f : xi1[idx - 1] ) + xi2[idx] - ( y == 0 ? 0.0f : xi2[idx - pitch] )) -
				factor_degrade_update * degraded[idx]
			);

		// write back to output image
		u[idx] = u_new;
		uor[idx] = overrelaxation * u_new + (1.0f - overrelaxation) * u_old;
	}
}

__global__ void primal1N_sm
(
		const float *xi1,
		const float *xi2,
		const float *degraded,		// temp3_g
		float *u,
		float *uor,
		int nx,
		int ny,
		int pitch,
		float factor_tv_update,
		float factor_degrade_update,
		float tau_p,
		float overrelaxation
)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	int idx = y * pitch + x;
	
	__shared__ float s_xi1[SR_BW+1][SR_BH];
	__shared__ float s_xi2[SR_BW][SR_BH+1];

	// loading data to shared memory
	if (x < nx && y < ny)
	{
		s_xi1[tx+1][ty] = xi1[idx];
		s_xi2[tx][ty+1] = xi2[idx];
				
		s_xi1[0][ty] = 0.0f; // removing the loop for x == 0 brings slight performance improvement
		
		if( threadIdx.x == 0)
		{
			s_xi1[0][ty] = xi1[idx-1];
		}
		
		s_xi2[tx][0] = 0.0f; // removing the loop for y == 0 brings slight performance improvement
		
		if( threadIdx.y == 0 )
		{
			s_xi2[tx][0] = xi2[idx-pitch];
		}
	}
	
	__syncthreads();
	
	if (x < nx && y < ny)
	{		
		float u_old = u[idx];
	
		// change of indices for xi1 & xi2 due to the way shared memory copying is done !
		// produces, correct results!
		float u_new = u_old + tau_p * ( factor_tv_update * 
								( s_xi1[tx+1][ty] - s_xi1[tx][ty] + s_xi2[tx][ty+1] - s_xi2[tx][ty]) -
								factor_degrade_update * degraded[idx]);
		
		// write back to output image
		u[idx] = u_new;
		uor[idx] = overrelaxation * u_new + (1.0f - overrelaxation) * u_old;		
	}
}


//TODO write comment
// global memory version of primal1N
__global__ void dualTVHuber_gm
	(
		float 	*uor_g,								// Field of overrelaxed primal variables
		float 	*xi1_g, 							// Dual Variable for TV regularization in X direction
		float 	*xi2_g,								// Dual Variable for TV regularization in Y direction
		int   	nx,									// New High-Resolution Width
		int   	ny,									// New High-Resolution Height
		int   	pitchf1,							// GPU pitch (padded width) of the superresolution high-res fields
		float   factor_update,
		float   factor_clipping,
		float   huber_denom,
		float   tau_d
	)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < nx && y < ny ) // guards
	{
		int x1 = x + 1;
		if( x1 >= nx ){	x1 = nx-1; }	// at x boundary

		int y1 = y+1;
		if( y1 >= ny ){ y1 = ny-1; }	// at y boundary

		// do xi1_g, xi2_g & uor_g have same pitch ? confirm - YES
		const int p = y * pitchf1 + x;

		float dx = (xi1_g[p] + tau_d * factor_update * (uor_g[y*pitchf1+x1] - uor_g[p])) / huber_denom;
		float dy = (xi2_g[p] + tau_d * factor_update * (uor_g[y1*pitchf1+x] - uor_g[p])) / huber_denom;
		float denom = sqrtf( dx * dx + dy * dy ) / factor_clipping;

		if(denom < 1.0f) denom = 1.0f;
		xi1_g[p] = dx / denom;
		xi2_g[p] = dy / denom;
	}
}

//TODO write comment
// shared memory version of primal1N
__global__ void dualTVHuber_sm
(
		float 	*uor_g,								// Field of overrelaxed primal variables
		float 	*xi1_g, 							// Dual Variable for TV regularization in X direction
		float 	*xi2_g,								// Dual Variable for TV regularization in Y direction
		int   	nx,									// New High-Resolution Width
		int   	ny,									// New High-Resolution Height
		int   	pitchf1,							// GPU pitch (padded width) of the superresolution high-res fields
		float   factor_update,
		float   factor_clipping,
		float   huber_denom,
		float   tau_d
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ float uor[SR_BW+1][SR_BH+1];
	
	int idx = y * pitchf1 + x;

	// load data into shared memory
	// NOTE: not using shared memory for xi1 & xi2 reduces execution time from
	// 420 to 403 micro sec on 48 core GPU
	if( x < nx && y < ny ) // guards
	{
		uor[tx][ty] = uor_g[idx];
				
		if( x == nx -1 )
		{
			uor[tx+1][ty] = uor[tx][ty];
		}
		else if( threadIdx.x == SR_BW - 1 )
		{
			uor[tx+1][ty] = uor_g[idx+1];
		}
		
		if( y == ny -1 )
		{
			uor[tx][ty+1] = uor[tx][ty];
		}
		else if( threadIdx.y == SR_BH -1 )
		{
			uor[tx][ty+1] = uor_g[idx+pitchf1];
		}
	}
	
	__syncthreads();
	
	if(x < nx && y < ny)// guards
	{
		// compute
		float dx = (xi1_g[idx] + tau_d * factor_update * (uor[tx+1][ty] - uor[tx][ty])) / huber_denom;
		float dy = (xi2_g[idx] + tau_d * factor_update * (uor[tx][ty+1] - uor[tx][ty])) / huber_denom;
		
		float denom = sqrtf( dx * dx + dy * dy ) / factor_clipping;
		
		if(denom < 1.0f) denom = 1.0f;
		xi1_g[idx] = dx / denom;
		xi2_g[idx] = dy / denom;
	}
}

void computeSuperresolutionUngerGPU
(
		float *xi1_g, 							// Dual Variable for TV regularization in X direction
		float *xi2_g,							// Dual Variable for TV regularization in Y direction
		float *temp1_g,							// Helper array
		float *temp2_g,
		float *temp3_g,
		float *temp4_g,
		float *uor_g,							// Field of overrelaxed primal variables
		float *u_g,								// GPU memory for the result image
		std::vector<float*> &q_g,				// Dual variables for L1 difference penalization
		std::vector<float*> &images_g,			// Dual variables for L1 difference penalization
		std::list<FlowGPU> &flowsGPU,			// GPU memory for the displacement fields
												//   class FlowGPU { void clear(); float *u_g; float *v_g; int nx; int ny; }
		int   &nx,								// New High-Resolution Width
		int   &ny,								// New High-Resolution Height
		int   &pitchf1,							// GPU pitch (padded width) of the superresolution high-res fields
		int   &nx_orig,							// Original Low-Resolution Width
		int   &ny_orig,							// Original Low-Resolution Height
		int   &pitchf1_orig,					// GPU pitch (padded width) of the original low-res images
		int   &oi,								// Number of Iterations
		float &tau_p,							// Primal Update Step Size
		float &tau_d,							// Dual Update Step Size
		float &factor_tv,						// The weight of Total Variation Penalization
		float &huber_epsilon,					// Parameter for Huber norm regularization
		float &factor_rescale_x,				// High-Resolution Width divided by Low-Resolution Width
		float &factor_rescale_y,				// High-Resolution Height divided by Low-Resolution Height
		float &blur,							// The amount of Gaussian Blur present in the degrading process
		float &overrelaxation,					// Overrelaxation parameter in the range of [1,2]
		int   debug								// Debug Flag, if activated the class produces Debug output.
)
{
    #if TIME_DEBUG
		struct timespec start, end ,gauss1TimeTS1,gauss1TimeTS2,gauss2TimeTS,forwardTimeTS1,forwardTimeTS2,backTimeTS1,backTimeTS2
		                ,resampleTimeTS1,resampleTimeTS2;
		double   gauss1Time, gauss2Time,forwardTime, overallTime,backwardTime, resample1Time;
		clock_gettime(CLOCK_MONOTONIC,  &start);
		fprintf(stderr,"\n computeSuperresolutionUngerGPU start ...");
    #endif

	// grid and block dimensions
	int gridsize_x = ((nx - 1) / SR_BW) + 1;
	int gridsize_y = ((ny - 1) / SR_BH) + 1;
	dim3 dimGrid ( gridsize_x, gridsize_y );
	dim3 dimBlock ( SR_BW, SR_BH );

	// initialise xi1_g and xi2_g to zero
	setKernel <<<dimGrid, dimBlock>>>( xi1_g, nx, ny, pitchf1, 0.0f );
	setKernel <<<dimGrid, dimBlock>>>( xi2_g, nx, ny, pitchf1, 0.0f );

	// initialise u_g and uor_g to 64.0f
	setKernel <<<dimGrid, dimBlock>>>( u_g,   nx, ny, pitchf1, 64.0f );
	setKernel <<<dimGrid, dimBlock>>>( uor_g, nx, ny, pitchf1, 64.0f );

	// initialise all elements of q_g to zero
	for(unsigned int k = 0; k < q_g.size(); ++k )
	{
		setKernel <<<dimGrid, dimBlock>>>( q_g[k], nx_orig, ny_orig, pitchf1_orig, 0.0f );
	}

	float factorquad              = factor_rescale_x * factor_rescale_x * factor_rescale_y * factor_rescale_y;
	float factor_degrade_update   = pow( factorquad, CLIPPING_TRADEOFF_DEGRADE_1N );

	float factor_degrade_clipping = factorquad / factor_degrade_update;
	float huber_denom_degrade     = 1.0f + huber_epsilon * tau_d / factor_degrade_clipping;

	float factor_tv_update        = pow( factor_tv, CLIPPING_TRADEOFF_TV );
	float factor_tv_clipping      = factor_tv / factor_tv_update;
	float huber_denom_tv          = 1.0f + huber_epsilon * tau_d / factor_tv;

	for( int i = 0; i < oi; ++i )
	{
		// calculate dual tv
		#if SHARED_MEM
			dualTVHuber_sm<<<dimGrid,dimBlock>>>
				( uor_g, xi1_g, xi2_g, nx, ny, pitchf1, factor_tv_update, factor_tv_clipping, huber_denom_tv, tau_d );
		#else
			dualTVHuber_gm<<<dimGrid,dimBlock>>>
				( uor_g, xi1_g, xi2_g, nx, ny, pitchf1, factor_tv_update, factor_tv_clipping, huber_denom_tv, tau_d );
		#endif

		#if IMAGE_DEBUG
			if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 )
			{
				fprintf( stderr, "\nSaving uor..." );
				snprintf( cudaDebug, 128, "debug/superresolution/%02d_02_uor_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
				saveCudaImage( cudaDebug, uor_g, nx, ny, pitchf1, 1 );
			}
		#endif

		// DUAL DATA
		// iterating over all images
		std::vector<float*>::iterator image = images_g.begin();
		std::list<FlowGPU>::iterator flow   = flowsGPU.begin();
		for( unsigned int k = 0; image != images_g.end() && flow != flowsGPU.end() && k < q_g.size(); ++k, ++flow, ++image )
		{
				#if TIME_DEBUG
			    clock_gettime(CLOCK_MONOTONIC,  &backTimeTS1);
				#endif

				// call backward warping
				#if BACKWARDSWARPING_VALUE_TEXTURE_MEM
				backwardRegistrationBilinearValueTex ( uor_g, flow->u_g, flow->v_g, temp1_g, 0.0f, nx, ny, pitchf1, pitchf1, 1.0f, 1.0f );
				#else
				backwardRegistrationBilinearValueTex_gm ( uor_g, flow->u_g, flow->v_g, temp1_g, 0.0f, nx, ny, pitchf1, pitchf1, 1.0f, 1.0f );
				#endif

				#if TIME_DEBUG
					clock_gettime(CLOCK_MONOTONIC,  &backTimeTS2);
					backwardTime = (backTimeTS2.tv_sec - backTimeTS1.tv_sec) + (double) (backTimeTS2.tv_nsec - backTimeTS1.tv_nsec) * 1e-9;
			    #endif

                #if IMAGE_DEBUG
					if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
					{
						fprintf( stderr, "\nSaving uor backwarped..." );
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_03_uor_backwarped_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug, temp1_g, nx, ny, pitchf1, 1 );

						fprintf( stderr, "\nSaving input image..." );
						float *f = (float*)(*image);
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_04_input_image_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug, f, nx_orig, ny_orig, pitchf1_orig, 1 );
					}
			    #endif

				if( blur > 0.0f )
				{
					#if TIME_DEBUG
					clock_gettime(CLOCK_MONOTONIC,  &gauss1TimeTS1);
					#endif

					// blur image
					// gauss with global memory
					gaussBlurSeparateMirrorGpu_gm ( temp1_g, temp2_g, nx, ny, pitchf1, blur, blur, (int)(3.0f * blur), temp4_g, 0 );

					#if TIME_DEBUG
						clock_gettime(CLOCK_MONOTONIC,  &gauss1TimeTS2);
						gauss1Time = (gauss1TimeTS2.tv_sec - gauss1TimeTS1.tv_sec) + (double) (gauss1TimeTS2.tv_nsec - gauss1TimeTS1.tv_nsec) * 1e-9;
					#endif

					#if IMAGE_DEBUG
					if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
					{
						fprintf( stderr, "\nSaving uor blurred..." );
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_05_uor_blurred_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug, temp2_g, nx, ny, pitchf1, 1 );
					}
					#endif
				}
				else
				{
					// swap the helper array pointers
					float *temp = temp1_g;
					temp1_g = temp2_g;
					temp2_g = temp;
				}

				if( factor_rescale_x > 1.0f || factor_rescale_y > 1.0f )
				{
					
					#if TIME_DEBUG
						clock_gettime(CLOCK_MONOTONIC,  &resampleTimeTS1);
					#endif

					// call resample
					resampleAreaParallelSeparate(
							temp2_g,			// input image
							temp1_g,			// output image
							nx, ny,				// input size
							pitchf1,			// input pitch
							nx_orig, ny_orig,	// output size
							pitchf1_orig,		// output pitch
							temp4_g				// helper array
						);

					#if TIME_DEBUG
						clock_gettime(CLOCK_MONOTONIC,  &resampleTimeTS2);
						resample1Time = (resampleTimeTS2.tv_sec - resampleTimeTS1.tv_sec) + (double) (resampleTimeTS2.tv_nsec - resampleTimeTS1.tv_nsec) * 1e-9;
					#endif

					#if IMAGE_DEBUG
						if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
						{
							fprintf( stderr, "\nSaving uor resampled..." );
							snprintf( cudaDebug, 128, "debug/superresolution/%02d_06_uor_resampled_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
							saveCudaImage( cudaDebug, temp1_g, nx_orig, ny_orig, pitchf1_orig, 1 );
						}
					#endif
				}
				else
				{
					// swap the helper array pointers
					float *temp = temp1_g;
					temp1_g = temp2_g;
					temp2_g = temp;
				}


				#if SHARED_MEM
					dualL1Difference_sm<<<dimGrid,dimBlock>>>
								( temp1_g, *image, q_g[k], nx_orig, ny_orig, pitchf1_orig ,
								  factor_degrade_update, factor_degrade_clipping, huber_denom_degrade, tau_d);
				#else
					dualL1Difference_gm<<<dimGrid, dimBlock>>>
								( temp1_g, *image, q_g[k], nx_orig, ny_orig, pitchf1_orig,
								  factor_degrade_update, factor_degrade_clipping, huber_denom_degrade, tau_d);
				#endif

				#if IMAGE_DEBUG
					if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
					{
						fprintf( stderr, "\nSaving difference image..." );
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_07_difference_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug,  q_g[k], nx_orig, ny_orig, pitchf1_orig, 1 );
					}
				#endif
		}

		// set 3rd helper array to zero
		setKernel <<<dimGrid, dimBlock>>>( temp3_g, nx, ny, pitchf1, 0.0f );

		// iterating over all images
		image = images_g.begin();
		flow   = flowsGPU.begin();
		for( unsigned int k = 0; image != images_g.end() && flow != flowsGPU.end() && k < q_g.size(); ++k, ++flow, ++image )
		{
			if( factor_rescale_x > 1.0f || factor_rescale_y > 1.0f )
			{
				resampleAreaParallelSeparateAdjoined( q_g[k], temp1_g, nx_orig, ny_orig, pitchf1_orig, nx, ny, pitchf1, temp4_g );

				#if IMAGE_DEBUG
					if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
					{
						fprintf( stderr, "\nSaving difference resampled..." );
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_08_difference_resampled_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug, temp1_g, nx, ny, pitchf1, 1 );
					}
				#endif
			}
			else
			{
				// copy q_g[k] to temp1_g
				cudaMemcpy( temp1_g, q_g[k], ny * pitchf1, cudaMemcpyDeviceToDevice );
			}

			if( blur > 0.0f )
			{
				// blur image
				gaussBlurSeparateMirrorGpu_gm ( temp1_g, temp2_g, nx, ny, pitchf1, blur, blur, (int)(3.0f * blur), temp4_g, 0 );

				#if IMAGE_DEBUG
					if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
					{
						fprintf( stderr, "\nSaving difference blurred..." );
						snprintf( cudaDebug, 128, "debug/superresolution/%02d_09_difference_blurred_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
						saveCudaImage( cudaDebug, temp2_g, nx, ny, pitchf1, 1 );
					}
				#endif
			}
			else
			{
				// swap the helper array pointers
				float *temp = temp1_g;
				temp1_g = temp2_g;
				temp2_g = temp;
			}

			#if TIME_DEBUG
				clock_gettime(CLOCK_MONOTONIC,  &forwardTimeTS1);
			#endif

			// foreward warping
			forewardRegistrationBilinearAtomic (
					flow->u_g, flow->v_g,
					temp2_g, temp1_g,
					nx, ny,
					pitchf1
				);

			#if TIME_DEBUG
				clock_gettime(CLOCK_MONOTONIC,  &forwardTimeTS2);
				forwardTime = (forwardTimeTS2.tv_sec - forwardTimeTS2.tv_sec) + (double) (forwardTimeTS2.tv_nsec - forwardTimeTS1.tv_nsec) * 1e-9;
	    	#endif

			#if IMAGE_DEBUG
				if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 && k == 0 )
				{
					fprintf( stderr, "\nSaving difference foreward warped..." );
					snprintf( cudaDebug, 128, "debug/superresolution/%02d_10_difference_foreward_warped_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
					saveCudaImage( cudaDebug, temp1_g, nx, ny, pitchf1, 1 );
				}
			#endif

			// add 1st to 3rd helper array
			addKernel<<<dimGrid, dimBlock>>>( temp1_g, temp3_g, nx, ny, pitchf1 );
		}

		#if IMAGE_DEBUG
			if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 )
			{
				fprintf( stderr, "\nSaving difference sum..." );
				snprintf( cudaDebug, 128, "debug/superresolution/%02d_11_sum_differences_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
				saveCudaImage( cudaDebug, temp3_g, nx, ny, pitchf1, 1 );
			}
		#endif

		#if SHARED_MEM
		primal1N_sm<<< dimGrid, dimBlock>>>(xi1_g, xi2_g, temp3_g, u_g, uor_g, nx, ny, pitchf1, factor_tv_update, factor_degrade_update, tau_p, overrelaxation);
		#else
	    primal1N_gm<<< dimGrid, dimBlock>>>(xi1_g, xi2_g, temp3_g, u_g, uor_g, nx, ny, pitchf1, factor_tv_update, factor_degrade_update, tau_p, overrelaxation);
		#endif

		#if IMAGE_DEBUG
			if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE && i == 10 )
			{
				fprintf( stderr, "\nSaving temporary result image..." );
				snprintf( cudaDebug, 128, "debug/superresolution/%02d_12_result_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
				saveCudaImage( cudaDebug, u_g, nx, ny, pitchf1, 1 );

				fprintf( stderr, "\nSaving new uor..." );
				snprintf( cudaDebug, 128, "debug/superresolution/%02d_13_new_uor_oi_%d.png", IMAGE_DEBUG_CALL_NUM, i );
				saveCudaImage( cudaDebug, uor_g, nx, ny, pitchf1, 1 );
			}
		#endif
	}

	#if TIME_DEBUG
	fprintf(stderr,"\n computeSuperresolutionUngerGPU end ...");
    clock_gettime(CLOCK_MONOTONIC,  &end);
    overallTime = (end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) * 1e-9;

    fprintf(stderr,"\n Time for one image %f", overallTime);
    fprintf(stderr,"\n Time for backward %f", backwardTime);
    fprintf(stderr,"\n Time for forward %f", forwardTime);
    fprintf(stderr,"\n Time for gauss 1 %f",gauss1Time);
    fprintf(stderr,"\n Time for resample 1 %f",resample1Time);
    
	#endif

	#if IMAGE_DEBUG
		if( IMAGE_DEBUG_CALL_NUM == IMAGE_DEBUG_IMAGE )
		{
			fprintf( stderr, "\nSaving result image..." );
			snprintf( cudaDebug, 128, "debug/superresolution/%02d_14_result.png", IMAGE_DEBUG_CALL_NUM );
			saveCudaImage( cudaDebug, u_g, nx, ny, pitchf1, 1 );
		}
	#endif

	#if IMAGE_DEBUG
		++IMAGE_DEBUG_CALL_NUM;
	#endif

}

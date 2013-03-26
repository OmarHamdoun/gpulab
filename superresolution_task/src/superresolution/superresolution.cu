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

#include <linearoperations/linearoperations.h>


extern __shared__ float smem[];

void computeSuperresolutionUngerGPU
(
		float *xi1_g, 							//! Dual Variable for TV regularization in X direction
		float *xi2_g,							//! Dual Variable for TV regularization in X direction
		float *temp1_g,							//! Helper array
		float *temp2_g,
		float *temp3_g,
		float *temp4_g,
		float *uor_g,							//! Field of overrelaxed primal variables
		float *u_g,								//! GPU memory for the result image
		std::vector<float*> &q_g,				//! Dual variables for L1 difference penalization
		std::vector<float*> &images_g,			//! Dual variables for L1 difference penalization
		std::list<FlowGPU> &flowsGPU,			//! GPU memory for the displacement fields
		int   &nx,								//! New High-Resolution Width
		int   &ny,								//! New High-Resolution Height
		int   &pitchf1,							//! GPU pitch (padded width) of the superresolution high-res fields
		int   &nx_orig,							//! Original Low-Resolution Width
		int   &ny_orig,							//! Original Low-Resolution Height
		int   &pitchf1_orig,					//! GPU pitch (padded width) of the original low-res images
		int   &oi,								//! Number of Iterations
		float &tau_p,							//! Primal Update Step Size
		float &tau_d,							//! Dual Update Step Size
		float &factor_tv,						//! The weight of Total Variation Penalization
		float &huber_epsilon,					//! Parameter for Huber norm regularization
		float &factor_rescale_x,				//! High-Resolution Width divided by Low-Resolution Width
		float &factor_rescale_y,				//! High-Resolution Height divided by Low-Resolution Height
		float &blur,							//! The amount of Gaussian Blur present in the degrading process
		float &overrelaxation,					//! Overrelaxation parameter in the range of [1,2]
		int   debug								//! Debug Flag, if activated the class produces Debug output.
)
{
	//### Implement me###
	fprintf(stderr,"\nComputing 1N Superresolution from %i Images on GPU",(int)_images_original.size());

	// replacing u by u_g ( pointer to resultant data)
	
	// TODO: KERNEL TO INITIALISE xi1_g & xi2_g to 0.0f
	// TODO: KERNEL TO INITIALISE u_g & uor_g to 64.0f
	
	// TODO: KERNEL TO SET q_g TO 0.0f, LOOPS _q.size() TIMES 
	
	float factorquad = factor_rescale_x*factor_rescale_y*factor_rescale_x*factor_rescale_y;
	float factor_degrade_update = pow(factorquad,CLIPPING_TRADEOFF_DEGRADE_1N);
	
	float factor_degrade_clipping = factorquad/factor_degrade_update;
	float huber_denom_degrade = 1.0f + huber_epsilon*tau_d/factor_degrade_clipping;

	float factor_tv_update = pow(_factor_tv,CLIPPING_TRADEOFF_TV);
	float factor_tv_clipping = factor_tv/factor_tv_update;
	float huber_denom_tv = 1.0f + huber_epsilon*tau_d/_factor_tv;
	
	for(int i=0;i<_iterations;i++)
	{
		fprintf(stderr," %i",i);

		//TODO: KERNEL FOR DUAL TV
		//dualTVHuber(_u_overrelaxed,_xi1,_xi2,_nx,_ny,factor_tv_update,factor_tv_clipping,huber_denom_tv,_tau_d);

		//DUAL DATA
		unsigned int k=0;
		
		// NEED TO INITIALISE A ITERATOR FOR ORIGINAL IMAGES
		//std::vector<cv::Mat*>::iterator image = _images_original.begin();
		
		// NEED TO SET A ITERATOR FOR FLOWS
		//std::list<Flow>::iterator flow = _flows.begin();
				
		// for( k=0; k< _(while image is not the last original image)__ ; k++ ) // TODO
		{
				// TODO: KERNEL BACKWARDREGISTRATIONBILINEARVALUE
				
		
				if(blur > 0.0f)
				{
					// TODO: KERNEL FOR GAUSSBLURSEPARATEMIRROR
				}
				else
				{
					// SWAP THE HELPER ARRAY POINTERS
					float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
				}
		
				if(factor_rescale_x > 1.0f || factor_rescale_y > 1.0f)
				{
					resampleAreaParallelSeparate(temp2_g, temp1_g, nx, ny,
												pitchf1, nx_orig, ny_orig,
												pitchf1_orig, temp4_g);
				}
				else
				{
					// SWAP THE HELPER ARRAY POINTERS
					float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
				}
				
				// TODO: KERNEL FOR dualL1Difference
				
				// USE k IN FOR LOOP
				//k++; 
		}
		
		// TODO: KERNEL TO SET 3RD HELPER ARRAY TO 0.00f
		
		k=0;
		image = _images_original.begin();
		flow = _flows.begin();
		// for( k=0; k< _(while image is not the last original image)__ ; k++ ) // TODO
		{
			if(factor_rescale_x > 1.0f || factor_rescale_y > 1.0f)
			{
				// TODO: WRITE KERNEL resampleAreaParallelizableSeparateAdjoined
			}
			else
			{
				// TODO: KERNEL TO COPY q_g[k] to temp1_g
			}
			if(blur > 0.0f)
			{
				// TODO: KERNEL FOR GAUSSBLURSEPARATEMIRROR
				// lookout for change in parameters, if any
			}
			else
			{
				// SWAP THE HELPER ARRAY POINTERS
				float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
			}
			
			// TODO: IMPLEMENT forewardRegistrationBilinear
			
			// TODO: KERNEL TO ADD 1ST TO 3RD HELPER ARRAY
		}	
		
		// TODO: IMPLMENT KERNEL primal1N
}






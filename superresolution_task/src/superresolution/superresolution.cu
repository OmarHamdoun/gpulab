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

__global__ void dualL1Difference
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
    dual[idx] = (dual[idx] + tau_d * factor_update * (primal[idx] - constant[idx]))
    		    / huber_denom;
    if (dual[idx] < -factor_clipping)
    {
    	dual[idx] = -factor_clipping;
    }

    if (dual[idx] > factor_clipping)
    {
    	dual[idx] = factor_clipping;
    }
  }
}

__global__ void primal1N
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
  if (x < nx && y < ny)
  {
    const int idx = y * pitch + x;
    float u_old = u[idx];
    float u_new = u[idx] + tau_p *
        (factor_tv_update * (xi1[idx] - (x == 0 ? 0.0f : xi1[idx - 1]) + xi2[idx] - (y == 0 ? 0.0f : xi2[idx - nx]))
            - factor_degrade_update * degraded[idx]);
    u[idx] = u_new;
    uor[idx] = overrelaxation * u_new + (1.0f - overrelaxation) * u_old;
  }
}


void computeSuperresolutionUngerGPU
(
		float *xi1_g,
		float *xi2_g,
		float *temp1_g,
		float *temp2_g,
		float *temp3_g,
		float *temp4_g,
		float *uor_g,
		float *u_g,
		std::vector<float*> &q_g,
		std::vector<float*> &images_g,
		std::list<FlowGPU> &flowsGPU,
		int   &nx,
		int   &ny,
		int   &pitchf1,
		int   &nx_orig,
		int   &ny_orig,
		int   &pitchf1_orig,
		int   &oi,
		float &tau_p,
		float &tau_d,
		float &factor_tv,
		float &huber_epsilon,
		float &factor_rescale_x,
		float &factor_rescale_y,
		float &blur,
		float &overrelaxation,
		int   debug
)
{
      // set blocksize
	  dim3 blockSize(SR_BW, SR_BH);
	  dim3 gridSize(((nx % SR_BW) ? (nx / SR_BW + 1) : (nx / SR_BW)),
			  ((ny % SR_BH) ? (ny / SR_BH + 1) : (ny / SR_BH)));

	  // initialize everything
	  setKernel<<<gridSize, blockSize>>>(xi1_g, nx, ny, pitchf1, 0.0f);
	  setKernel<<<gridSize, blockSize>>>(xi2_g, nx, ny, pitchf1, 0.0f);
	  setKernel<<<gridSize, blockSize>>>(u_g, nx, ny, pitchf1, 64.0f);
	  setKernel<<<gridSize, blockSize>>>(uor_g, nx, ny, pitchf1, 64.0f);

	  float factorquad = factor_rescale_x * factor_rescale_y * factor_rescale_x * factor_rescale_y;
	  float factor_degrade_update = pow(factorquad, static_cast<float>(CLIPPING_TRADEOFF_DEGRADE_1N));
	  float factor_degrade_clipping = factorquad / factor_degrade_update;
	  float huber_denom_degrade = 1.0f + huber_epsilon * tau_d / factor_degrade_clipping;

	  float factor_tv_update = pow(factor_tv, static_cast<float>(CLIPPING_TRADEOFF_TV));
	  float factor_tv_clipping = factor_tv / factor_tv_update;
	  float huber_denom_tv = 1.0f + huber_epsilon * tau_d / factor_tv;

	  //DUAL TV
	  //TODO implement gpu dualTVHuber
	  //dualTVHuber(_u_overrelaxed,_xi1,_xi2,_nx,_ny,factor_tv_update,factor_tv_clipping,huber_denom_tv,_tau_d);

	  //DUAL DATA
	  unsigned int k=0;
	  std::vector<float*>::iterator image = images_g.begin();
	  std::list<FlowGPU>::iterator flow = flowsGPU.begin();
	  while (image != images_g.end() && flow != flowsGPU.end() && k < q_g.size())
	  {
		  float* f = *image;
		  //backwardRegistrationBilinearValue(_u_overrelaxed,_help1,flow->u1,flow->u2,0.0f,_nx,_ny,1.0f,1.0f);
		  if(blur > 0.0f)
		  {
			  //gaussBlurSeparateMirror(_help1,_help2,_nx,_ny,_blur,_blur,(int)(3.0f*_blur),_help4,0);
		  }
		  else
		  {
			  float *temp = temp1_g;
			  temp1_g = temp2_g;
			  temp2_g = temp;
		  }

		  if(factor_rescale_x > 1.0f || factor_rescale_y > 1.0f)
		  {
			  //resampleAreaParallelizableSeparate(_help2,_help1,_nx,_ny,_nx_orig,_ny_orig,_help4);
		  }
		  else
		  {
		      float *temp = temp1_g;
		      temp1_g = temp2_g;
		      temp2_g = temp;
		  }

		  //dualL1Difference(_help1,f,_q[k],_nx_orig,_ny_orig,factor_degrade_update,factor_degrade_clipping,huber_denom_degrade,_tau_d);
		  k++;
		  flow++;
		  image++;
	 }


	 //PROX
	 //setKernel<<<getGridSize(nx, ny), blockSize>>>(temp3_g, nx, ny, pitchf1, 64.0f);

	 image = images_g.begin();
	 flow = flowsGPU.begin();
	 while (image != images_g.end() && flow != flowsGPU.end() && k < q_g.size())
	 {
		 //resampleAreaParallelSeparateAdjoined(q_g[k], temp1_g, nx_orig, ny_orig, pitchf1_orig, nx, ny, pitchf1, temp4_g);
	     if (blur > 0.0f)
	     {
	         //gaussBlurSeparateMirror(temp1_g, temp2_g, nx, ny, blur, blur, (int) (3.0f * blur), temp4_g, 0);
	     }
		 else
		 {
		     float* temp = temp1_g;
		     temp1_g = temp2_g;
		     temp2_g = temp;
		 }

	     if (factor_rescale_x > 1.0f || factor_rescale_y > 1.0f)
		 {
		     //setKernel<<<getGridSize(nx, ny), blockSize>>>(temp1_g, nx, ny, pitchf1, 0.0f);
		     //  forewardRegistrationBilinearAtomic<<<getGridSize(nx, ny), blockSize>>>(flow->u_g, flow->v_g,temp2_g, temp1_g,  nx, ny, pitchf1);
		 }
		 else
		 {
		     float *temp = temp1_g;
		     temp1_g = temp2_g;
		     temp2_g = temp;
		 }
		 //addKernel<<<getGridSize(nx, ny), blockSize>>>(temp3_g, temp1_g, nx, ny, pitchf1);
		 k++;
		 flow++;
		 image++;
	}
    //primal1N<<<getGridSize(nx, ny), blockSize>>>(xi1_g, xi2_g, temp3_g, u_g, uor_g, nx, ny, pitchf1, factor_tv_update, factor_degrade_update, tau_p, overrelaxation);
}






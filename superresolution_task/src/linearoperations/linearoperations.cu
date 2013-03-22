/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
 *
 * time:    winter term 2012/13 / March 11-18, 2013
 *
 * project: superresolution
 * file:    linearoperations.cu
 *
 *
 * implement all functions with ### implement me ### in the function body
 \****************************************************************************/

/*
 * linearoperations.cu
 *
 *  Created on: Aug 3, 2012
 *      Author: steinbrf
 */

#include <auxiliary/cuda_basic.cuh>

cudaChannelFormatDesc linearoperation_float_tex =
		cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;

#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   21    // maximum allowed kernel radius + 1
__constant__ float constKernel[MAXKERNELSIZE];

void setTexturesLinearOperations(int mode)
{
	tex_linearoperation.addressMode[0] = cudaAddressModeClamp;
	tex_linearoperation.addressMode[1] = cudaAddressModeClamp;
	if (mode == 0)
		tex_linearoperation.filterMode = cudaFilterModePoint;
	else
		tex_linearoperation.filterMode = cudaFilterModeLinear;
	tex_linearoperation.normalized = false;
}

#define LO_TEXTURE_OFFSET 0.5f
#define LO_RS_AREA_OFFSET 0.0f

#ifdef DGT400
#define LO_BW 32
#define LO_BH 16
#else
#define LO_BW 16
#define LO_BH 16
#endif

#ifndef RESAMPLE_EPSILON
#define RESAMPLE_EPSILON 0.005f
#endif

#ifndef atomicAdd
__device__ float atomicAdd(float* address, double val)
{
	unsigned int* address_as_ull = (unsigned int*) address;
	unsigned int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val + __int_as_float(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

#endif

void backwardRegistrationBilinearValueTex(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g, float value,
		int nx, int ny, int pitchf1_in, int pitchf1_out, float hx, float hy)
{
	// ### Implement me ###
}

// gpu warping kernel
__global__ void backwardRegistrationBilinearFunctionGlobalGpu(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (x < nx && y < ny)
	{
		const float xx = (float) x + flow1_g[y * pitchf1_in + x] / hx;
		const float yy = (float) y + flow2_g[y * pitchf1_in + x] / hy;

		int xxFloor = (int) floor(xx);
		int yyFloor = (int) floor(yy);

		int xxCeil = xxFloor == nx - 1 ? xxFloor : xxFloor + 1;
		int yyCeil = yyFloor == ny - 1 ? yyFloor : yyFloor + 1;

		float xxRest = xx - (float) xxFloor;
		float yyRest = yy - (float) yyFloor;

		//same weird expression as in cpp
		out_g[y * pitchf1_out + x] =
		(xx < 0.0f || yy < 0.0f || xx > (float) (nx - 1)
						|| yy > (float) (ny - 1)) ? constant_g[y * pitchf1_in + x] :
						(1.0f - xxRest) * (1.0f - yyRest)* in_g[yyFloor * pitchf1_in + xxFloor]
								+ xxRest * (1.0f - yyRest)* in_g[yyFloor * pitchf1_in + xxCeil]
								+ (1.0f - xxRest) * yyRest* in_g[yyCeil * pitchf1_in + xxFloor]
								+ xxRest * yyRest* in_g[yyCeil * pitchf1_in + xxCeil];

	}
}

// initialize cuda warping kernel
void backwardRegistrationBilinearFunctionGlobal(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy)
{
	//same construction as in main flow to compute block and grid size
	int ngx = (nx % LO_BW) ? ((nx / LO_BW) + 1) : (nx / LO_BW);
	int ngy = (ny % LO_BH) ? ((ny / LO_BH) + 1) : (ny / LO_BH);

	dim3 dimGrid(ngx, ngy);
	dim3 dimBlock(LO_BW, LO_BH);

	//call warp method on gpu
	backwardRegistrationBilinearFunctionGlobalGpu<<<dimGrid, dimBlock>>>(in_g,
			flow1_g, flow2_g, out_g, constant_g, nx, ny, pitchf1_in,
			pitchf1_out, hx, hy);
}

void backwardRegistrationBilinearFunctionTex(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy)
{
	// ### Implement me, if you want ###
}

void forewardRegistrationBilinearAtomic(const float *flow1_g,
		const float *flow2_g, const float *in_g, float *out_g, int nx, int ny,
		int pitchf1)
{
	// ### Implement me ###
}

void gaussBlurSeparateMirrorGpu(float *in_g, float *out_g, int nx, int ny,
		int pitchf1, float sigmax, float sigmay, int radius, float *temp_g,
		float *mask)
{
	// ### Implement me ###
}

void resampleAreaParallelSeparate(const float *in_g, float *out_g, int nx_in,
		int ny_in, int pitchf1_in, int nx_out, int ny_out, int pitchf1_out,
		float *help_g, float scalefactor)
{
	// ### Implement me ###
}

void resampleAreaParallelSeparateAdjoined(const float *in_g, float *out_g,
		int nx_in, int ny_in, int pitchf1_in, int nx_out, int ny_out,
		int pitchf1_out, float *help_g, float scalefactor)
{
	// ### Implement me ###
}

__global__ void addKernel(const float *increment_g, float *accumulator_g,
		int nx, int ny, int pitchf1)
{
	// ### Implement me ###
}

__global__ void subKernel(const float *increment_g, float *accumulator_g,
		int nx, int ny, int pitchf1)
{
	// ### Implement me ###
}

__global__ void setKernel(float *field_g, int nx, int ny, int pitchf1,
		float value)
{
	// ### Implement me ###
}


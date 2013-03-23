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

__global__ void resampleAreaParallelSeparate_x
(
		const float * in_g,
		float * out_g,
		int nx,
		int ny,
		float hx,
		int pitchf1_in,
		float factor = 0.0f
)
{
	if( factor == 0.0f ) { factor = 1/hx; }

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ix + iy * pitchf1_in; // global index for out image

	if( ix < nx && iy < ny)
	{
		// initialising out
		out_g[ index ] = 0.0f;

		float px = (float)ix * hx;

		float left = ceil(px) - px;
		if(left > hx) left = hx;

		float midx = hx - left;
		float right = midx - floorf(midx);
		midx = midx - right;

		if( left > 0.0f )
		{
			// using pitchf1_in instead of nx_orig in original code
			out_g[index] += in_g[ iy * pitchf1_in + (int) floor(px) ] * left * factor; // look out for conversion of coordinates
			px += 1.0f;
		}
		while(midx > 0.0f)
		{
			// using pitchf1_in instead of nx_orig in original code
			out_g[index] += in_g[ iy * pitchf1_in + (int)(floor(px))] * factor;
			px += 1.0f;
			midx -= 1.0f;
		}
		if(right > RESAMPLE_EPSILON)
		{
			// using pitchf1_in instead of nx_orig in original code
			out_g[index] += in_g[ iy * pitchf1_in + (int)(floor(px))] * right * factor;
		}
	}
}

__global__ void resampleAreaParallelSeparate_y
(
		const float * in_g,
		float * out_g,
		int nx,
		int ny,
		float hy,
		int pitchf1_out,
		float factor = 0.0f // need
)
{
	if(factor == 0.0f) factor = 1.0f/hy;

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ix + iy * pitchf1_out; // global index for out image
									  // used pitch instead  of blockDim.x

	if( ix < nx && iy < ny ) // guards
	{
		out_g[index] = 0.0f;

		float py = (float)iy * hy;
		float top = ceil(py) - py;

		if(top > hy) top = hy;
		float midy = hy - top;

		float bottom = midy - floorf(midy);
		midy = midy - bottom;

		if(top > 0.0f)
		{
			// using pitch for helper array since these all arrays have same pitch
			out_g[index] += in_g[(int)(floor(py)) * pitchf1_out + ix ] * top * factor;
			py += 1.0f;
		}
		while(midy > 0.0f)
		{
			out_g[index] += in_g[(int)(floor(py)) * pitchf1_out + ix ] * factor;
			py += 1.0f;
			midy -= 1.0f;
		}
		if(bottom > RESAMPLE_EPSILON)
		{
			out_g[index] += in_g[(int)(floor(py)) * pitchf1_out + ix ] * bottom * factor;
		}
	}
}

void resampleAreaParallelSeparate(const float *in_g, float *out_g, int nx_in,
		int ny_in, int pitchf1_in, int nx_out, int ny_out, int pitchf1_out,
		float *help_g, float scalefactor)
{
	// helper array is already allocated on the GPU as _b1, now help_g

	// can reduce no of blocks for first pass
	dim3 dimGrid((int)ceil((float)nx_out/LO_BW), (int)ceil((float)ny_out/LO_BH));
	dim3 dimBlock(LO_BW,LO_BH);

	float hx = (float) nx_in/ (float) nx_out;
	float factor = (float)(nx_out)/(float)(nx_in);
	resampleAreaParallelSeparate_x<<< dimGrid, dimBlock >>>( in_g, help_g, nx_out, ny_in, hx, pitchf1_in, factor);

	float hy = (float)(ny_in)/(float)(ny_out);
	factor = scalefactor*(float)(ny_out)/(float)(ny_in);
	resampleAreaParallelSeparate_y<<< dimGrid, dimBlock >>>( help_g, out_g, nx_out, ny_out, hy, pitchf1_out, factor );
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


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

/*
 * Although it may depend on the graphics card, the performance gain of using the interpolation
 * of the texture memory was hardly noticeable. Anyway, texture memory is faster than global
 * memory, even with software interpolation, due to random access.
 *
 * We are creating the kernels for the gaussian convolution only once (at the first call) and
 * leave them in constant memory. This may not be intended for other applications, but in the
 * superresolution task it is advantageous.
 *
 * As the gaussian convolution is a linear operation, it should be possible to combine the
 * separate convolutions for x and y direction in one kernel.
 *
 */

#include <auxiliary/cuda_basic.cuh>
#include <iostream>
#include <linearoperations/linearoperations.cuh>

// TEXTURES

// Use hardware interpolation instead of software interpolation
// in bilinear interpolation kernels with texture memory

#define USE_HARDWARE_INTERPOLATION	  0

#define TEXTURE_OFFSET      0.5f  // offset for indexing textures

cudaChannelFormatDesc linearoperation_float_tex = cudaCreateChannelDesc<float>();
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;

// CONSTANT MEMORY KERNELS

#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   	21    // maximum allowed kernel radius + 1

__constant__ float constKernelX[MAXKERNELSIZE];
__constant__ float constKernelY[MAXKERNELSIZE];
// create and bind the gaussian convolution kernels only once
bool constant_kernel_bound = false;

void gpu_bindKernelToConstantMemory_x ( const float* kernel_x, const int size ) 
{
	cutilSafeCall( cudaMemcpyToSymbol( constKernelX, kernel_x, size * sizeof(float) ) );
}

void gpu_bindKernelToConstantMemory_y ( const float* kernel_y, const int size ) 
{
	cutilSafeCall( cudaMemcpyToSymbol( constKernelY, kernel_y, size * sizeof(float) ) );
}

// TEXTURE METHODS

void gpu_bindTextureMemory( float *d_inputImage, int iWidth, int iHeight, size_t iPitchBytes )
{
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, d_inputImage, &linearoperation_float_tex, iWidth, iHeight, iPitchBytes) );
}
void gpu_bindTextureMemory( const float *d_inputImage, int iWidth, int iHeight, size_t iPitchBytes )
{
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, d_inputImage, &linearoperation_float_tex, iWidth, iHeight, iPitchBytes) );
}

void gpu_unbindTextureMemory()
{
	cutilSafeCall( cudaUnbindTexture(tex_linearoperation) );
}

// added mirroring as possible border condition, default is clamping to keep existing code working
void setTexturesLinearOperations( int mode, int borderCondition = 0 )
{
	if( borderCondition == 0 )
	{
		tex_linearoperation.addressMode[0] = cudaAddressModeClamp;
		tex_linearoperation.addressMode[1] = cudaAddressModeClamp;
	}
	else
	{
		tex_linearoperation.addressMode[0] = cudaAddressModeMirror;
		tex_linearoperation.addressMode[1] = cudaAddressModeMirror;
	}

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

//================================================================
// backward warping value
//================================================================

/*
 * Texture is faster than global memory, as memory access may be random
 * Access locations depending on flow direction
 */
__global__ void backwardRegistrationBilinearValueTexKernel (
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		float value,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < nx && y < ny )
	{
		// backwards warped coordinates
		float hx_1 = 1.0f / hx;
		float hy_1 = 1.0f / hy;
	
		// get target area, where flow points at
		float ii_fp = x + (flow1_g[y * pitchf1_in + x] * hx_1);
		float jj_fp = y + (flow2_g[y * pitchf1_in + x] * hy_1);
	
		// set result to a given value, if the flow points outside the image or is not a number
		if( (ii_fp < 0.0f) || (jj_fp < 0.0f || !isfinite( ii_fp ) || !isfinite( jj_fp ) )
					 || (ii_fp > (float)(nx - 1)) || (jj_fp > (float)(ny - 1)) )
		{
			out_g[y * pitchf1_out + x] = value;
		}
		else
		{
			// interpolate between the 4 pixels around target coordinates

			// left and upper pixel coordinates
			int xx = (int)ii_fp;
			int yy = (int)jj_fp;
	
			int xx1 = xx == nx - 1 ? xx : xx + 1;
			int yy1 = yy == ny - 1 ? yy : yy + 1;
	
			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;
	
			// interpolate...
			out_g[y * pitchf1_out + x] =
					(1.0f - xx_rest) * (1.0f - yy_rest) * tex2D( tex_linearoperation, xx, yy )
					+ xx_rest * (1.0f - yy_rest)        * tex2D( tex_linearoperation, xx1, yy )
					+ (1.0f - xx_rest) * yy_rest        * tex2D( tex_linearoperation, xx, yy1 )
					+ xx_rest * yy_rest                 * tex2D( tex_linearoperation, xx1, yy1 );
		}
	}
}

__global__ void backwardRegistrationBilinearValueTexInterpolationKernel (
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		float value,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < nx && y < ny )
	{
		// backwards warped coordinates
		float hx_1 = 1.0f / hx;
		float hy_1 = 1.0f / hy;
	
		// get target area, where flow points at
		float xx = x + (flow1_g[y * pitchf1_in + x] * hx_1);
		float yy = y + (flow2_g[y * pitchf1_in + x] * hy_1);
	
		// set result to a given value, if the flow points outside the image or is not a number
		if( (xx < 0.0f) || (yy < 0.0f || !isfinite( xx ) || !isfinite( yy ) )
					 || (xx > (float)(nx - 1)) || (yy > (float)(ny - 1)) )
		{
			out_g[y * pitchf1_out + x] = value;
		}
		else
		{
			// interpolate between the 4 pixels around target coordinates
			out_g[y * pitchf1_out + x] =
					tex2D( tex_linearoperation, xx + TEXTURE_OFFSET, yy + TEXTURE_OFFSET );
		}
	}
}

/*
 * Texture memory version
 */
void backwardRegistrationBilinearValueTex (
		float* in_g,
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		float value,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	// TODO: has texture to be bound every time?

	// prepare texture
	#if USE_HARDWARE_INTERPOLATION
		setTexturesLinearOperations( 1, 0 ); // filter mode: linear, address mode: clamping
	#else
		setTexturesLinearOperations( 0, 0 ); // filter mode: point (no offset necessary), address mode: clamping
	#endif


	// bind input image to texture
	gpu_bindTextureMemory( in_g, nx, ny, pitchf1_in * sizeof(float) );

#if USE_HARDWARE_INTERPOLATION
	backwardRegistrationBilinearValueTexInterpolationKernel<<<dimGrid, dimBlock>>>
			( flow1_g, flow2_g, out_g, value, nx, ny, pitchf1_in, pitchf1_out, hx, hy );
#else
	backwardRegistrationBilinearValueTexKernel<<<dimGrid, dimBlock>>>
			( flow1_g, flow2_g, out_g, value, nx, ny, pitchf1_in, pitchf1_out, hx, hy );
#endif


	// release texture
	gpu_unbindTextureMemory();

}



// using global memory
__global__ void backwardRegistrationBilinearValueTexKernel_gm
	(
		const float* in_g,
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		float value,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < nx && y < ny )
	{
		// backwards warped coordinates
		float ii_fp = x + (flow1_g[y * pitchf1_in + x] / hx);
		float jj_fp = y + (flow2_g[y * pitchf1_in + x] / hy);
	
		// set result to a given value, if the flow points outside the image or is not a number
		if (
				   (ii_fp < 0.0f) || (jj_fp < 0.0f)
				|| (ii_fp > (float)(nx - 1)) || (jj_fp > (float)(ny - 1))
				|| !isfinite( ii_fp ) || !isfinite( jj_fp )
			)
		{
			// if not, use given value
			out_g[y * pitchf1_out + x] = value;
		}
		else
		{
			// interpolate between the 4 target pixels

			// prepare coordinates for interpolation
			int xx = (int)ii_fp;
			int yy = (int)jj_fp;
	
			int xx1 = xx == nx - 1 ? xx : xx + 1;
			int yy1 = yy == ny - 1 ? yy : yy + 1;
	
			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;
	
			// interpolate between pixels around target coordinates
			out_g[y * pitchf1_out + x] =
					(1.0f - xx_rest) * (1.0f - yy_rest) * in_g[yy  * pitchf1_in + xx]
					+ xx_rest * (1.0f - yy_rest)        * in_g[yy  * pitchf1_in + xx1]
					+ (1.0f - xx_rest) * yy_rest        * in_g[yy1 * pitchf1_in + xx]
					+ xx_rest * yy_rest                 * in_g[yy1 * pitchf1_in + xx1];
		}
	}
}

/*
 * Global memory version for speed comparison
 */
void backwardRegistrationBilinearValueTex_gm (
		const float* in_g,
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		float value,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	backwardRegistrationBilinearValueTexKernel_gm<<<dimGrid, dimBlock>>>
			( in_g, flow1_g, flow2_g, out_g, value, nx, ny, pitchf1_in, pitchf1_out, hx, hy );
}




//================================================================
// backward warping
//================================================================


// gpu warping kernel with global memory
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
		// backwards warped coordinates
		const float xx = (float) x + flow1_g[y * pitchf1_in + x] / hx;
		const float yy = (float) y + flow2_g[y * pitchf1_in + x] / hy;

		// prepare coordinates for interpolation
		int xxFloor = (int) floor(xx);
		int yyFloor = (int) floor(yy);

		int xxCeil = xxFloor == nx - 1 ? xxFloor : xxFloor + 1;
		int yyCeil = yyFloor == ny - 1 ? yyFloor : yyFloor + 1;

		float xxRest = xx - (float) xxFloor;
		float yyRest = yy - (float) yyFloor;

		// interpolate between the 4 target pixels
		out_g[y * pitchf1_out + x] =
				// check, if warped coordinates are within image boundary
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx - 1) || yy > (float)(ny - 1))
				?
					// if not, take grey values from target image (to which we warp the actual one)
					constant_g[y * pitchf1_in + x]
			    :
					// interpolate between pixels around target coordinates
					  (1.0f - xxRest) * (1.0f - yyRest)
					  	* in_g[yyFloor * pitchf1_in + xxFloor]
					+ xxRest * (1.0f - yyRest)
						* in_g[yyFloor * pitchf1_in + xxCeil]
					+ (1.0f - xxRest) * yyRest
						* in_g[yyCeil * pitchf1_in + xxFloor]
					+ xxRest * yyRest
						* in_g[yyCeil * pitchf1_in + xxCeil];

	}
}

// initialize cuda warping kernel
void backwardRegistrationBilinearFunctionGlobal(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	//call warp method on gpu
	backwardRegistrationBilinearFunctionGlobalGpu<<<dimGrid, dimBlock>>>(in_g,
			flow1_g, flow2_g, out_g, constant_g, nx, ny, pitchf1_in,
			pitchf1_out, hx, hy);
}






// gpu warping kernel with texture memory
__global__ void backwardRegistrationBilinearFunctionTextureGpu
	(
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		const float *constant_g,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (x < nx && y < ny)
	{
		// backwards warped coordinates
		const float xx = (float) x + flow1_g[y * pitchf1_in + x] / hx;
		const float yy = (float) y + flow2_g[y * pitchf1_in + x] / hy;

		// prepare coordinates for software interpolation
		int xxFloor = (int) floor(xx);
		int yyFloor = (int) floor(yy);

		int xxCeil = (xxFloor == nx - 1) ? (xxFloor) : (xxFloor + 1);
		int yyCeil = (yyFloor == ny - 1) ? (yyFloor) : (yyFloor + 1);

		float xxRest = xx - (float) xxFloor;
		float yyRest = yy - (float) yyFloor;

		// interpolate between the 4 target pixels
		out_g[y * pitchf1_out + x] =
				// check, if warped coordinates are within image boundary
				(xx < 0.0f || yy < 0.0f || xx > (float) (nx - 1) || yy > (float) (ny - 1))
				?
					// if not, take grey values from target image (to which we warp the actual one)
					constant_g[y * pitchf1_in + x]
				:
					// interpolate between pixels around target coordinates
					  (1.0f - xxRest) * (1.0f - yyRest) * tex2D( tex_linearoperation, xxFloor, yyFloor ) // ingen offset 
					+ xxRest          * (1.0f - yyRest) * tex2D( tex_linearoperation, xxCeil,  yyFloor )
					+ (1.0f - xxRest) * yyRest          * tex2D( tex_linearoperation, xxFloor, yyCeil )
					+ xxRest          * yyRest          * tex2D( tex_linearoperation, xxCeil,  yyCeil );

	}
}

// gpu warping kernel with texture memory and hardware interpolation
__global__ void backwardRegistrationBilinearFunctionTextureInterpolationGpu
	(
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		const float *constant_g,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (x < nx && y < ny)
	{
		// backwards warped coordinates
		const float xx = (float) x + flow1_g[y * pitchf1_in + x] / hx;
		const float yy = (float) y + flow2_g[y * pitchf1_in + x] / hy;

		out_g[y * pitchf1_out + x] =
				// check, if warped coordinates are within image boundary
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx - 1) || yy > (float)(ny - 1))
				?
					// if not, take grey values from target image (to which we warp the actual one)
					constant_g[y * pitchf1_in + x]
				:
					// using GPU interpolation hardware
					tex2D( tex_linearoperation, xx + TEXTURE_OFFSET, yy + TEXTURE_OFFSET );

	}
}

void backwardRegistrationBilinearFunctionTex
	(
		const float* in_g,
		const float* flow1_g,
		const float* flow2_g,
		float* out_g,
		const float* constant_g,
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,
		float hy
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	// TODO: has texture to be bound every time?

	// prepare texture
	#if USE_HARDWARE_INTERPOLATION
		setTexturesLinearOperations( 1, 0 ); // filter mode: linear, address mode: clamping
	#else
		setTexturesLinearOperations( 0, 0 ); // filter mode: point (no offset necessary), address mode: clamping
	#endif

	// bind input image to texture
	gpu_bindTextureMemory( in_g, nx, ny, pitchf1_in * sizeof(float) );

	#if USE_HARDWARE_INTERPOLATION
		backwardRegistrationBilinearFunctionTextureInterpolationGpu<<<dimGrid, dimBlock>>>(
					flow1_g, flow2_g, out_g, constant_g, nx, ny, pitchf1_in,
					pitchf1_out, hx, hy );
	#else
		backwardRegistrationBilinearFunctionTextureGpu<<<dimGrid, dimBlock>>>(
				flow1_g, flow2_g, out_g, constant_g, nx, ny, pitchf1_in,
				pitchf1_out, hx, hy );
	#endif

	// release texture
	gpu_unbindTextureMemory();
}



//================================================================
// forward warping
//================================================================


__global__ void foreward_warp_kernel_atomic (
		const float *flow1_g,	// flow.u
		const float *flow2_g,	// flow.v
		const float *in_g,		// temp2_g
		float *out_g,			// temp1_g
		int nx,
		int ny,
		int pitchf1
	)
{
	// get thread coordinates and index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < nx && y < ny )
	{
		const unsigned int idx = y * pitchf1 + x;
		
		// calculate target coordinates: coords + flow values
		const float xx = (float)x + flow1_g[idx];
		const float yy = (float)y + flow2_g[idx];
	
		// continue only if target area inside image
		if (
				xx >= 0.0f &&
				xx <= (float)(nx - 2) &&
				yy >= 0.0f &&
				yy <= (float)(ny - 2)
			)
		{
			float xxf = floor(xx);
			float yyf = floor(yy);
		
			// target pixel coordinates
			const int xxi = (int)xxf;
			const int yyi = (int)yyf;
		
			xxf = xx - xxf;
			yyf = yy - yyf;
		
			// distribute input pixel value to adjacent pixels of target pixel
			const float in_value = in_g[idx];

			// eject the warp core!
			// avoid race conditions by use of atomic operations
			atomicAdd( out_g + (yyi * pitchf1 + xxi),           in_value * (1.0f - xxf) * (1.0f - yyf) );
			atomicAdd( out_g + (yyi * pitchf1 + xxi + 1),       in_value * xxf * (1.0f - yyf) );
			atomicAdd( out_g + ((yyi + 1) * pitchf1 + xxi),     in_value * (1.0f - xxf) * yyf );
			atomicAdd( out_g + ((yyi + 1) * pitchf1 + xxi + 1), in_value * xxf * yyf );

			// hierarchical atomics not reasonable:
			// target coordinates can be anywhere on image,
			// so memory for whole image has to be allocated per block
			// but still a lot of atomics would be necessary
			// to get rid of the atomics, memory for the whole image
			// had to be stored per thread, which is an absolute overkill,
			// as each thread writes to only 4 of teh nx*ny values and
			// each thread has to loop over nx*ny thread memories afterwards
			// to summarize the values
		}
	}
}



/*
 * Forward warping
 */
void forewardRegistrationBilinearAtomic (
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
		float *out_g,
		int nx,
		int ny,
		int pitchf1
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	// reset target array
	// this cost us hours again, not to do it in the warp kernel...
	setKernel <<< dimGrid, dimBlock >>> ( out_g, nx, ny, pitchf1, 0.0f );

	// invoke atomic warp kernel on gpu
	foreward_warp_kernel_atomic <<< dimGrid, dimBlock >>> ( flow1_g, flow2_g, in_g, out_g, nx, ny, pitchf1 );
}

//================================================================
// gaussian blur (mirrored) - global + constant memory
//================================================================

/*
 * Convolution kernel for gaussian blur in x direction
 * mirrored border
 * 
 * global memory
 *
 * Of course it is fine to have a universal convolution
 * function that can be used for x and y direction,
 * but specialized ones are faster. A similar kernel
 * could be used with both X radius and Y radius, first
 * invoked with Xradius = radius and Yradius = 1, and
 * then the other way round.
 * Here the kernel has been splittet into two similar
 * ones, the first with a hardcoded Yradius of 1, the
 * second with a hardcoded Xradius of 1.
 */
__global__ void gaussBlurSeparateMirrorGPU_gm_x
	(
		const float*	in,			// input image
		      float*	out,		// convoluted output
		const int		nx,
		const int		ny,
		const int		pitchf1,	// pitch for this image
		const int		radius		// kernel radius
	)
{
	// get thread/pixel coordinates
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	// guards: continue calculation only for pixels inside the image
	if( x < nx && y < ny )
	{
		// the kernel is symmetric and therefore only the center and the right half is given
		int idx = y * pitchf1 + x;

		// calculate center outside of the loop (otherwise it would be computed twice)		
		float result = constKernelX[0] * in[idx]; // temp var for output pixel
		
		for( int i = 1; i <= radius; ++i )
		{
			// computing offset symmetrically
			int xShiftLeft = x - i;
			int xShiftRight = x + i;
			
			result += constKernelX[i] * (
					// left side of kernel
					( (xShiftLeft >= 0) ? in[y * pitchf1 + xShiftLeft] : in[y * pitchf1 + (-1 - xShiftLeft) ] ) + // border condition: mirroring

					// right side of kernel
					( (xShiftRight < nx) ? in[y * pitchf1 + xShiftRight] : in[y * pitchf1 + nx - (xShiftRight - (nx - 1)) ]) // border condition: mirroring
				);
		}

		// write to output image
		out[idx] = result;
	}
}

__global__ void gaussBlurSeparateMirrorGPU_gm_y
	(
		const float*	in,			// input image
		      float*	out,		// convoluted output
		const int		nx,
		const int		ny,
		const int		pitchf1,	// pitch for this image
		const int		radius		// kernel radius
	)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if( x < nx && y < ny ) // guards
	{
		int idx = y * pitchf1 + x;
				
		float result = constKernelY[0] * in[idx];
		
		for( int i = 1; i <= radius; ++i )
		{
			int yShiftLeft = y - i;
			int yShiftRight = y + i;
			
			result += constKernelY[i] * (
					( (yShiftLeft >= 0) ? in[yShiftLeft * pitchf1 + x] : in[(-1 - yShiftLeft) * pitchf1 + x ] ) + 
					( (yShiftRight < ny) ? in[yShiftRight * pitchf1 + x] : in[(ny - (yShiftRight - (ny - 1))) * pitchf1 + x ]));
		}
		
		out[idx] = result;		
	}
}






/*
 * Helper method to create kernel and bind it
 * to constant device memory
 *
 * mask is supposed to be a CPU pointer!
 */
void createConstantGaussKernels
	(
		int radius,
		float sigmax,
		float sigmay,
		float* mask		// CPU Pointer!
	)
{
	// set radius automatically, if not given
	if( radius == 0 )
	{
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)( 3.0f * maxsigma );
	}

	// adapt sigmas for kernel
	sigmax = 1.0f / (sigmax * sigmax);
	sigmay = 1.0f / (sigmay * sigmay);

	// allocate mask memory, if not given
	bool selfallocmask = mask == 0;
	
	if(selfallocmask)
	{
		mask = new float[radius + 1];
	}

	// prepare gaussian kernel (1D) for x direction
	float sum = 1.0f;
	mask[0] = 1.0f;

	for( int x = 1; x <= radius; ++x )
	{
		mask[x] = exp( -0.5f * ( (float)(x * x) * sigmax) );
		sum += 2.0f * mask[x];
	}

	// normalize kernel
	for( int x = 0; x <= radius; ++x )
	{
		mask[x] /= sum;
	}

	// bind kernel to constant memory
	gpu_bindKernelToConstantMemory_x ( mask, radius + 1 );

	//-----------------------
	// kernel for y direction
	//-----------------------

	// reuse mask memory, as it is copied to device: x mask is not overwritten
	mask[0] = sum = 1.0f;

	// prepare gaussian kernel (1D)
	for( int gx = 1; gx <= radius; ++gx )
	{
		mask[gx] = exp( -0.5f * ( (float)(gx * gx) * sigmay) );
		sum += 2.0f * mask[gx];
	}

	// normalize kernel
	for( int gx = 0; gx <= radius; ++gx )
	{
		mask[gx] /= sum;
	}

	// bind kernel to constant memory
	gpu_bindKernelToConstantMemory_y ( mask, radius + 1 );

	// cleanup
	if( selfallocmask )
		delete [] mask;

	constant_kernel_bound = true;
}




/*
 * wrapping method for gaussian convolution gpu kernel with global memory
 *
 * mask is supposed to be a CPU pointer!
 */
void gaussBlurSeparateMirrorGpu_gm
	(
		float* 	in_g,		// input image
		float* 	out_g,		// convoluted output image
		int 	nx,
		int 	ny,
		int 	pitchf1,	// pitch for this image
		float 	sigmax,		// gauss parameters
		float 	sigmay,
		int 	radius,
		float* 	temp_g,
		float* 	mask		// gauss kernel memory: pointer to CPU memory!
	)
{
	if( sigmax <= 0.0f || sigmay <= 0.0f || radius < 0 )
	{
		// copy input to output memory
		cudaMemcpy( in_g, out_g, ny * pitchf1, cudaMemcpyDeviceToDevice );

		// not going for a kernel call
		fprintf( stderr, "Warning: Gaussian blur skipped!" );
		return;
	}

	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	//-----------------------
	// memory preparation
	//-----------------------

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin; // pitch is the same as pitchf1
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}

	//-----------------------
	// kernel preparation
	//-----------------------
	
	// compute kernel only once and keep it in constant memory
	if( !constant_kernel_bound )
	{
		createConstantGaussKernels( radius, sigmax, sigmay, mask );
	}

	//-----------------------
	// convolution
	//-----------------------


	// invoke gauss kernels on gpu for convolution in x and y direction
	// MAXKERNELSIZE and MAXKERNELRADIUS do not allow to combine x and y in one kernel
	gaussBlurSeparateMirrorGPU_gm_x<<<dimGrid,dimBlock>>>( in_g, temp_g, nx, ny, pitchf1, radius );
	gaussBlurSeparateMirrorGPU_gm_y<<<dimGrid,dimBlock>>>( temp_g, out_g, nx, ny, pitchf1, radius );

	//-----------------------
	// cleanup
	//-----------------------

	if( selfalloctemp )
		cutilSafeCall( cudaFree( temp_g ) );
}




//================================================================
// gaussian blur (mirrored) - texture + constant memory
//================================================================

/*
 * Convolution kernel for gaussian blur in x direction
 * mirrored border
 * 
 * texture + constant memory
 *
 * Of course it is fine to have a universal convolution
 * function that can be used for x and y direction,
 * but specialized ones are faster. A similar kernel
 * could be used with both X radius and Y radius, first
 * invoked with Xradius = radius and Yradius = 1, and
 * then the other way round.
 * Here the kernel has been splittet into two similar
 * ones, the first with a hardcoded Yradius of 1, the
 * second with a hardcoded Xradius of 1.
 */
__global__ void gaussBlurSeparateMirrorGPU_tex_cm_x
	(
		      float*	out,		// convoluted output
		const int		nx,
		const int		ny,
		const int		pitchf1,	// pitch for this image
		const int		radius		// kernel radius
	)
{
	// get thread/pixel coordinates
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	// guards: continue calculation only for pixels inside the image
	if( x < nx && y < ny )
	{
		// the kernel is symmetric and therefore only the center and the right half is given
		
		// calculate center outside of the loop (otherwise it would be computed twice)		
		float result = constKernelX[0] * tex2D( tex_linearoperation, x, y );
		
		for( int i = 1; i <= radius; ++i )
		{
			// no offset of 0.5f required, as filter mode point is used
			// used address mode: mirror

			// computing offset symmetrically
			result += constKernelX[i] * (
					tex2D( tex_linearoperation, x - i, y ) + // left part of kernel
					tex2D( tex_linearoperation, x + i, y )   // right part of the kernel
				);
		}

		// write to output image
		out[y * pitchf1 + x] = result;
	}
}

__global__ void gaussBlurSeparateMirrorGPU_tex_cm_y
	(
		      float*	out,		// convoluted output
		const int		nx,
		const int		ny,
		const int		pitchf1,	// pitch for this image
		const int		radius		// kernel radius
	)
{
	// get thread/pixel coordinates
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	// guards: continue calculation only for pixels inside the image
	if( x < nx && y < ny )
	{
		// the kernel is symmetric and therefore only the center and the lower half is given

		// calculate center outside of the loop (otherwise it would be computed twice)		
		float result = constKernelY[0] * tex2D( tex_linearoperation, x, y );
		
		// computing offset symmetrically
		for( int i = 1; i <= radius; ++i )
		{
			// no offset of 0.5f for texture required, as filter mode point is used
			// used address mode: mirror

			result += constKernelY[i] * (
					tex2D( tex_linearoperation, x, y - i ) + // upper part of kernel
					tex2D( tex_linearoperation, x, y + i )   // lower part of kernel
				);
		}

		// write to output image
		out[y * pitchf1 + x] = result;		
	}
}

/*
 * wrapping method for gaussian convolution gpu kernel with texture and constant memory
 *
 * mask is supposed to be a CPU pointer!
 */
void gaussBlurSeparateMirrorGpu
	(
		float* 	in_g,		// input image
		float* 	out_g,		// convoluted output image
		int 	nx,
		int 	ny,
		int 	pitchf1,	// pitch for this image
		float 	sigmax,		// gauss parameters
		float 	sigmay,
		int 	radius,
		float* 	temp_g,
		float* 	mask		// gauss kernel memory: pointer to CPU memory!
	)
{
	if( sigmax <= 0.0f || sigmay <= 0.0f || radius < 0 )
	{
		// copy input to output memory
		cudaMemcpy( in_g, out_g, ny * pitchf1, cudaMemcpyDeviceToDevice );

		// not going for a kernel call
		fprintf( stderr, "Warning: Gaussian blur skipped!" );
		return;
	}

	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	//-----------------------
	// memory preparation
	//-----------------------

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin; // pitch is the same as pitchf1
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}

	//-----------------------
	// kernel preparation
	//-----------------------
	
	// compute kernel only once and keep it in constant memory
	if( !constant_kernel_bound )
	{
		createConstantGaussKernels( radius, sigmax, sigmay, mask );
	}

	//-----------------------
	// convolution
	//-----------------------

	// prepare texture
	setTexturesLinearOperations( 0, 1 ); // filter mode: point (no offset necessary), address mode: mirror

	// invoke gauss kernels on gpu for convolution in x and y direction
	// MAXKERNELSIZE and MAXKERNELRADIUS do not allow to combine x and y in one kernel

	// bind input image to texture
	gpu_bindTextureMemory( in_g, nx, ny, pitchf1 * sizeof(float) );

	gaussBlurSeparateMirrorGPU_tex_cm_x<<<dimGrid,dimBlock>>>( temp_g, nx, ny, pitchf1, radius );

	// bind input image to texture
	gpu_bindTextureMemory( temp_g, nx, ny, pitchf1 * sizeof(float) );

	gaussBlurSeparateMirrorGPU_tex_cm_y<<<dimGrid,dimBlock>>>( out_g, nx, ny, pitchf1, radius );

	//-----------------------
	// cleanup
	//-----------------------

	// unbind texture
	gpu_unbindTextureMemory();
	
	// free self allocated memory
	if( selfalloctemp )
		cutilSafeCall( cudaFree( temp_g ) );
}


//================================================================
// gaussian blur (mirrored) - shared + constant memory
//================================================================


/*
 * Convolution kernel for gaussian blur in x direction
 *
 * using shared memory for image data and
 * constant memory for kernel
 */
__global__ void gaussBlurSeparateMirrorGPU_sm_cm_x
	(
		const float* inputImage,
		float* outputImage,
		int iWidth,
		int iHeight,
		size_t iPitch,
		int radius
	)
{	
	// get thread/pixel coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int localThreadIndex = threadIdx.y * LO_BW + threadIdx.x;
	
	//==================================
	// load image data to shared memory
	//==================================
	
	// get size of image chunk
	int chunkPitch  = LO_BW + ( radius << 1 );
	int chunkWidth  = chunkPitch;
	//int chunkHeight = LO_BH;
	
	// create shared memory variable
	extern __shared__ float sharedImage[];
	
	int numThreads = LO_BW * LO_BH;
	
	// get block and chunk position (upper left corner of the chunk in the global image)
	int blockX = blockIdx.x * LO_BW;
	int blockY = blockIdx.y * LO_BH;
	int chunkX = blockX - radius;
	int chunkY = blockY;
	
	// left chunk offset if padding reaches out of image
	int chunkOffX = chunkX < 0 ? -1 * chunkX : 0;

	// update chunk width (inside image)
	chunkWidth = min( chunkX + chunkWidth, iWidth ) - chunkX; // -= chunkX + chunkWidth > iWidth ? iWidth - chunkX : 0; 

	// number of pixels
	int pixelNum = chunkPitch * LO_BH; // = chunkPitch * chunkHeight

	// chunk and image coordinates
	int cx, cy, ix, iy;

	// load pixels from global to shared memory
	for( int i = localThreadIndex; i < pixelNum; i += numThreads )
	{
		cx = i % chunkPitch;
		cy = i / chunkPitch;

		// get corresponding global coordinates. Border condition: mirror
		if( cx < chunkOffX ) 			// chunkOffX is always 0, if the chunk does not reach out of the image on the left side
		{
			ix = chunkOffX - 1 - cx;
		}
		else if ( cx >= chunkWidth )
			ix = iWidth - 1 - (cx - chunkWidth);
		else
			ix = chunkX + cx;

		// y is never outside image, as the kernel is 1D
		iy = chunkY + cy;

		sharedImage[ cx + chunkPitch * cy ] = inputImage[ ix + iPitch * iy ];
	}

	// synchronize threads
	__syncthreads();
	
	
	//==================================
	// convolution
	//==================================
	
	// continue calculation only for pixels inside the image
	if( x < iWidth && y < iHeight )
	{
		// the kernel is symmetric and therefore only the center and the right half is given
		// calculate center outside of the loop (otherwise it would be computed twice)

		// shared memory coordinates
		int tx = threadIdx.x + radius;
		int ty = threadIdx.y;

		float value = constKernelX[0] * sharedImage[ty * chunkPitch + tx]; // temp var for output pixel

		// kernel loop
		++radius; // just because < is faster than <=
		for( int i = 1; i < radius; ++i )
		{
			value += constKernelX[i] * (
					sharedImage[ty * chunkPitch + (tx - i)]		// left side of kernel
						+
					sharedImage[ty * chunkPitch + (tx + i)]		// right side of kernel
				);
		}
		// end of kernel loop
	
		// write to output image
		outputImage[ y * iPitch + x ] = value;
	}
}


/*
 * Convolution kernel for gaussian blur in y direction
 *
 * using shared memory for image data and
 * constant memory for kernel
 */
__global__ void gaussBlurSeparateMirrorGPU_sm_cm_y
	(
		const float* inputImage,
		float* outputImage,
		int iWidth,
		int iHeight,
		size_t iPitch,
		int radius
	)
{	
	// get thread/pixel coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int localThreadIndex = threadIdx.y * LO_BW + threadIdx.x;
	
	//==================================
	// load image data to shared memory
	//==================================
	
	// get size of image chunk
	//int chunkPitch  = LO_BW;
	int chunkHeight = LO_BH + ( radius << 1 );
	
	// create shared memory variable
	extern __shared__ float sharedImage[];
	
	int numThreads = LO_BW * LO_BH;
	
	// get block and chunk position (upper left corner of the chunk in the global image)
	int blockX = blockIdx.x * LO_BW;
	int blockY = blockIdx.y * LO_BH;
	int chunkX = blockX;
	int chunkY = blockY - radius;
	
	// top chunk offset if padding reaches out of image
	int chunkOffY = chunkY < 0 ? -1 * chunkY : 0;

	// update chunk height (inside image)
	chunkHeight = min( chunkY + chunkHeight, iHeight ) - chunkY; // -= chunkY + chunkHeight > iHeight ? iHeight - chunkY : 0; 

	// number of pixels
	int pixelNum = LO_BW * chunkHeight; // = chunkPitch * chunkHeight

	// chunk and image coordinates
	int cx, cy, ix, iy;

	// load pixels from global to shared memory
	for( int i = localThreadIndex; i < pixelNum; i += numThreads )
	{
		cx = i % LO_BW; // LO_BW = chunkPitch
		cy = i / LO_BW;

		// x is never outside image, as the kernel is 1D
		ix = chunkX + cx;

		// get corresponding global coordinates. Border condition: mirror
		if( cy < chunkOffY )			// chunkOffY is always 0, if the chunk does not reach out of the image on the top side
			iy = chunkOffY - 1 - cy;
		else if ( cy >= chunkHeight )
			iy = iHeight - 1 - (cy - chunkHeight);
		else
			iy = chunkY + cy;

		sharedImage[ cx + LO_BW * cy ] = inputImage[ ix + iPitch * iy ]; // LO_BW = chunkpitch
	}

	// synchronize threads
	__syncthreads();
	
	
	//==================================
	// convolution
	//==================================
	
	// continue calculation only for pixels inside the image
	if( x < iWidth && y < iHeight )
	{
		// the kernel is symmetric and therefore only the center and the right half is given
		// calculate center outside of the loop (otherwise it would be computed twice)

		// shared memory coordinates
		int tx = threadIdx.x;
		int ty = threadIdx.y + radius;

		float value = constKernelY[0] * sharedImage[ty * LO_BW + tx]; // temp var for output pixel, LO_BW = chunkPitch

		// kernel loop
		++radius; // just because < is faster than <=
		for( int i = 1; i < radius; ++i )
		{
			value += constKernelY[i] * (
					sharedImage[(ty - i) * LO_BW + tx]		// upper part of kernel, LO_BW = chunkPitch
						+
					sharedImage[(ty + i) * LO_BW + tx]		// lower part of kernel
				);

		}
		// end of kernel loop
	
		// write to output image
		outputImage[ y * iPitch + x ] = value;
	}
}



/*
 * wrapping method for gaussian convolution gpu kernel with texture and constant memory
 *
 * mask is supposed to be a CPU pointer!
 */
void gaussBlurSeparateMirrorGpu_sm
	(
		float* 	in_g,		// input image
		float* 	out_g,		// convoluted output image
		int 	nx,
		int 	ny,
		int 	pitchf1,	// pitch for this image
		float 	sigmax,		// gauss parameters
		float 	sigmay,
		int 	radius,
		float* 	temp_g,
		float* 	mask		// gauss kernel memory: pointer to CPU memory!
	)
{
	if( sigmax <= 0.0f || sigmay <= 0.0f || radius < 0 )
	{
		// copy input to output memory
		cudaMemcpy( in_g, out_g, ny * pitchf1, cudaMemcpyDeviceToDevice );

		// not going for a kernel call
		fprintf( stderr, "Warning: Gaussian blur skipped!" );
		return;
	}

	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	//-----------------------
	// memory preparation
	//-----------------------

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin; // pitch is the same as pitchf1
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}

	// get size of dynamically allocated shared memory
	int sharedMemorySize =  ( LO_BW + ( radius << 1) ) * LO_BH * sizeof(float);

	//-----------------------
	// kernel preparation
	//-----------------------

	// compute kernel only once and keep it in constant memory
	if( !constant_kernel_bound )
	{
		createConstantGaussKernels( radius, sigmax, sigmay, mask );
	}
	
	//-----------------------
	// convolution
	//-----------------------

	// invoke gauss kernels on gpu for convolution in x and y direction
	// MAXKERNELSIZE and MAXKERNELRADIUS do not allow to combine x and y in one kernel

	gaussBlurSeparateMirrorGPU_sm_cm_x<<< dimGrid, dimBlock, sharedMemorySize >>>( in_g, temp_g, nx, ny, pitchf1, radius );

	gaussBlurSeparateMirrorGPU_sm_cm_y<<< dimGrid, dimBlock, sharedMemorySize >>>( temp_g, out_g, nx, ny, pitchf1, radius );

	//-----------------------
	// cleanup
	//-----------------------

	if( selfalloctemp )
		cutilSafeCall( cudaFree( temp_g ) );
}







//================================================================
// resample separate
//================================================================

__global__ void resampleAreaParallelSeparate_x
	(
		const float* in_g,
		float* out_g,
		int nx,
		int ny,
		float hx,
		int pitchf1_in,
		int pitchf1_out,
		float factor = 0.0f
	)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int index = ix + iy * pitchf1_out; // global index for out image
	
	if( factor == 0.0f )
		factor = 1 / hx;

	if( ix < nx && iy < ny)
	{
		// initialising out
		out_g[ index ] = 0.0f;
		
		float px = (float)ix * hx;
		
		float left = ceil(px) - px;
		if(left > hx) left = hx;
		
		float midx  = hx - left;
		float right = midx - floorf(midx);
		
		midx = midx - right;
		
		if( left > 0.0f )
		{
			out_g[index] += in_g[ iy * pitchf1_in + (int)floor(px) ] * left * factor;
			px += 1.0f;
		}
		while( midx > 0.0f )
		{
			out_g[index] += in_g[ iy * pitchf1_in + (int)floor(px) ] * factor;
			px += 1.0f;
			midx -= 1.0f;
		}
		if( right > RESAMPLE_EPSILON )
		{
			out_g[index] += in_g[ iy * pitchf1_in + (int)floor(px) ] * right * factor;
		}
	}
}

__global__ void resampleAreaParallelSeparate_y
	(
		const float* in_g,
		float* out_g,
		int nx,
		int ny,
		float hy,
		int pitchf1_out,
		float factor = 0.0f // need
	)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int index = ix + iy * pitchf1_out; // global index for out image
	
	if( factor == 0.0f )
		factor = 1.0f / hy;
	
	if( ix < nx && iy < ny ) // guards
	{
		out_g[index] = 0.0f;
		
		float py = (float)iy * hy;
		float top = ceil(py) - py;
		
		if( top > hy )
			top = hy;
		
		float midy = hy - top;
		
		float bottom = midy - floorf(midy);
		midy = midy - bottom;
		
		if( top > 0.0f )
		{
			out_g[index] += in_g[(int)floor(py) * pitchf1_out + ix ] * top * factor;
			py += 1.0f;
		}
		while( midy > 0.0f )
		{
			out_g[index] += in_g[(int)floor(py) * pitchf1_out + ix ] * factor;
			py += 1.0f;
			midy -= 1.0f;
		}
		if( bottom > RESAMPLE_EPSILON )
		{
			out_g[index] += in_g[(int)floor(py) * pitchf1_out + ix ] * bottom * factor;
		}
	}
}


void resampleAreaParallelSeparate (
		const float *in_g,
		float *out_g,
		int nx_in,
		int ny_in,
		int pitchf1_in,
		int nx_out,
		int ny_out,
		int pitchf1_out,
		float *help_g,
		float scalefactor
	)
{
	// can reduce no of blocks for first pass
	int gridsize_x = ((nx_out - 1) / LO_BW) + 1;
	int gridsize_y = ((ny_in - 1) / LO_BH) + 1;
	
	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	//-----------------------
	// memory preparation
	//-----------------------

	// allocate helper array, if not given
	bool selfalloctemp = help_g == NULL;
	if (selfalloctemp)
	{
		// pitch should be the same as output pitch
		cuda_malloc2D( (void**)&help_g, std::max( nx_in, nx_out ), std::max( ny_in, ny_out ),
				1, sizeof(float), &pitchf1_out );
	}
	
	float hx = (float)nx_in / (float)nx_out;
	float factor = (float)(nx_out)/(float)(nx_in);
	
	resampleAreaParallelSeparate_x<<< dimGrid, dimBlock >>>( in_g, help_g, nx_out, ny_in,
				hx, pitchf1_in, pitchf1_out, factor);
	
	// this cost us a lot of time -> resize grid to y_out
	gridsize_y = (ny_out % LO_BH) ? ((ny_out / LO_BH)+1) : (ny_out / LO_BH);
	dimGrid = dim3( gridsize_x, gridsize_y );
	
	float hy = (float)ny_in / (float)ny_out;
	factor = scalefactor*(float)ny_out / (float)ny_in;
	
	resampleAreaParallelSeparate_y<<< dimGrid, dimBlock >>>( help_g, out_g, nx_out, ny_out,
			hy, pitchf1_out, factor );

	//-----------------------
	// cleanup
	//-----------------------

	if( selfalloctemp )
		cutilSafeCall( cudaFree( help_g ) );
}

//================================================================
// resample adjoined
//================================================================

void resampleAreaParallelSeparateAdjoined(const float *in_g, float *out_g,
		int nx_in, int ny_in, int pitchf1_in, int nx_out, int ny_out,
		int pitchf1_out, float *help_g, float scalefactor)
{	
	int gridsize_x = ( nx_out % LO_BW ) ? (nx_out / LO_BW) + 1 : (nx_out / LO_BW);
	int gridsize_y = ( ny_in % LO_BH ) ? ( ny_in / LO_BH ) + 1 : ( ny_in / LO_BH );

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	//-----------------------
	// memory preparation
	//-----------------------

	// allocate helper array, if not given
	bool selfalloctemp = help_g == NULL;
	if (selfalloctemp)
	{
		// pitch is the same as output pitch
		cuda_malloc2D( (void**)&help_g, std::max( nx_in, nx_out ), std::max( ny_in, ny_out ),
				1, sizeof(float), &pitchf1_out );
	}

	float hx = (float)(nx_in)/(float)(nx_out);
	resampleAreaParallelSeparate_x<<<dimGrid, dimBlock>>>( in_g, help_g, nx_out, ny_in, hx, pitchf1_in, pitchf1_out, 1.0f);
		
	gridsize_y = ( ny_out % LO_BH ) ? ( ny_out / LO_BH ) + 1 : ( ny_out / LO_BH );
	dimGrid = dim3( gridsize_x, gridsize_y );
	
	float hy = (float)(ny_in)/(float)(ny_out);
	
	resampleAreaParallelSeparate_y<<<dimGrid, dimBlock>>>( help_g, out_g, nx_out, ny_out, hy, pitchf1_out, scalefactor );	
	
	//-----------------------
	// cleanup
	//-----------------------

	if( selfalloctemp )
		cutilSafeCall( cudaFree( help_g ) );
}



//================================================================
// simple add sub and set kernels
//================================================================


__global__ void addKernel(const float *increment_g, float *accumulator_g,
		int nx, int ny, int pitchf1)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y * pitchf1 + x;
	
	if( x < nx && y < ny )
	{
		accumulator_g[idx] += increment_g[idx];
	}
}

__global__ void subKernel(const float *increment_g, float *accumulator_g,
		int nx, int ny, int pitchf1)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y * pitchf1 + x;
	
	if( x < nx && y < ny )
	{
		accumulator_g[idx] -= increment_g[idx];
	}
}

__global__ void setKernel(float *field_g, int nx, int ny, int pitchf1,
		float value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y * pitchf1 + x;
	
	if( x < nx && y < ny )
	{
		field_g[idx] = value;
	}
}

// philipp's great debugging kernel, altered to produce a pattern of diagonal lines
__global__ void debugKernel( float *field_g, int nx, int ny, int pitchf1 )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y * pitchf1 + x;

	if( x < nx && y < ny )
	{
		if( (x + y) % 6 < 3 )
			field_g[idx] = 0.0f;
		else
			field_g[idx] = 255.0f;
	}
}

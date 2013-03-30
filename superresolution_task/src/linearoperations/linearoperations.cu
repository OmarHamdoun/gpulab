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
#include <iostream>
#include <linearoperations/linearoperations.cuh>

#define SHARED_MEM 0

// TEXTURES

#define TEXTURE_OFFSET      0.5f  // offset for indexing textures
cudaChannelFormatDesc linearoperation_float_tex = cudaCreateChannelDesc<float>();
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;

// CONSTANT MEMORY KERNELS

#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   	21    // maximum allowed kernel radius + 1

__constant__ float constKernelX[MAXKERNELSIZE];
__constant__ float constKernelY[MAXKERNELSIZE];
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
// backward warping
//================================================================

// TODO: change global memory to texture
/*
 * Texture is faster than global memory, as memory access may be random
 * Access locations depending on flow direction
 */
__global__ void backwardRegistrationBilinearValueTexKernel (
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
		float hx_1 = 1.0f / hx;
		float hy_1 = 1.0f / hy;
	
		float ii_fp = x + (flow1_g[y * pitchf1_in + x] * hx_1);
		float jj_fp = y + (flow2_g[y * pitchf1_in + x] * hy_1);
	
		if( (ii_fp < 0.0f) || (jj_fp < 0.0f)
					 || (ii_fp > (float)(nx - 1)) || (jj_fp > (float)(ny - 1)) )
		{
			out_g[y * pitchf1_out + x] = value;
		}
		else if( !isfinite( ii_fp ) || !isfinite( jj_fp ) )
		{
			//fprintf(stderr,"!");
			out_g[ y * pitchf1_out + x] = value;
		}
		else
		{
			int xx = (int)ii_fp;
			int yy = (int)jj_fp;
	
			int xx1 = xx == nx - 1 ? xx : xx + 1;
			int yy1 = yy == ny - 1 ? yy : yy + 1;
	
			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;
	
			out_g[y * pitchf1_out + x] =
					(1.0f - xx_rest) * (1.0f - yy_rest) * in_g[yy  * pitchf1_in + xx]
					+ xx_rest * (1.0f - yy_rest)        * in_g[yy  * pitchf1_in + xx1]
					+ (1.0f - xx_rest) * yy_rest        * in_g[yy1 * pitchf1_in + xx]
					+ xx_rest * yy_rest                 * in_g[yy1 * pitchf1_in + xx1];
		}
	}
}

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
		float ii_fp = x + (flow1_g[y * pitchf1_in + x] / hx);
		float jj_fp = y + (flow2_g[y * pitchf1_in + x] / hy);
	
		if( (ii_fp < 0.0f) || (jj_fp < 0.0f)
					 || (ii_fp > (float)(nx - 1)) || (jj_fp > (float)(ny - 1)) )
		{
			out_g[y * pitchf1_out + x] = value;
		}
		else if( !isfinite( ii_fp ) || !isfinite( jj_fp ) )
		{
			//fprintf(stderr,"!");
			out_g[ y * pitchf1_out + x] = value;
		}
		else
		{
			int xx = (int)ii_fp;
			int yy = (int)jj_fp;
	
			int xx1 = xx == nx - 1 ? xx : xx + 1;
			int yy1 = yy == ny - 1 ? yy : yy + 1;
	
			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;
	
			out_g[y * pitchf1_out + x] =
					(1.0f - xx_rest) * (1.0f - yy_rest) * in_g[yy  * pitchf1_in + xx]
					+ xx_rest * (1.0f - yy_rest)        * in_g[yy  * pitchf1_in + xx1]
					+ (1.0f - xx_rest) * yy_rest        * in_g[yy1 * pitchf1_in + xx]
					+ xx_rest * yy_rest                 * in_g[yy1 * pitchf1_in + xx1];
		}
	}
}


void backwardRegistrationBilinearValueTex (
		const float* in_g,		// _u_overrelaxed
		const float* flow1_g,	// flow->u1
		const float* flow2_g,	// flow->u2
		float* out_g,			// _help1
		float value,			// 0.0f
		int nx,
		int ny,
		int pitchf1_in,
		int pitchf1_out,
		float hx,				// 1.0f
		float hy				// 1.0f
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	


#if SHARED_MEM
		// TODO: binding of texture

		backwardRegistrationBilinearValueTexKernel<<<dimGrid, dimBlock>>>(
				in_g,
				flow1_g,
				flow2_g,
				out_g,
				value,
				nx,
				ny,
				pitchf1_in,
				pitchf1_out,
				hx,
				hy
			);

		// TODO: release texture
#else
		backwardRegistrationBilinearValueTexKernel_gm<<<dimGrid, dimBlock>>>(
				in_g,
				flow1_g,
				flow2_g,
				out_g,
				value,
				nx,
				ny,
				pitchf1_in,
				pitchf1_out,
				hx,
				hy
			);
#endif

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
						|| yy > (float) (ny - 1)) ?
						constant_g[y * pitchf1_in + x] :
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

void backwardRegistrationBilinearFunctionTex(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy)
{
	// ### Implement me, if you want ###
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
	const unsigned int idx = y * pitchf1 + x;
	
	if( x < nx && y < ny )
	{
		
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
			float out_xy   = in_g[idx] * (1.0f - xxf) * (1.0f - yyf);
			float out_x1y  = in_g[idx] * xxf * (1.0f - yyf);
			float out_xy1  = in_g[idx] * (1.0f - xxf) * yyf;
			float out_x1y1 = in_g[idx] * xxf * yyf;		
				
			// eject the warp core!
			// avoid race conditions by use of atomic operations
			atomicAdd( out_g + (yyi * pitchf1 + xxi),           out_xy );
			atomicAdd( out_g + (yyi * pitchf1 + xxi + 1),       out_x1y );
			atomicAdd( out_g + ((yyi + 1) * pitchf1 + xxi),     out_xy1 );
			atomicAdd( out_g + ((yyi + 1) * pitchf1 + xxi + 1), out_x1y1 );

			// TODO: think about hierarchical atomics
			// problem: target coordinates can be anywhere on image,
			// so shared memory per block is limited reasonable	
		}
	}
}



/*
 * Forward warping
 */
// TODO: try textures
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
	// this cost us hours again, not to do it in the warpi kernel... 
	setKernel <<< dimGrid, dimBlock >>> ( out_g, nx, ny, pitchf1, 0.0f );

	// invoke atomic warp kernel on gpu
	foreward_warp_kernel_atomic <<< dimGrid, dimBlock >>> ( flow1_g, flow2_g, in_g, out_g, nx, ny, pitchf1 );
}



//================================================================
// gaussian blur (mirrored) - global memory
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

	// allocate mask memory, if not given
	bool selfallocmask = mask == 0;

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin; // pitch is the same as pitchf1
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}

	//-----------------------
	// gauss preparation
	//-----------------------
	
	// set radius automatically, if not given
	if( radius == 0 )
	{
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)( 3.0f * maxsigma );
	}

	sigmax = 1.0f / (sigmax * sigmax);
	sigmay = 1.0f / (sigmay * sigmay);


	//-----------------------
	// kernel preparation
	//-----------------------
	
	// compute kernel only once and keep it in constant memory
	if( !constant_kernel_bound )
	{
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

		constant_kernel_bound = true;
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
	
	// free self allocated memory
	if( selfallocmask )
		delete [] mask;

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
		float result = constKernelX[0] * tex2D( tex_linearoperation, (float)x + 0.5f, (float)y + 0.5f );
		
		for( int i = 1; i <= radius; ++i )
		{
			// no offset of 0.5f required, as filter mode point is used
			// used address mode: mirror

			// computing offset symmetrically
			result += constKernelX[i] * (
					tex2D( tex_linearoperation, (float)(x - i) + 0.5f, (float)y + 0.5f ) + // left part of kernel
					tex2D( tex_linearoperation, (float)(x + i) + 0.5f, (float)y + 0.5f )   // right part of the kernel
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
		float result = constKernelY[0] * tex2D( tex_linearoperation, (float)x + 0.5f, (float)y + 0.5f );
		
		// computing offset symmetrically
		for( int i = 1; i <= radius; ++i )
		{
			// no offset of 0.5f for texture required, as filter mode point is used
			// used address mode: mirror

			result += constKernelY[i] * (
					tex2D( tex_linearoperation, (float)x + 0.5f, (float)(y - i) + 0.5f ) + // upper part of kernel
					tex2D( tex_linearoperation, (float)x + 0.5f, (float)(y + i) + 0.5f )   // lower part of kernel
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

	// allocate mask memory, if not given
	bool selfallocmask = mask == 0;

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin; // pitch is the same as pitchf1
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}

	//-----------------------
	// gauss preparation
	//-----------------------
	
	// set radius automatically, if not given
	if( radius == 0 )
	{
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)( 3.0f * maxsigma );
	}

	sigmax = 1.0f / (sigmax * sigmax);
	sigmay = 1.0f / (sigmay * sigmay);


	//-----------------------
	// kernel preparation
	//-----------------------
	
	// compute kernel only once and keep it in constant memory
	if( !constant_kernel_bound )
	{
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

		constant_kernel_bound = true;
	}

	//-----------------------
	// convolution
	//-----------------------

	// prepare texture
	setTexturesLinearOperations( 1, 1 ); // filter mode: point (no offset necessary), address mode: mirror

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
	if( selfallocmask )
		delete [] mask;

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
			// using pitchf1_in instead of nx_orig in original code
			out_g[index] += in_g[ iy * pitchf1_in + (int)floor(px) ] * left * factor; // look out for conversion of coordinates
			px += 1.0f;
		}
		while( midx > 0.0f )
		{
			// using pitchf1_in instead of nx_orig in original code
			out_g[index] += in_g[ iy * pitchf1_in + (int)floor(px) ] * factor;
			px += 1.0f;
			midx -= 1.0f;
		}
		if( right > RESAMPLE_EPSILON )
		{
			// using pitchf1_in instead of nx_orig in original code
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
	// used pitch instead  of blockDim.x
	
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
			// using pitch for helper array since these all arrays have same pitch
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
	// TODO: add allocation of help_g, if not allocated	

	// can reduce no of blocks for first pass
	int gridsize_x = ((nx_out - 1) / LO_BW) + 1;
	int gridsize_y = ((ny_in - 1) / LO_BH) + 1;
	
	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	
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

	// TODO: free help_g if self allocated
}

//================================================================
// resample adjoined
//================================================================

void resampleAreaParallelSeparateAdjoined(const float *in_g, float *out_g,
		int nx_in, int ny_in, int pitchf1_in, int nx_out, int ny_out,
		int pitchf1_out, float *help_g, float scalefactor)
{	
	/*  Here, 
	 * in_g = q_g[k]   	nx_orig, ny_orig, pitchf1_orig
	 * out_g = temp1_g 	nx,ny,pitchf1
	 * (nx_in, ny_in) = ( nx_orig, ny_orig)
	 * pitchf1_in  = pitchf1_orig
	 * (nx_out,ny_out) = (nx, ny )
	 * pitchf1_out = pitchf1
	 * help_g = temp4_g
	 * scalefactor = 1.00f (default value)
	 */
	
	// TODO: add allocation of help_g, if not allocated	
	
	// AM HELL SCARED TO WRITE THIS METHOD DUE TO BLUNDER IN LAST :p STEFAN AN PHILIP, PLZ CROSSCHECK	
	int gridsize_x = ( nx_out % LO_BW ) ? (nx_out / LO_BW) + 1 : (nx_out / LO_BW);
	int gridsize_y = ( ny_in % LO_BH ) ? ( ny_in / LO_BH ) + 1 : ( ny_in / LO_BH );
	
	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );
	
	float hx = (float)(nx_in)/(float)(nx_out);
	resampleAreaParallelSeparate_x<<<dimGrid, dimBlock>>>( in_g, help_g, nx_out, ny_in, hx, pitchf1_in, pitchf1_out, 1.0f);
	//CPU//resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,(float)(nx_in)/(float)(nx_out),nx_in,1.0f);
	
	gridsize_y = ( ny_out % LO_BH ) ? ( ny_out / LO_BH ) + 1 : ( ny_out / LO_BH );
	dimGrid = dim3( gridsize_x, gridsize_y );
	
	float hy = (float)(ny_in)/(float)(ny_out);
	
	resampleAreaParallelSeparate_y<<<dimGrid, dimBlock>>>( help_g, out_g, nx_out, ny_out, hy, pitchf1_out, scalefactor );	
	//CPU//resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,(float)(ny_in)/(float)(ny_out),scalefactor);

	// TODO: free help_g if self allocated
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

// TODO: remove
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

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

#define SHARED_MEM 0

cudaChannelFormatDesc linearoperation_float_tex =
		cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;

#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   21    // maximum allowed kernel radius + 1

__constant__ float constKernel[MAXKERNELSIZE];

void gpu_bindConstantMemory( const float *kernel, int size ) 
{
	fprintf( stderr, "\n\nBINDING TEXTURE MEMORY..." );
	cutilSafeCall( cudaMemcpyToSymbol( constKernel, kernel, size * sizeof(float) ) );
	fprintf( stderr, "\n\nTEXTURE MEMORY BOUND" );
}


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

//================================================================
// backward warping
//================================================================

// TODO: change global memory to texture
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
	
		float ii_fp = x + (flow1_g[y * nx + x] * hx_1);
		float jj_fp = y + (flow2_g[y * nx + x] * hy_1);
	
		if( (ii_fp < 0.0f) || (jj_fp < 0.0f)
					 || (ii_fp > (float)(nx - 1)) || (jj_fp > (float)(ny - 1)) )
		{
			out_g[y*nx+x] = value;
		}
		else if( !isfinite( ii_fp ) || !isfinite( jj_fp ) )
		{
			//fprintf(stderr,"!");
			out_g[ y * nx + x] = value;
		}
		else
		{
			int xx = (int)ii_fp;
			int yy = (int)jj_fp;
	
			int xx1 = xx == nx - 1 ? xx : xx + 1;
			int yy1 = yy == ny - 1 ? yy : yy + 1;
	
			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;
	
			out_g[y * nx + x] =
					(1.0f - xx_rest) * (1.0f - yy_rest) * in_g[yy * nx + xx]
					+ xx_rest * (1.0f - yy_rest)        * in_g[yy * nx + xx1]
					+ (1.0f - xx_rest) * yy_rest        * in_g[yy1 * nx + xx]
					+ xx_rest * yy_rest                 * in_g[yy1 * nx + xx1];
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
			
	// reset shared memory to zero
	out_g[idx] = 0.0f;
		
	// calculate target coordinates: coords + flow values
	const float xx = (float)x + flow1_g[idx];
	const float yy = (float)y + flow2_g[idx];
	
	// continue only if target area inside image
	if(
			xx >= 0.0f &&
			xx <= (float)(nx - 2) &&
			yy >= 0.0f &&
			yy <= (float)(ny - 2))
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
		atomicAdd( out_g + (yyi * nx + xxi),           out_xy );
		atomicAdd( out_g + (yyi * nx + xxi + 1),       out_x1y );
		atomicAdd( out_g + ((yyi + 1) * nx + xxi),     out_xy1 );
		atomicAdd( out_g + ((yyi + 1) * nx + xxi + 1), out_x1y1 );
		
		// TODO: think about hierarchical atomics
		// problem: target coordinates can be anywhere on image,
		// so shared memory per block is limited reasonable
		
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

	// invoke atomic warp kernel on gpu
	foreward_warp_kernel_atomic <<< dimGrid, dimBlock >>> ( flow1_g, flow2_g, in_g, out_g, nx, ny, pitchf1 );
}



//================================================================
// gaussian blur (mirrored)
//================================================================


/*
 * gaussian blur with mirrored border
 * 
 * global memory
 */
/* __global__ void gaussBlurSeparateMirrorGpuKernel_global (
		float* in_g,
		float* out_g,
		int nx,
		int ny,
		int pitchf1,
		float sigmax,
		float sigmay,
		int radius,
		float* mask
	)
{
	// get thread coordinates and index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	//const unsigned int idx = y * pitchf1 + x;
	
	float result;

	// todo: currently assuming that temp_g is given
	//bool selfalloctemp = temp_g == NULL;
	//if( selfalloctemp )
	//	temp_g = new float[nx*ny];

	sigmax = 1.0f / (sigmax * sigmax);
	sigmay = 1.0f / (sigmay * sigmay);



	// convolution
	result = mask[0] * in_g[y * pitchf1 + x];

	for( int i = 1; i <= radius; i++ )
	{
		result += mask[i] * (
				(x - i >= 0) ? 
				in_g[y * pitchf1 + (x - i)] :
				in_g[y * pitchf1 + (-1 - (x-i))] 
+
				(x + i < nx) ?
				in_g[y * pitchf1 + (x + i)] : 
				in_g[y * pitchf1 + (nx - (x+i - nx-1))] 
			);
	}
	
	out_g[y * pitchf1 + x] = result;	



	( (y-i >= 0) ? in_g[(y-i) * pitchf1 + x] : in_g[(-1 - (y-i)) * pitchf1 + x]) +
	( (y+i < ny) ? in_g[(y+i) * pitchf1 + x] : in_g[(ny - (y+i - ny-1)) * pitchf1 + x])




}
*/







/*
 * Convolution kernel for gaussian blur in x direction
 *
 * Of course it is fine to have a universal function,
 * but specialized ones are faster. A similar kernel
 * could be used with both X radius and Y radius, first
 * invoked with Xradius = radius and Yradius = 1, and
 * then the other way round.
 * Here the kernel has been splittet into two similar
 * ones, the first with a hardcoded Yradius of 1, the
 * second with a hardcoded Xradius of 1.
 */
__global__ void gaussBlurConvolutionSeparatedMirrorGpu_x
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
	int chunkWidth  = LO_BW + ( radius << 1 );
	int chunkHeight = LO_BH;
	
	// create shared memory variable
	extern __shared__ float sharedImage[];
	
	int numThreads = LO_BW * LO_BH;
	
	// get block and chunk position (upper left corner of the chunk in the global image)
	int blockX = blockIdx.x * LO_BW;
	int blockY = blockIdx.y * LO_BH;
	int chunkX = blockX - radius;
	int chunkY = blockY;
	
	// chunk offset if padding reaches out of image
	int chunkOffX = chunkX < 0 ? -1 * chunkX : 0;
	//int chunkOffY = 0;

	// get updated chunk pos (inside image)
	chunkX += chunkOffX;
	//chunkY += chunkOffY;
	
	// get actual chunk size (inside image)
	int actualChunkWidth  = min( chunkX + chunkWidth  - chunkOffX, iWidth  ) - chunkX;
	int actualChunkHeight = min( chunkY + chunkHeight            , iHeight ) - chunkY;
	
	// number of pixels
	int pixelNum = actualChunkWidth * actualChunkHeight;


	// the offsets are relative to the top left corner of the chunk
	int offsetX, offsetY;

	// load pixels from global to shared memory
	for( int i = localThreadIndex; i < pixelNum; i += numThreads )
	{
		offsetX = i % actualChunkWidth;
		offsetY = i / actualChunkWidth;
		
		sharedImage[ (chunkOffX + offsetX) + chunkWidth * offsetY ] = 
				inputImage[ (chunkX + offsetX) + iPitch * (chunkY + offsetY) ];
	}

	// synchronize threads
	__syncthreads();
	
	
	//==================================
	// convolution
	//==================================
	
	// continue calculation only for pixels inside the image
	if( x < iWidth && y < iHeight )
	{
		// determine kernel size
		//const int kWidth  = radius + 1;
		//const int kHeight = 1;
		
		// the kernel is symmetric and therefore only the center and the right half is given
		// calculate center outside of the loop (otherwise it would be computed twice)

		// shared memory coordinates
		int tx = threadIdx.x + radius;
		int ty = threadIdx.y;

		float value = constKernel[0] * sharedImage[ty * chunkWidth + tx]; // temp var for output pixel

		// kernel loop
		for( int i = 1; i <= radius; ++i )
		{
			value += constKernel[i] * (
				
				// left side of kernel
				(x - i >= 0) ? 
					sharedImage[ty * chunkWidth + (tx - i)] :
					sharedImage[ty * chunkWidth + (-1 - (tx - i))] // border condition: mirroring

					+

				// right side of kernel
				(x + i < iWidth) ?
					sharedImage[ty * chunkWidth + (tx + i)] : 
					sharedImage[ty * chunkWidth + (iWidth - (tx+i - iWidth-1))] // border condition: mirroring
			);
		}
		// end of kernel loop
	
		// write to output image
		outputImage[ y * iPitch + x ] = value;
	}
} 


/*
 * Convolution kernel for gaussian blur in y direction
 */
__global__ void gaussBlurConvolutionSeparatedMirrorGpu_y
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
	int chunkWidth  = LO_BW;
	int chunkHeight = LO_BH + ( radius << 1 );
	
	// create shared memory variable
	extern __shared__ float sharedImage[];
	
	int numThreads = LO_BW * LO_BH;
	
	// get block and chunk position (upper left corner of the chunk in the global image)
	int blockX = blockIdx.x * LO_BW;
	int blockY = blockIdx.y * LO_BH;
	int chunkX = blockX;
	int chunkY = blockY - radius;
	
	// chunk offset if padding reaches out of image
	//int chunkOffX = 0;
	int chunkOffY = chunkY < 0 ? -1 * chunkY : 0;

	// get updated chunk pos (inside image)
	//chunkX += chunkOffX;
	chunkY += chunkOffY;
	
	// get actual chunk size (inside image)
	int actualChunkWidth  = min( chunkX + chunkWidth             , iWidth  ) - chunkX;
	int actualChunkHeight = min( chunkY + chunkHeight - chunkOffY, iHeight ) - chunkY;
	
	// number of pixels
	int pixelNum = actualChunkWidth * actualChunkHeight;


	// the offsets are relative to the top left corner of the chunk
	int offsetX, offsetY;

	// load pixels from global to shared memory
	for( int i = localThreadIndex; i < pixelNum; i += numThreads )
	{
		offsetX = i % actualChunkWidth;
		offsetY = i / actualChunkWidth;
		
		sharedImage[ offsetX + chunkWidth * (chunkOffY + offsetY) ] = 
				inputImage[ (chunkX + offsetX) + iPitch * (chunkY + offsetY) ];
	}

	// synchronize threads
	__syncthreads();
	
	
	//==================================
	// convolution
	//==================================
	
	// continue calculation only for pixels inside the image
	if( x < iWidth && y < iHeight )
	{
		// determine kernel size
		//const int kWidth  = radius + 1;
		//const int kHeight = 1;
		
		// the kernel is symmetric and therefore only the center and the right half is given
		// calculate center outside of the loop (otherwise it would be computed twice)

		// shared memory coordinates
		int tx = threadIdx.x;
		int ty = threadIdx.y + radius;

		float value = constKernel[0] * sharedImage[ty * chunkWidth + tx]; // temp var for output pixel

		// kernel loop
		for( int i = 1; i <= radius; ++i )
		{
			value += constKernel[i] * (
				
				// left side of kernel
				(y - i >= 0) ? 
					sharedImage[(ty - i) * chunkWidth + tx] :
					sharedImage[(-1 - (ty - i)) * chunkWidth + tx] // border condition: mirroring

					+

				// right side of kernel
				(y + i < iHeight) ?
					sharedImage[(ty + i) * chunkWidth + tx] :
					sharedImage[(iHeight - (ty + i - iHeight-1)) * chunkWidth + tx] // border condition: mirroring
			);
		}
		// end of kernel loop
	
		// write to output image
		outputImage[ y * iPitch + x ] = value;
	}
}



// TODO: test performance of texture instead of dynamically allocated shared memory
/*
 * wrapping method for gaussian convolution gpu kernel
 *
 * mask is supposed to be a CPU pointer!
 */
void gaussBlurSeparateMirrorGpu (
		float* in_g,
		float* out_g,
		int nx,
		int ny,
		int pitchf1,
		float sigmax,
		float sigmay,
		int radius,
		float* temp_g,
		float* mask		// pointer to CPU memory!
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );


	// todo: necessary? => copy memory? (swaping pointers here not possible)
	// if( sigmax <= 0.0f || sigmay <= 0.0f || radius < 0 )
	//	 return;

	// allocate mask memory, if not given
	bool selfallocmask = mask == NULL;
	if(selfallocmask)
		mask = new float[radius + 1];

	// allocate helper array, if not given
	bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		int pitchBin;
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}
	
	// set radius automatically, if not given
	if( radius == 0 )
	{
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)( 3.0f * maxsigma );
	}

	sigmax = 1.0f / (sigmax * sigmax);
	sigmay = 1.0f / (sigmay * sigmay);

	//---------------------
	// gauss in x direction
	//---------------------
	
	// prepare gaussian kernel (1D) for x direction
	float sum = 1.0f;
	mask[0] = 1.0f;
	for( int gx = 1; gx <= radius; ++gx )
	{
		mask[gx] = exp( -0.5f * ( (float)(gx * gx) * sigmax) );
		sum += 2.0f * mask[gx];
	}
	// normalize kernel
	for( int gx = 0; gx <= radius; ++gx )
	{
		mask[gx] /= sum;
	}

	// bind kernel to constant memory
	gpu_bindConstantMemory( mask, radius + 1 );

	// get size of shared memory chunk for x convolution dynamically
	int sharedMemorySize =  ( LO_BW + ( radius << 1) ) * LO_BH * sizeof(float);

	// invoke gauss kernel on gpu for convolution in x direction
	// MAXKERNELSIZE and MAXKERNELRADIUS do not allow to combine x and y in one kernel
	gaussBlurConvolutionSeparatedMirrorGpu_x <<< dimGrid, dimBlock, sharedMemorySize >>> ( in_g, temp_g, nx, ny, pitchf1, radius );

	//---------------------
	// gauss in y direction
	//---------------------

	mask[0] = sum = 1.0f;

	// prepare gaussian kernel (1D)
	// todo: move to shared memory, if computed in threads
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
	gpu_bindConstantMemory( mask, radius + 1 );

	// update size of dynamically allocated shared memory chunk for y convolution
	sharedMemorySize = LO_BW * ( LO_BH + ( radius << 1) ) * sizeof(float);

	// invoke gauss kernel on gpu for convolution in y direction
	gaussBlurConvolutionSeparatedMirrorGpu_y <<< dimGrid, dimBlock, sharedMemorySize >>> ( in_g, temp_g, nx, ny, pitchf1, radius );
	
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
	// helper array is already allocated on the GPU as _b1, now help_g

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
	
	// ### Implement me ###		
	// help_g is already allocated on GPU global memory, no need to check
	
	// AM HELL SCARED TO WRITE THIS METHOD DUE TO BLUNDER IN LAST :p STEFAN AN PHILIP, PLZ CROSSCHECK	
	int xBlocks = ( nx_out % LO_BW ) ? (nx_out / LO_BW) + 1 : (nx_out / LO_BW);
	int yBlocks = ( ny_in % LO_BH ) ? ( ny_in / LO_BH ) + 1 : ( ny_in / LO_BH );
	
	dim3 dimGrid( xBlocks, yBlocks );
	dim3 dimBlock( LO_BW, LO_BH );
	
	float hx = (float)(nx_in)/(float)(nx_out);
	resampleAreaParallelSeparate_x<<<dimGrid, dimBlock>>>( in_g, help_g, nx_out, ny_in, hx, pitchf1_in, pitchf1_out, 1.0f);
	//CPU//resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,(float)(nx_in)/(float)(nx_out),nx_in,1.0f);
	
	yBlocks = ( ny_out % LO_BH ) ? ( ny_out / LO_BH ) + 1 : ( ny_out / LO_BH );
	dimGrid = dim3( xBlocks, yBlocks );
	
	float hy = (float)(ny_in)/(float)(ny_out);
	
	resampleAreaParallelSeparate_y<<<dimGrid, dimBlock>>>( help_g, out_g, nx_out, ny_out, hy, pitchf1_out, scalefactor );	
	//CPU//resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,(float)(ny_in)/(float)(ny_out),scalefactor);
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

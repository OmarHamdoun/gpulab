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
		int radius,
		float* kernel // gm
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

		float value = kernel[0] * sharedImage[ty * chunkWidth + tx]; // temp var for output pixel //constKernel

		// kernel loop
		for( int i = 1; i <= radius; ++i )
		{
			value += kernel[i] * ( //constKernel
				
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
		int radius,
		float* kernel // gm
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

		float value = kernel[0] * sharedImage[ty * chunkWidth + tx]; // temp var for output pixel //constKernel

		// kernel loop
		for( int i = 1; i <= radius; ++i )
		{
			value += kernel[i] * ( //constKernel
				
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
void gaussBlurSeparateMirrorGpu
	(
		float* in_g,	// input image
		float* out_g,	// convoluted output image
		int nx,
		int ny,
		int pitchf1,	// pitch for this image
		float sigmax,	// gauss parameters
		float sigmay,
		int radius,
		float* temp_g,
		float* mask		// gauss kernel memory: pointer to CPU memory!
	)
{
	if( sigmax <= 0.0f || sigmay <= 0.0f || radius < 0 )
	{
		return; // not going for a kernel call
		// TODO: copy memory
	}

	printf("\ncalled gaussblur");
	// block and grid size
	int gridsize_x = ((nx - 1) / LO_BW) + 1;
	int gridsize_y = ((ny - 1) / LO_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( LO_BW, LO_BH );

	// allocate mask memory, if not given
	bool selfallocmask = mask == 0;
	if(selfallocmask)
	{
		printf("\nallocating mask");
		mask = new float[radius + 1];
	}

	// allocate helper array, if not given
	// TODO
	/*bool selfalloctemp = temp_g == NULL;
	if (selfalloctemp)
	{
		printf("\nallocating temp array gpu memory");
		int pitchBin;
		cuda_malloc2D( (void**)&temp_g, nx, ny, 1, sizeof(float), &pitchBin );
	}*/
	
	// set radius automatically, if not given
	if( radius == 0 )
	{
		printf("\nsetting radius");
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
	printf("\npreparing x kernel");
	for( int x = 1; x <= radius; ++x )
	{
		mask[x] = exp( -0.5f * ( (float)(x * x) * sigmax) );
		sum += 2.0f * mask[x];
	}
	printf("\nnormalising x kernel");
	// normalize kernel
	for( int x = 0; x <= radius; ++x )
	{
		mask[x] /= sum;
	}

	printf("\nallocating kernel gpu memory");
	float* kernel_g; // gm
	int maskBytes = (radius + 1) * sizeof(float); // gm
	cutilSafeCall( cudaMalloc ( (void**)&kernel_g, maskBytes ) ); // gm
	printf("\ncopy kernel to gm");
	// bind kernel to constant memory
	//gpu_bindConstantMemory( mask, radius + 1 );
	cutilSafeCall( cudaMemcpy ( kernel_g, mask, maskBytes, cudaMemcpyHostToDevice ) ); // gm

	printf("\ncalculating sm size");
	// get size of shared memory chunk for x convolution dynamically
	int sharedMemorySize =  ( LO_BW + ( radius << 1) ) * LO_BH * sizeof(float);

	printf("\ncalling gauss x");
	// invoke gauss kernel on gpu for convolution in x direction
	// MAXKERNELSIZE and MAXKERNELRADIUS do not allow to combine x and y in one kernel
	gaussBlurConvolutionSeparatedMirrorGpu_x <<< dimGrid, dimBlock, sharedMemorySize >>> ( in_g, temp_g, nx, ny, pitchf1, radius, kernel_g );

	//---------------------
	// gauss in y direction
	//---------------------

	mask[0] = sum = 1.0f;

	printf("\npreparing y kernel");
	// prepare gaussian kernel (1D)
	// todo: move to shared memory, if computed in threads
	for( int gx = 1; gx <= radius; ++gx )
	{
		mask[gx] = exp( -0.5f * ( (float)(gx * gx) * sigmay) );
		sum += 2.0f * mask[gx];
	}
	// normalize kernel
	printf("\nnormalizing y kernel");
	for( int gx = 0; gx <= radius; ++gx )
	{
		mask[gx] /= sum;
	}
	
	printf("\ncopy kernel to gm");
	// bind kernel to constant memory
	//gpu_bindConstantMemory( mask, radius + 1 );
	cutilSafeCall( cudaMemcpy ( kernel_g, mask, maskBytes, cudaMemcpyHostToDevice ) ); // gm

	printf("\ncalculating sm size");
	// update size of dynamically allocated shared memory chunk for y convolution
	sharedMemorySize = LO_BW * ( LO_BH + ( radius << 1) ) * sizeof(float);

	printf("\ncalling gauss y");
	// invoke gauss kernel on gpu for convolution in y direction
	gaussBlurConvolutionSeparatedMirrorGpu_y <<< dimGrid, dimBlock, sharedMemorySize >>> ( in_g, temp_g, nx, ny, pitchf1, radius, kernel_g );
	
	// free self allocated memory
	if( selfallocmask )
	{
		printf("\ndelete[] mask");
		delete [] mask;
	}

	printf("\ncudaFree kernel");
	cutilSafeCall( cudaFree( kernel_g ) ); // gm
	printf("\nkernel deleting done");

// TODO
//	printf("\nfree temp");
//	if( selfalloctemp )
//		cutilSafeCall( cudaFree( temp_g ) );
//	printf("\ntemp free");

	printf("\ngauss done");
}

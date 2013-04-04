#include <auxiliary/cuda_basic.cuh>
#include <linearoperations/linearoperations.cuh>
#include <auxiliary/debug.hpp>
#include <string>

#define IP_BW 16
#define IP_BH 16

#define IMAGES_TO_INTERPOLATE 20








__global__ void foreward_warp_kernel_atomic_factor (
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
		float *out_g,
		int nx,
		int ny,
		int pitchf1,
		float factor
	)
{
	// get thread coordinates and index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x < nx && y < ny )
	{
		const unsigned int idx = y * pitchf1 + x;

		// calculate target coordinates: coords + flow values
		const float xx = (float)x + factor * flow1_g[idx];
		const float yy = (float)y + factor * flow2_g[idx];

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
		}
	}
}





// gpu warping kernel with global memory
__global__ void backwardRegistrationBilinearFunctionGlobalFactorGpu(const float *in_g,
		const float *flow1_g, const float *flow2_g, float *out_g,
		const float *constant_g, int nx, int ny, int pitchf1_in,
		int pitchf1_out, float hx, float hy, float factor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (x < nx && y < ny)
	{
		const float xx = (float) x + factor * flow1_g[y * pitchf1_in + x] / hx;
		const float yy = (float) y + factor * flow2_g[y * pitchf1_in + x] / hy;

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




void interpolateImages
	(
		float* image1_g,
		float* image2_g,
		float* u_g,
		float* v_g,
		int nx,
		int ny,
		int pitch
	)
{
	// block and grid size
	int gridsize_x = ((nx - 1) / IP_BW) + 1;
	int gridsize_y = ((ny - 1) / IP_BH) + 1;

	dim3 dimGrid( gridsize_x, gridsize_y );
	dim3 dimBlock( IP_BW, IP_BH );

	// allocate memory for file names
	char fileName[128];

	// create output directory
	// TODO

	// allocate GPU memory for result images
	float* result_g;
	cuda_malloc2D( (void**)&result_g, nx, ny, 1, sizeof(float), &pitch );

	// reset result image
	setKernel <<< dimGrid, dimBlock >>> ( result_g, nx, ny, pitch, 0.0f );



	float stepSize = 1 / (float)(IMAGES_TO_INTERPOLATE + 1);
	float factor = stepSize;

	fprintf( stderr, "\nFlow factor: %f, step size: %f", factor, stepSize );

	for( int i = 0; i < IMAGES_TO_INTERPOLATE; ++i )
	{
		// warp image
		//foreward_warp_kernel_atomic_factor <<< dimGrid, dimBlock >>> (
		//		u_g, v_g, image1_g, result_g, nx, ny, pitch, factor );
		backwardRegistrationBilinearFunctionGlobalFactorGpu<<<dimGrid, dimBlock>>>(
				image2_g, u_g, v_g, result_g, image1_g, nx, ny, pitch,
					pitch, 1.0f, 1.0f, factor );

		// save interpolated image
		fprintf( stderr, "\nSaving interpolated image %d (factor %f)...", i, factor );
		snprintf( fileName, 128, "interpolation/interpolation_%02d.png", i );
		saveCudaImage( fileName, result_g, nx, ny, pitch, 1 );

		factor += stepSize;

	}



	// clean up
	cutilSafeCall( cudaFree( result_g ) );
}

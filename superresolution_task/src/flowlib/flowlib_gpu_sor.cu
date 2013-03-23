/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
 *
 * time:    winter term 2012/13 / March 11-18, 2013
 *
 * project: superresolution
 * file:    flowlib_gpu_sor.cu
 *
 *
 * implement all functions with ### implement me ### in the function body
 \****************************************************************************/

/*
 * flowlib_gpu_sor.cu
 *
 *  Created on: Mar 14, 2012
 *      Author: steinbrf
 */

//#include <flowlib_gpu_sor.hpp>
#include "flowlib.hpp"
#include <auxiliary/cuda_basic.cuh>
#include <linearoperations/linearoperations.cuh>
#include <auxiliary/debug.hpp>

cudaChannelFormatDesc flow_sor_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I1;
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I2;
bool textures_flow_sor_initialized = false;

#define IMAGE_FILTER_METHOD cudaFilterModeLinear
#define SF_TEXTURE_OFFSET 0.5f
#define DEBUG  1

#define SF_BW 16
#define SF_BH 16

FlowLibGpuSOR::FlowLibGpuSOR(int par_nx, int par_ny) :
		FlowLib(par_nx, par_ny), FlowLibGpu(par_nx, par_ny), FlowLibSOR(par_nx,
				par_ny)
{

	cuda_malloc2D((void**) &_penDat, _nx, _ny, 1, sizeof(float), &_pitchf1);
	cuda_malloc2D((void**) &_penReg, _nx, _ny, 1, sizeof(float), &_pitchf1);

	cuda_malloc2D((void**) &_b1, _nx, _ny, 1, sizeof(float), &_pitchf1);
	cuda_malloc2D((void**) &_b2, _nx, _ny, 1, sizeof(float), &_pitchf1);

}

FlowLibGpuSOR::~FlowLibGpuSOR()
{
	if (_penDat)
		cutilSafeCall(cudaFree(_penDat));
	if (_penReg)
		cutilSafeCall(cudaFree(_penReg));
	if (_b1)
		cutilSafeCall(cudaFree(_b1));
	if (_b2)
		cutilSafeCall(cudaFree(_b2));
}

void bind_textures(const float *I1_g, const float *I2_g, int nx, int ny,
		int pitchf1)
{
	tex_flow_sor_I1.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I1.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I1.filterMode = IMAGE_FILTER_METHOD;
	tex_flow_sor_I1.normalized = false;

	tex_flow_sor_I2.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I2.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I2.filterMode = IMAGE_FILTER_METHOD;
	tex_flow_sor_I2.normalized = false;

	cutilSafeCall(
			cudaBindTexture2D(0, &tex_flow_sor_I1, I1_g, &flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)));
	cutilSafeCall(
			cudaBindTexture2D(0, &tex_flow_sor_I2, I2_g, &flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)));
}

void unbind_textures_flow_sor()
{
	cutilSafeCall(cudaUnbindTexture(tex_flow_sor_I1));
	cutilSafeCall(cudaUnbindTexture(tex_flow_sor_I2));
}

void update_textures_flow_sor(const float *I2_resampled_warped_g, int nx_fine,
		int ny_fine, int pitchf1)
{
	cutilSafeCall(cudaUnbindTexture(tex_flow_sor_I2));
	cutilSafeCall(
			cudaBindTexture2D(0, &tex_flow_sor_I2, I2_resampled_warped_g, &flow_sor_float_tex, nx_fine, ny_fine, pitchf1*sizeof(float)));
}

/**
 * @brief Adds one flow field onto another
 * @param du_g Horizontal increment
 * @param dv_g Vertical increment
 * @param u_g Horizontal accumulation
 * @param v_g Vertical accumulation
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 */__global__ void add_flow_fields(const float *du_g, const float *dv_g,
		float *u_g, float *v_g, int nx, int ny, int pitchf1)
{
	// ### Implement Me###
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < nx && y < ny)
	{
		int idx = y * pitchf1 + x;
		u_g[idx] += du_g[idx];
		v_g[idx] += dv_g[idx];
	}
}

/**
 * @brief Kernel to compute the penalty values for several
 * lagged-diffusivity iterations taking into account pixel sizes for warping.
 * Image derivatives are read from texture, flow derivatives from shared memory
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param du_g Pointer to global device memory for the horizontal
 * flow component of the increment flow field
 * @param dv_g Pointer to global device memory for the vertical
 * flow component of the increment flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 * @param pitchf1 Image pitch for single float images
 */__global__ void sorflow_update_robustifications_warp_tex_shared(
		const float *u_g, const float *v_g, const float *du_g,
		const float *dv_g, float *penaltyd_g, float *penaltyr_g, int nx, int ny,
		float hx, float hy, float data_epsilon, float diff_epsilon, int pitchf1)
{
	// ### Implement Me###
}

/**
 * @brief Precomputes one value as the sum of all values not depending of the
 * current flow increment
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param bu_g Pointer to global memory for horizontal result value
 * @param bv_g Pointer to global memory for vertical result value
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param pitchf1 Image pitch for single float images
 */__global__ void sorflow_update_righthandside_shared(const float *u_g,
		const float *v_g, const float *penaltyd_g, const float *penaltyr_g,
		float *bu_g, float *bv_g, int nx, int ny, float hx, float hy,
		float lambda, int pitchf1)
{
	// ### Implement Me###
}

/**
 * @brief Kernel to compute one Red-Black-SOR iteration for the nonlinear
 * Euler-Lagrange equation taking into account penalty values and pixel
 * size for warping
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param relaxation Overrelaxation for the SOR-solver
 * @param red Parameter deciding whether the red or black fields of a
 * checkerboard pattern are being updated
 * @param pitchf1 Image pitch for single float images
 */__global__ void sorflow_nonlinear_warp_sor_shared(const float *bu_g,
		const float *bv_g, const float *penaltyd_g, const float *penaltyr_g,
		float *du_g, float *dv_g, int nx, int ny, float hx, float hy,
		float lambda, float relaxation, int red, int pitchf1)
{
	// ### Implement Me ###
}

/**
 * @brief Method that calls the sorflow_nonlinear_warp_sor_shared in a loop,
 * with an outer loop for computing the diffisivity values for
 * one level of a coarse-to-fine implementation.
 * @param u_g Pointer to global device memory for the horizontal
 * flow component
 * @param v_g Pointer to global device memory for the vertical
 * flow component
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param outer_iterations Number of iterations of the penalty computation
 * @param inner_iterations Number of iterations for the SOR-solver
 * @param relaxation Overrelaxation for the SOR-solver
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 */
void sorflow_gpu_nonlinear_warp_level(const float *u_g, const float *v_g,
		float *du_g, float *dv_g, float *bu_g, float *bv_g, float *penaltyd_g,
		float *penaltyr_g, int nx, int ny, int pitchf1, float hx, float hy,
		float lambda, float overrelaxation, int outer_iterations,
		int inner_iterations, float data_epsilon, float diff_epsilon)
{
	bool red = 0;

	// grid and block dimensions
	int ngx = (nx % SF_BW) ? ((nx / SF_BW) + 1) : (nx / SF_BW);
	int ngy = (ny % SF_BH) ? ((ny / SF_BH) + 1) : (ny / SF_BH);
	dim3 dimGrid(ngx, ngy);
	dim3 dimBlock(SF_BW, SF_BH);

	for (int i = 0; i < outer_iterations; i++)
	{

		//Update Robustifications
		sorflow_update_robustifications_warp_tex_shared<<<dimGrid, dimBlock>>>(
				u_g, v_g, du_g, dv_g, penaltyd_g, penaltyr_g, nx, ny, hx, hy,
				data_epsilon, diff_epsilon, pitchf1);
		//Update Righthand Side
		sorflow_update_righthandside_shared<<<dimGrid, dimBlock>>>(u_g, v_g,
				penaltyd_g, penaltyr_g, bu_g, bv_g, nx, ny, hx, hy, lambda,
				pitchf1);

		//
		for (int j = 0; j < inner_iterations; j++)
		{
			red = 0;
			sorflow_nonlinear_warp_sor_shared<<<dimGrid, dimBlock>>>(bu_g, bv_g,
					penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, lambda,
					overrelaxation, red, pitchf1);

			red = 1;
			sorflow_nonlinear_warp_sor_shared<<<dimGrid, dimBlock>>>(bu_g, bv_g,
					penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, lambda,
					overrelaxation, red, pitchf1);

		}
	}
}

/*
 * Initializes an float array to zero
 */__global__ void initializeToZero(float* array, int width, int height,
		int pitch, bool black)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		if (black)
			array[x + y * pitch] = 0.0f;
		else
			array[x + y * pitch] = 255.0f;
	}
}

/*
 * Initializes two similar float arrays to zero
 */__global__ void initializeTwoToZero(float* array1, float* array2, int width,
		int height, int pitch)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		array1[x + y * pitch] = 255.0f;
		array2[x + y * pitch] = 128.0f;
	}
}

float FlowLibGpuSOR::computeFlow()
{
	bool verbose = 1;
	// main algorithm goes here
	if (verbose)
		fprintf(stderr, "\computeFlowGPU\n");

	float lambda = _lambda * 255.0f;

	//all per run variables goes here
	int max_rec_depth;/*maximal depth*/
	int warp_max_levels;/*maximal warp levels*/
	int rec_depth;/*actual depth*/
	float hx_fine, hy_fine;
	unsigned int nx_fine, ny_fine, nx_coarse = 0, ny_coarse = 0;

	//get max warp levels
	warp_max_levels = computeMaxWarpLevels();
	//get max rec depth
	max_rec_depth = (
			((_start_level + 1) < warp_max_levels) ?
					(_start_level + 1) : warp_max_levels) - 1;

	//get max rec depth
	if (max_rec_depth >= _I1pyramid->nl)
		max_rec_depth = _I1pyramid->nl - 1;

#ifdef DEBUG
	printf("max_rec_depth=%d, warp_max_levels=%d\n", max_rec_depth,
			warp_max_levels);
#endif

	// initial grid and block dimensions
	int initial_ngx = (_nx % SF_BW) ? ((_nx / SF_BW) + 1) : (_nx / SF_BW);
	int initial_ngy = (_ny % SF_BH) ? ((_ny / SF_BH) + 1) : (_ny / SF_BH);
	dim3 initial_dimGrid(initial_ngx, initial_ngy);
	dim3 initial_dimBlock(SF_BW, SF_BH);

	if (verbose)
		fprintf(stderr, "\nInitializing _u1_g & _u2_g to black");

	//set initial vector components to zero
	for (unsigned int p = 0; p < _nx * _ny; p++)
		_u1[p] = _u2[p] = 0.0f;

	// initialize horizontal and vertical components of the flow
	if (verbose)
		fprintf(stderr, "\nInitializing _u1_g & _u2_g to black");
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u1_g, _nx, _ny,
			_pitchf1, true);
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u2_g, _nx, _ny,
			_pitchf1, true);

#ifdef DEBUG
	char* cudaDebug = "1_debug/cu1lvl.png";
	showCudaImage(cudaDebug, _u2_g, _nx, _ny, _pitchf1, 1);
#endif

	if (verbose)
		fprintf(stderr, "\nInitializing coarse portion _u1_g & _u2_g to white");
	// hardcoding parameters for nx & ny as dimension of lowest resolution
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u1_g, 9, 16,
			_pitchf1, false);
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u2_g, 9, 16,
			_pitchf1, false);

#ifdef DEBUG
	cudaDebug = "2_debug/cu1lvl.png";
	showCudaImage(cudaDebug, _u2_g, _nx, _ny, _pitchf1, 1);
#endif

	//////////////////////////////////////////////////////////////
	// loop through image pyramide - main algorithm starts here //
	//////////////////////////////////////////////////////////////
	for (rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--)
	{
		//all per interation variables goes here
		nx_fine = _I1pyramid->nx[rec_depth];
		ny_fine = _I1pyramid->ny[rec_depth];

		hx_fine = (float) _nx / (float) nx_fine;
		hy_fine = (float) _ny / (float) ny_fine;

#ifdef DEBUG
		printf(
				"level=%d, (rec_depth=%d) ===> nx_fine=%d, ny_fine=%d, nx_coarse=%d, ny_coarse=%d, hx_fine=%f, hy_fine=%f\n",
				max_rec_depth - rec_depth, rec_depth, nx_fine, ny_fine,
				nx_coarse, ny_coarse, hx_fine, hy_fine);
#endif

		// current grid and block dimensions
		int ngx =
				(nx_fine % SF_BW) ? ((nx_fine / SF_BW) + 1) : (nx_fine / SF_BW);
		int ngy =
				(ny_fine % SF_BH) ? ((ny_fine / SF_BH) + 1) : (ny_fine / SF_BH);
		dim3 dimGrid(ngx, ngy);
		dim3 dimBlock(SF_BW, SF_BH);

		// resize flowfield to current level
		if (rec_depth < max_rec_depth)
		{
			resampleAreaParallelSeparate(_u1_g, _u1_g, nx_coarse, ny_coarse,
					_I2pyramid->pitch[rec_depth + 1], nx_fine, ny_fine,
					_I2pyramid->pitch[rec_depth], _b1);
			resampleAreaParallelSeparate(_u2_g, _u2_g, nx_coarse, ny_coarse,
					_I2pyramid->pitch[rec_depth + 1], nx_fine, ny_fine,
					_I2pyramid->pitch[rec_depth], _b2);
		}

		//bind resampled images
		//TODO understand texture binding
		int current_pitch = _I1pyramid->pitch[rec_depth];
		bind_textures(_I1pyramid->level[rec_depth],
				_I2pyramid->level[rec_depth], nx_fine, ny_fine, current_pitch);

		if (rec_depth >= _end_level)
		{
			// warp original image by resized flow field
			backwardRegistrationBilinearFunctionGlobal(
					_I2pyramid->level[rec_depth], _u1_g, _u2_g, _I2warp,
					_I1pyramid->level[rec_depth], nx_fine, ny_fine,
					_I2pyramid->pitch[rec_depth], current_pitch, hx_fine,
					hy_fine);

			// synchronize threads
			cutilSafeCall(cudaThreadSynchronize());

			//set du/dv to zero
			initializeToZero<<<dimGrid, dimBlock>>>(_u1lvl, nx_fine, ny_fine,
					current_pitch, true);
			initializeToZero<<<dimGrid, dimBlock>>>(_u2lvl, nx_fine, ny_fine,
					current_pitch, true);

			// compute incremental update for this level // A*x = b
			sorflow_gpu_nonlinear_warp_level(_u1_g, _u2_g, _u1lvl, _u2lvl, _b1,
					_b2, _penDat, _penReg, nx_fine, ny_fine, _pitchf1, hx_fine,
					hy_fine, lambda, _overrelaxation, _oi, _ii, _dat_epsilon,
					_reg_epsilon);

			// add the flow fields
			add_flow_fields<<<dimGrid, dimBlock>>>(_u1lvl, _u2lvl, _u1_g, _u2_g,
					nx_fine, ny_fine, _pitchf1);
		}

		nx_coarse = nx_fine;
		ny_coarse = ny_fine;

#ifdef DEBUG
		printf("End of interation\n");
#endif

	}

	unbind_textures_flow_sor();
	return -1.0f;

}


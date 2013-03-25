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
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//scale
	const float hx_1 = 1.0f / (2.0f * hx);
	const float hy_1 = 1.0f / (2.0f * hy);

	int idx = y * pitchf1 + x;

	//allocate shared mem blocks for flow field
	__shared__ float shared_u1[SF_BW + 2][SF_BH + 2];
	__shared__ float shared_u2[SF_BW + 2][SF_BH + 2];
	__shared__ float shared_du1[SF_BW + 2][SF_BH + 2];
	__shared__ float shared_du2[SF_BW + 2][SF_BH + 2];

	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;

	if (x < nx && y < ny)
	{
		// setup shared memory
		shared_u1[tx][ty] = u_g[idx];
		shared_u2[tx][ty] = v_g[idx];
		shared_du1[tx][ty] = du_g[idx];
		shared_du2[tx][ty] = dv_g[idx];

		//left border
		if (x == 0)
		{
			// left side of the image
			shared_u1[0][ty] = shared_u1[tx][ty];
			shared_u2[0][ty] = shared_u2[tx][ty];
			shared_du1[0][ty] = shared_du1[tx][ty];
			shared_du2[0][ty] = shared_du2[tx][ty];
		}
		else if (threadIdx.x == 0)
		{
			// left side of the block
			shared_u1[0][ty] = u_g[idx - 1];
			shared_u2[0][ty] = v_g[idx - 1];
			shared_du1[0][ty] = du_g[idx - 1];
			shared_du2[0][ty] = dv_g[idx - 1];
		}

		//right border
		if (x == nx - 1)
		{
			// right side of the image
			shared_u1[tx + 1][ty] = shared_u1[tx][ty];
			shared_u2[tx + 1][ty] = shared_u2[tx][ty];
			shared_du1[tx + 1][ty] = shared_du1[tx][ty];
			shared_du2[tx + 1][ty] = shared_du2[tx][ty];
		}
		else if (threadIdx.x == SF_BW - 1)
		{
			// right side of the block
			shared_u1[tx + 1][ty] = u_g[idx + 1];
			shared_u2[tx + 1][ty] = v_g[idx + 1];
			shared_du1[tx + 1][ty] = du_g[idx + 1];
			shared_du2[tx + 1][ty] = dv_g[idx + 1];
		}

		//top border
		if (y == 0)
		{
			// top side of the image
			shared_u1[tx][0] = shared_u1[tx][ty];
			shared_u2[tx][0] = shared_u2[tx][ty];
			shared_du1[tx][0] = shared_du1[tx][ty];
			shared_du2[tx][0] = shared_du2[tx][ty];
		}
		else if (threadIdx.y == 0)
		{
			// top side of the block
			shared_u1[tx][0] = u_g[idx - pitchf1];
			shared_u2[tx][0] = v_g[idx - pitchf1];
			shared_du1[tx][0] = du_g[idx - pitchf1];
			shared_du2[tx][0] = dv_g[idx - pitchf1];
		}

		//bikini bottom border
		if (y == ny - 1)
		{
			// bottom side of the image
			shared_u1[tx][ty + 1] = shared_u1[tx][ty];
			shared_u2[tx][ty + 1] = shared_u2[tx][ty];
			shared_du1[tx][ty + 1] = shared_du1[tx][ty];
			shared_du2[tx][ty + 1] = shared_du2[tx][ty];
		}
		else if (threadIdx.y == SF_BH - 1)
		{
			// bottom side of the block
			shared_u1[tx][ty + 1] = u_g[idx + pitchf1];
			shared_u2[tx][ty + 1] = v_g[idx + pitchf1];
			shared_du1[tx][ty + 1] = du_g[idx + pitchf1];
			shared_du2[tx][ty + 1] = dv_g[idx + pitchf1];
		}
	}

	__syncthreads();

	if (x < nx && y < ny)
	{
		// local sm indices
		unsigned int sm_xminus1 = x == 0 ? tx : tx - 1;
		unsigned int sm_xplus1 = x == nx - 1 ? tx : tx + 1;
		unsigned int sm_yminus1 = y == 0 ? ty : ty - 1;
		unsigned int sm_yplus1 = y == ny - 1 ? ty : ty + 1;

		//global texture indices
		const float tx_x   = (float) x                            + SF_TEXTURE_OFFSET;
		const float tx_x1  = (float) ((x == nx - 1) ? x : x + 1)  + SF_TEXTURE_OFFSET;
		const float tx_x_1 = (float) ((x == 0) ? x : x - 1)       + SF_TEXTURE_OFFSET;
		const float tx_y   = (float) y                            + SF_TEXTURE_OFFSET;
		const float tx_y1  = (float) ((y == ny - 1) ? y : y + 1)  + SF_TEXTURE_OFFSET;
		const float tx_y_1 = (float) ((y == 0) ? y : y - 1)       + SF_TEXTURE_OFFSET;

		//calculate Ix, Iy, It
		float Ix = 0.5f
				* (tex2D(tex_flow_sor_I2, tx_x1, tx_y)
						- tex2D(tex_flow_sor_I2, tx_x_1, tx_y)
						+ tex2D(tex_flow_sor_I1, tx_x1, tx_y)
						- tex2D(tex_flow_sor_I1, tx_x_1, tx_y)) * hx_1;

		float Iy = 0.5f
				* (tex2D(tex_flow_sor_I2, tx_x, tx_y1)
						- tex2D(tex_flow_sor_I2, tx_x, tx_y_1)
						+ tex2D(tex_flow_sor_I1, tx_x, tx_y1)
						- tex2D(tex_flow_sor_I1, tx_x, tx_y_1)) * hy_1;

		float It = tex2D(tex_flow_sor_I2, tx_x, tx_y)
				- tex2D(tex_flow_sor_I1, tx_x, tx_y);

		double dxu = (shared_u1[sm_xplus1][ty] - shared_u1[sm_xminus1][ty]
				+ shared_du1[sm_xplus1][ty] - shared_du1[sm_xminus1][ty])
				* hx_1;
		double dyu = (shared_u1[tx][sm_yplus1] - shared_u1[tx][sm_yminus1]
				+ shared_du1[tx][sm_yplus1] - shared_du1[tx][sm_yminus1])
				* hy_1;
		double dxv = (shared_u2[sm_xplus1][ty] - shared_u2[sm_xminus1][ty]
				+ shared_du2[sm_xplus1][ty] - shared_du2[sm_xminus1][ty])
				* hx_1;
		double dyv = (shared_u2[tx][sm_yplus1] - shared_u2[tx][sm_yminus1]
				+ shared_du2[tx][sm_yplus1] - shared_du2[tx][sm_yminus1])
				* hy_1;

		double dataterm = shared_du1[tx][ty] * Ix + shared_du2[tx][ty] * Iy
				+ It;

		//calculate penalty terms
		penaltyd_g[idx] = 1.0f/sqrt(dataterm * dataterm + data_epsilon);
		penaltyr_g[idx] = 1.0f/sqrt(
				dxu * dxu + dxv * dxv + dyu * dyu + dyv * dyv + diff_epsilon);
	}
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
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const float hx_1 = 1.0f / (2.0f * hx);
	const float hy_1 = 1.0f / (2.0f * hy);
	const float hx_2 = lambda / (hx * hx);
	const float hy_2 = lambda / (hy * hy);

	__shared__ float shared_u1[SF_BW+2][SF_BH+2];
	__shared__ float shared_u2[SF_BW+2][SF_BH+2];
	__shared__ float shared_penaltyd[SF_BW+2][SF_BH+2];
	__shared__ float shared_penaltyr[SF_BW+2][SF_BH+2];

	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;
	const int idx = y * pitchf1 + x;

	//first setup shared memory
	if (x < nx && y < ny)
	{
		// setup shared memory
		shared_u1[tx][ty]       = u_g[idx];
		shared_u2[tx][ty]       = v_g[idx];
		shared_penaltyd[tx][ty] = penaltyd_g[idx];
		shared_penaltyr[tx][ty] = penaltyr_g[idx];

		//left border
		if (x == 0)
		{
			// left side of the image
			shared_u1[0][ty]       = shared_u1[tx][ty];
			shared_u2[0][ty]       = shared_u2[tx][ty];
			shared_penaltyd[0][ty] = shared_penaltyd[tx][ty];
			shared_penaltyr[0][ty] = shared_penaltyr[tx][ty];
		}
		else if (threadIdx.x == 0)
		{
			// left side of the block
			shared_u1[0][ty]       = u_g[idx - 1];
			shared_u2[0][ty]       = v_g[idx - 1];
			shared_penaltyd[0][ty] = penaltyd_g[idx - 1];
			shared_penaltyr[0][ty] = penaltyr_g[idx - 1];
		}

		//right border
		if (x == nx - 1)
		{
			// right side of the image
			shared_u1[tx + 1][ty]       = shared_u1[tx][ty];
			shared_u2[tx + 1][ty]       = shared_u2[tx][ty];
			shared_penaltyd[tx + 1][ty] = shared_penaltyd[tx][ty];
			shared_penaltyr[tx + 1][ty] = shared_penaltyr[tx][ty];
		}
		else if (threadIdx.x == SF_BW - 1)
		{
			// right side of the block
			shared_u1[tx + 1][ty]       = u_g[idx + 1];
			shared_u2[tx + 1][ty]       = v_g[idx + 1];
			shared_penaltyd[tx + 1][ty] = penaltyd_g[idx + 1];
			shared_penaltyd[tx + 1][ty] = penaltyr_g[idx + 1];
		}

		//top border
		if (y == 0)
		{
			// top side of the image
			shared_u1[tx][0]       = shared_u1[tx][ty];
			shared_u2[tx][0]       = shared_u2[tx][ty];
			shared_penaltyd[tx][0] = shared_penaltyd[tx][ty];
			shared_penaltyd[tx][0] = shared_penaltyd[tx][ty];
		}
		else if (threadIdx.y == 0)
		{
			// top side of the block
			shared_u1[tx][0]       = u_g[idx - pitchf1];
			shared_u2[tx][0]       = v_g[idx - pitchf1];
			shared_penaltyd[tx][0] = penaltyd_g[idx - pitchf1];
			shared_penaltyd[tx][0] = penaltyr_g[idx - pitchf1];
		}

		//bikini bottom border
		if (y == ny - 1)
		{
			// bottom side of the image
			shared_u1[tx][ty + 1]       = shared_u1[tx][ty];
			shared_u2[tx][ty + 1]       = shared_u2[tx][ty];
			shared_penaltyd[tx][ty + 1] = shared_penaltyd[tx][ty];
			shared_penaltyd[tx][ty + 1] = shared_penaltyd[tx][ty];
		}
		else if (threadIdx.y == SF_BH - 1)
		{
			// bottom side of the block
			shared_u1[tx][ty + 1]       = u_g[idx + pitchf1];
			shared_u2[tx][ty + 1]       = v_g[idx + pitchf1];
			shared_penaltyd[tx][ty + 1] = penaltyd_g[idx + pitchf1];
			shared_penaltyd[tx][ty + 1] = penaltyr_g[idx + pitchf1];
		}
	}
	__syncthreads();

	//then calculate righthand site
	if (x < nx && y < ny)
	{
		// local sm indices
		unsigned int sm_xminus1 = x == 0 ? tx : tx - 1;
		unsigned int sm_xplus1  = x == nx - 1 ? tx : tx + 1;
		unsigned int sm_yminus1 = y == 0 ? ty : ty - 1;
		unsigned int sm_yplus1  = y == ny - 1 ? ty : ty + 1;

		//global indices
		const float tx_x   = (float) x                            + SF_TEXTURE_OFFSET;
		const float tx_x1  = (float) ((x == nx - 1) ? x : x + 1)  + SF_TEXTURE_OFFSET;
		const float tx_x_1 = (float) ((x == 0) ? x : x - 1)       + SF_TEXTURE_OFFSET;
		const float tx_y   = (float) y                            + SF_TEXTURE_OFFSET;
		const float tx_y1  = (float) ((y == ny - 1) ? y : y + 1)  + SF_TEXTURE_OFFSET;
		const float tx_y_1 = (float) ((y == 0) ? y : y - 1)       + SF_TEXTURE_OFFSET;

		//calculate Ix, Iy, It
		float Ix = 0.5f
				* (tex2D(tex_flow_sor_I2, tx_x1, tx_y)
						- tex2D(tex_flow_sor_I2, tx_x_1, tx_y)
						+ tex2D(tex_flow_sor_I1, tx_x1, tx_y)
						- tex2D(tex_flow_sor_I1, tx_x_1, tx_y)) * hx_1;

		float Iy = 0.5f
				* (tex2D(tex_flow_sor_I2, tx_x, tx_y1)
						- tex2D(tex_flow_sor_I2, tx_x, tx_y_1)
						+ tex2D(tex_flow_sor_I1, tx_x, tx_y1)
						- tex2D(tex_flow_sor_I1, tx_x, tx_y_1)) * hy_1;

		float It = tex2D(tex_flow_sor_I2, tx_x, tx_y)
				  - tex2D(tex_flow_sor_I1, tx_x, tx_y);

		float xp = x<nx-1 ? (shared_penaltyr[sm_xplus1][ty]  + shared_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
		float xm = x>0    ? (shared_penaltyr[sm_xminus1][ty] + shared_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
		float yp = y<ny-1 ? (shared_penaltyr[tx][sm_yplus1]  + shared_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
		float ym = y>0    ? (shared_penaltyr[tx][sm_yminus1] + shared_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
		//sum up elements
		float sum = xp + xm + yp + ym;

		bu_g[idx] = -shared_penaltyd[tx][ty] * Ix*It
							+ (x>0    ? xm*shared_u1[sm_xminus1][ty] : 0.0f)
							+ (x<nx-1 ? xp*shared_u1[sm_xplus1][ty]  : 0.0f)
							+ (y>0    ? ym*shared_u1[tx][sm_yminus1] : 0.0f)
							+ (y<ny-1 ? yp*shared_u1[tx][sm_yplus1]  : 0.0f)
							- sum * shared_u1[tx][ty];

		bv_g[idx] = -shared_penaltyd[tx][ty] * Iy*It
							+ (x>0    ? xm*shared_u2[sm_xminus1][ty] : 0.0f)
							+ (x<nx-1 ? xp*shared_u2[sm_xplus1][ty]  : 0.0f)
							+ (y>0    ? ym*shared_u2[tx][sm_yminus1] : 0.0f)
							+ (y<ny-1 ? yp*shared_u2[tx][sm_yplus1]  : 0.0f)
							- sum * shared_u2[tx][ty];

	}
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
 */__global__ void sorflow_nonlinear_warp_sor_shared (
		 int i,
		const float *bu_g,			// right-hand side of lin. equations of horizontal flow
		const float *bv_g,			// right-hand side of lin. equations of vertical flow
		const float *penaltyd_g,	// data term penalty
		const float *penaltyr_g,	// regularity term
		float *du_g,				// horizontal flow increment
		float *dv_g,				// vertical flow increment
		int nx,						// image width	(= nx_fine)
		int ny,						// image height (= ny_fine)
		float hx,					// scale factor / pixel size in X direction
		float hy,					// scale factor / pixel size in Y direction
		const float lambda,
		float relaxation,			// overrelaxation value
		int red,					// checkerboard flag
		int pitchf1
	)
{
	// get thread coordinates and index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idx = y * pitchf1 + x;

	// get shared memory coordinates with border offset
	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;

	//allocate shared memory blocks for regularity penalty and flow increment
	__shared__ float shared_regPen[SF_BW + 2][SF_BH + 2];
	__shared__ float shared_du1[SF_BW + 2][SF_BH + 2];
	__shared__ float shared_du2[SF_BW + 2][SF_BH + 2];


	if (x < nx && y < ny)
	{

		// load data into shared memory
		shared_regPen[tx][ty] = penaltyr_g[idx];
		shared_du1[tx][ty]    = du_g[idx];
		shared_du2[tx][ty]    = dv_g[idx];

		//left border
		if (x == 0)
		{
			// left side of the image
			shared_regPen[0][ty] = shared_regPen[tx][ty];
			shared_du1[0][ty]    = shared_du1[tx][ty];
			shared_du2[0][ty]    = shared_du2[tx][ty];
		}
		else if (threadIdx.x == 0)
		{
			// left side of the block
			shared_regPen[0][ty] = penaltyr_g[idx - 1];
			shared_du1[0][ty]    = du_g[idx - 1];
			shared_du2[0][ty]    = dv_g[idx - 1];
		}

		//right border
		if (x == nx - 1)
		{
			// right side of the image
			shared_regPen[tx + 1][ty] = shared_regPen[tx][ty];
			shared_du1[tx + 1][ty]    = shared_du1[tx][ty];
			shared_du2[tx + 1][ty]    = shared_du2[tx][ty];
		}
		else if (threadIdx.x == SF_BW - 1)
		{
			// right side of the block
			shared_regPen[tx + 1][ty] = penaltyr_g[idx + 1];
			shared_du1[tx + 1][ty]    = du_g[idx + 1];
			shared_du2[tx + 1][ty]    = dv_g[idx + 1];
		}

		//top border
		if (y == 0)
		{
			// top side of the image
			shared_regPen[tx][0] = shared_regPen[tx][ty];
			shared_du1[tx][0]    = shared_du1[tx][ty];
			shared_du2[tx][0]    = shared_du2[tx][ty];
		}
		else if (threadIdx.y == 0)
		{
			// top side of the block
			shared_regPen[tx][0] = penaltyr_g[idx - pitchf1];
			shared_du1[tx][0]    = du_g[idx - pitchf1];
			shared_du2[tx][0]    = dv_g[idx - pitchf1];
		}

		//bikini bottom border
		if (y == ny - 1)
		{
			// bottom side of the image
			shared_regPen[tx][ty + 1] = shared_regPen[tx][ty];
			shared_du1[tx][ty + 1]    = shared_du1[tx][ty];
			shared_du2[tx][ty + 1]    = shared_du2[tx][ty];
		}
		else if (threadIdx.y == SF_BH - 1)
		{
			// bottom side of the block
			shared_regPen[tx][ty + 1] = penaltyr_g[idx + pitchf1];
			shared_du1[tx][ty + 1]    = du_g[idx + pitchf1];
			shared_du2[tx][ty + 1]    = dv_g[idx + pitchf1];
		}
	}

	__syncthreads();

	//if( x < nx && y < ny && ((x + y) & 1) == red ) // = ( (x + y) % 2 ) == red
	if( x < nx && y < ny && ( (x + y) % 2 ) == red )
	{
		//scale
		const float hx_1 = 1.0f / (2.0f * hx);
		const float hy_1 = 1.0f / (2.0f * hy);
		const float hx_2 = lambda / (hx * hx);
		const float hy_2 = lambda / (hy * hy);

		//printf("\nsor Cuda sor thread inter= %i, x: %i y: %i, hx=%f,  hy=%f, hx_2=%f,  hy_2=%f,  lamda=%f",i, x,y,hx, hy,hx_2, hy_2, lambda);
		//printf("\nsor Cuda sor thread lamda=%f", lambda);

		// precalculate coordinates of surrounding pixels
		unsigned int x_1 = x == 0      ? tx : tx - 1;
		unsigned int x1  = x == nx - 1 ? tx : tx + 1;
		unsigned int y_1 = y == 0      ? ty : ty - 1;
		unsigned int y1  = y == ny - 1 ? ty : ty + 1;

		//global texture indices
		const float xx = (float) x + SF_TEXTURE_OFFSET;
		const float yy = (float) y + SF_TEXTURE_OFFSET;

		float lala = tex2D(tex_flow_sor_I2, xx + 1, yy);

		//calculate Ix, Iy
		float Ix = 0.5f	 * (tex2D(tex_flow_sor_I2, xx + 1, yy)
						  - tex2D(tex_flow_sor_I2, xx - 1, yy)
						  + tex2D(tex_flow_sor_I1, xx + 1, yy)
						  - tex2D(tex_flow_sor_I1, xx - 1, yy)) * hx_1;


		float Iy = 0.5f	 * (tex2D(tex_flow_sor_I2, xx, yy + 1)
						  - tex2D(tex_flow_sor_I2, xx, yy - 1)
						  + tex2D(tex_flow_sor_I1, xx, yy + 1)
						  - tex2D(tex_flow_sor_I1, xx, yy - 1)) * hy_1;

		/*		//global texture indices
		const float tx_x   = (float) x                           + SF_TEXTURE_OFFSET;
		const float tx_x1  = (float) ((x == nx - 1) ? x : x + 1) + SF_TEXTURE_OFFSET;
		const float tx_x_1 = (float) ((x == 0)      ? x : x - 1) + SF_TEXTURE_OFFSET;
		const float tx_y   = (float) y                           + SF_TEXTURE_OFFSET;
		const float tx_y1  = (float) ((y == ny - 1) ? y : y + 1) + SF_TEXTURE_OFFSET;
		const float tx_y_1 = (float) ((y == 0)      ? y : y - 1) + SF_TEXTURE_OFFSET;

		//calculate Ix, Iy
		float Ix = 0.5f	* (tex2D(tex_flow_sor_I2, tx_x1, tx_y)
						 - tex2D(tex_flow_sor_I2, tx_x_1, tx_y)
						 + tex2D(tex_flow_sor_I1, tx_x1, tx_y)
						 - tex2D(tex_flow_sor_I1, tx_x_1, tx_y)) * hx_1;


		float Iy = 0.5f * (tex2D(tex_flow_sor_I2, tx_x, tx_y1)
						 - tex2D(tex_flow_sor_I2, tx_x, tx_y_1)
						 + tex2D(tex_flow_sor_I1, tx_x, tx_y1)
						 - tex2D(tex_flow_sor_I1, tx_x, tx_y_1)) * hy_1;*/




		float xp = x < nx - 1 ? ( shared_regPen[x1][ty]  + shared_regPen[tx][ty] ) * 0.5f * hx_2 : 0.0f;
		float xm = x > 0      ? ( shared_regPen[x_1][ty] + shared_regPen[tx][ty] ) * 0.5f * hx_2 : 0.0f;
		float yp = y < ny - 1 ? ( shared_regPen[tx][y1]  + shared_regPen[tx][ty] ) * 0.5f * hy_2 : 0.0f;
		float ym = y > 0      ? ( shared_regPen[tx][y_1] + shared_regPen[tx][ty] ) * 0.5f * hy_2 : 0.0f;
		float sum = xp + xm + yp + ym;

		//printf("\nsor inter=%i ,x: %i y: %i, xp=%f,  xm=%f,  yp=%f, ym=%f, ym=%f, hx_2=%f, hx=%f, lamda=%f", i,x,y,xp, xm, yp,ym,hx_2,hx,lambda);
		//printf("\nsor shared x: %i y: %i shared:%f",x,y, shared_regPen[tx][y1]);
		//printf("\nsor sor x: %i y: %i sum:%f",x,y, sum);

		float dataPenalty = penaltyd_g[idx];

		float u1new  = (1.0f - relaxation) * shared_du1[tx][ty] + relaxation *
				( bu_g[idx] - dataPenalty * Ix * Iy * shared_du2[tx][ty]
				+ (x > 0      ? xm * shared_du1[x_1][ty] : 0.0f)
				+ (x < nx - 1 ? xp * shared_du1[x1][ty]  : 0.0f)
				+ (y > 0      ? ym * shared_du1[tx][y_1] : 0.0f)
				+ (y < ny - 1 ? yp * shared_du1[tx][y1]  : 0.0f))
				/ (dataPenalty * Ix * Ix + sum);

		float u2new = (1.0f - relaxation) * shared_du2[tx][ty] + relaxation *
				( bv_g[idx] - dataPenalty * Ix * Iy * shared_du1[tx][ty]
				+ (x > 0      ? xm * shared_du2[x_1][ty] : 0.0f)
				+ (x < nx - 1 ? xp * shared_du2[x1][ty]  : 0.0f)
				+ (y > 0      ? ym * shared_du2[tx][y_1] : 0.0f)
				+ (y < ny - 1 ? yp * shared_du2[tx][y1]  : 0.0f))
				/ (dataPenalty * Iy * Iy + sum);

		// update flow increment
		du_g[idx] = u1new;
		dv_g[idx] = u2new;

		//printf("\nsor Cuda sor thread du_g[idx]=%f", du_g[idx]);
		//printf("\n\nsor Cuda sor thread x: %i y: %i, Ix=%f,  Iy=%f, du_g[idx]= %f,dv_g[idx]= %f,xx = %f,yy = %f", x,y,Ix, Iy, du_g[idx],dv_g[idx],xx,yy);
		//printf("\nsor Cuda sor thread x: %i y: %i, lala=%f,  hx=%f,  hx_1=%f, Ix=%f, du_g[idx]= %f,dv_g[idx]= %f", x,y,lala, hx, hx_1,Ix, du_g[idx],dv_g[idx]);
	}
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
void sorflow_gpu_nonlinear_warp_level(char* call,const float *u_g, const float *v_g,
		float *du_g, float *dv_g, float *bu_g, float *bv_g, float *penaltyd_g,
		float *penaltyr_g, int nx, int ny, int pitchf1, float hx, float hy,
		float lambda, float overrelaxation, int outer_iterations,
		int inner_iterations, float data_epsilon, float diff_epsilon)
{
// grid and block dimensions
	int ngx = (nx % SF_BW) ? ((nx / SF_BW) + 1) : (nx / SF_BW);
	int ngy = (ny % SF_BH) ? ((ny / SF_BH) + 1) : (ny / SF_BH);
	dim3 dimGrid(ngx, ngy);
	dim3 dimBlock(SF_BW, SF_BH);
	bool red = 0;

	//printKernel <<<dimGrid,dimBlock>>>(1,du_g, nx, ny, pitchf1, 0.0f);
	//printKernel <<<dimGrid,dimBlock>>>(1,dv_g, nx, ny, pitchf1, 0.0f);

	//for (int i = 0; i < 1; i++)
	for (int i = 0; i < 20; i++)
	{

		//robustifications
		sorflow_update_robustifications_warp_tex_shared<<<dimGrid, dimBlock>>>(
				u_g, v_g, du_g, dv_g, penaltyd_g, penaltyr_g, nx, ny, hx, hy,
				data_epsilon, diff_epsilon, pitchf1);
		catchkernel;

		//printKernel <<<dimGrid,dimBlock>>>(i,penaltyd_g, nx, ny, pitchf1, 0.0f);
			//printKernel <<<dimGrid,dimBlock>>>(1,dv_g, nx, ny, pitchf1, 0.0f);

/*
		 #ifdef DEBUG
		   char* cudaDebug = "1_debug/penaltyd_g.png";
		   showCudaImage(cudaDebug, penaltyd_g, nx, ny, pitchf1, 1);
		 #endif

		 #ifdef DEBUG
		   cudaDebug = "1_debug/penaltyr_g.png";
		   showCudaImage(cudaDebug, penaltyr_g, nx, ny, pitchf1, 1);
		 #endif
*/

		//righthand side
		sorflow_update_righthandside_shared<<<dimGrid, dimBlock>>>(u_g, v_g,
				penaltyd_g, penaltyr_g, bu_g, bv_g, nx, ny, hx, hy, lambda,
				pitchf1);
		catchkernel;

		/*
		#ifdef DEBUG
			char* cudaDebug = "1_debug/bu_g.png";
			showCudaImage(cudaDebug, bu_g, nx, ny, pitchf1, 1);
		#endif

		#ifdef DEBUG
			cudaDebug = "1_debug/bv_g.png";
			showCudaImage(cudaDebug, bv_g, nx, ny, pitchf1, 1);
		#endif
		*/

		//sor interation
		for (int j = 0; j < inner_iterations; j++)
		//for (int j = 0; j < 2; j++)
		{
			red = 0;
			sorflow_nonlinear_warp_sor_shared<<<dimGrid, dimBlock>>>(i*j,bu_g, bv_g,
					penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, lambda,
					overrelaxation, red, pitchf1);

			red = 1;
			sorflow_nonlinear_warp_sor_shared<<<dimGrid, dimBlock>>>(i*j,bu_g, bv_g,
					penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, lambda,
					overrelaxation, red, pitchf1);

			/*
			#ifdef DEBUG
				char* cudaDebug = "1_debug/du_g.png";
				showCudaImage(cudaDebug, du_g, nx, ny, pitchf1, 1);
			#endif

			#ifdef DEBUG
				cudaDebug = "1_debug/dv_g.png";
				showCudaImage(cudaDebug, dv_g, nx, ny, pitchf1, 1);
			#endif
			*/

		}

		//printKernel <<<dimGrid,dimBlock>>>(i,du_g, nx, ny, pitchf1, 0.0f);
	}

	//printKernel <<<dimGrid,dimBlock>>>(2,du_g, nx, ny, pitchf1, 0.0f);
	//printKernel <<<dimGrid,dimBlock>>>(2,dv_g, nx, ny, pitchf1, 0.0f);

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
	//cudaPrintfInit();

	bool verbose = true;
	// main algorithm goes here
	if (verbose)fprintf(stderr, "\computeFlowGPU");

	//the lambda
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


	if (verbose)fprintf(stderr,"\nmax_rec_depth=%d, warp_max_levels=%d",
			max_rec_depth,warp_max_levels);

	// initial grid and block dimensionslow  SOR
	int initial_ngx = (_nx % SF_BW) ? ((_nx / SF_BW) + 1) : (_nx / SF_BW);
	int initial_ngy = (_ny % SF_BH) ? ((_ny / SF_BH) + 1) : (_ny / SF_BH);
	dim3 initial_dimGrid(initial_ngx, initial_ngy);
	dim3 initial_dimBlock(SF_BW, SF_BH);

	// initialize horizontal and vertical components of the flow
	if (verbose)
		fprintf(stderr, "\nInitializing _u1_g & _u2_g to black");
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u1_g, _nx, _ny,
			_pitchf1, true);
	initializeToZero<<<initial_dimGrid, initial_dimBlock>>>(_u2_g, _nx, _ny,
			_pitchf1, true);
	if (verbose)
			fprintf(stderr, "\nInitialized _u1_g & _u2_g to black");

	if (verbose)
			fprintf(stderr, "\nRelaxation: %f", _overrelaxation);


	//////////////////////////////////////////////////////////////
	// loop through image pyramide - main algorithm starts here //
	//////////////////////////////////////////////////////////////
	for (rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--)
	{
       // rec_depth = 0;

		if (verbose)fprintf(stderr,
						"\n\nStart interation");

		//get image values for this interation
		nx_fine = _I1pyramid->nx[rec_depth];
		ny_fine = _I1pyramid->ny[rec_depth];

		hx_fine = (float) _nx / (float) nx_fine;
		hy_fine = (float) _ny / (float) ny_fine;

		if (verbose)fprintf(stderr,
				"\nlevel=%d, (rec_depth=%d) ===> nx_fine=%d, ny_fine=%d, nx_coarse=%d, ny_coarse=%d, hx_fine=%f, hy_fine=%f",
				max_rec_depth - rec_depth, rec_depth, nx_fine, ny_fine,
				nx_coarse, ny_coarse, hx_fine, hy_fine);


		 #ifdef DEBUG
		   char* cudaDebug = "1_debug/image1.png";
		   showCudaImage(cudaDebug, _I1pyramid->level[rec_depth], nx_fine, ny_fine, _I1pyramid->pitch[rec_depth], 1);
		 #endif

		 #ifdef DEBUG
		   cudaDebug = "2_debug/image2.png";
		   showCudaImage(cudaDebug, _I2pyramid->level[rec_depth], nx_fine, ny_fine, _I2pyramid->pitch[rec_depth], 1);
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

			if (verbose)
					fprintf(stderr, "\nResampling area starts");
			resampleAreaParallelSeparate(_u1_g, _u1_g, nx_coarse, ny_coarse,
					_I2pyramid->pitch[rec_depth + 1], nx_fine, ny_fine,
					_I2pyramid->pitch[rec_depth], _b1);
			resampleAreaParallelSeparate(_u2_g, _u2_g, nx_coarse, ny_coarse,
					_I2pyramid->pitch[rec_depth + 1], nx_fine, ny_fine,
					_I2pyramid->pitch[rec_depth], _b2);
			if (verbose)
					fprintf(stderr, "\nResampling area finish");

		}

		//bind resampled images
		int current_pitch = _I1pyramid->pitch[rec_depth];
		bind_textures(_I1pyramid->level[rec_depth],
				_I2pyramid->level[rec_depth], nx_fine, ny_fine, current_pitch);


		if (verbose)fprintf(stderr,"\nrec_depth=%i, _end_level=%i",
				rec_depth,_end_level);

		if (rec_depth >= _end_level)
		{
			if (verbose)
							fprintf(stderr, "\nWarping starts");
			// warp original image by resized flow field
			backwardRegistrationBilinearFunctionGlobal(
					_I2pyramid->level[rec_depth], _u1_g, _u2_g, _I2warp,
					_I1pyramid->level[rec_depth], nx_fine, ny_fine,
					current_pitch, current_pitch, hx_fine,
					hy_fine);
			catchkernel;
			if (verbose)
									fprintf(stderr, "\nWarping ends");

			// update I2 with warped image
			update_textures_flow_sor(_I2warp, nx_fine, ny_fine, current_pitch);

			//set du/dv to zero
			initializeToZero<<<dimGrid, dimBlock>>>(_u1lvl, nx_fine, ny_fine,
					current_pitch, true);
			initializeToZero<<<dimGrid, dimBlock>>>(_u2lvl, nx_fine, ny_fine,
					current_pitch, true);

			//set du/dv to zero
			setKernel <<<dimGrid,dimBlock>>>(_u1lvl, nx_fine, ny_fine, current_pitch, 0.0f);
			setKernel <<<dimGrid,dimBlock>>>(_u2lvl, nx_fine, ny_fine, current_pitch, 0.0f);

			// compute incremental update for this level // A*x = b
			sorflow_gpu_nonlinear_warp_level("1",_u1_g, _u2_g, _u1lvl, _u2lvl, _b1,
					_b2, _penDat, _penReg, nx_fine, ny_fine, current_pitch, hx_fine,
					hy_fine, lambda, _overrelaxation, _oi, _ii, _dat_epsilon,
					_reg_epsilon);

			// add the flow fields
			add_flow_fields<<<dimGrid, dimBlock>>>(_u1lvl, _u2lvl, _u1_g, _u2_g,
					nx_fine, ny_fine, current_pitch);
		}

		nx_coarse = nx_fine;
		ny_coarse = ny_fine;

		if (verbose)
					fprintf(stderr, "\nEnd of interation");

		unbind_textures_flow_sor();

	}

	return -1.0f;
}


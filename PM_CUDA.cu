
#include "PM_CUDA.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

// simulation parameters in constant memory
__constant__ SimParams params;

__device__ float floatMod(float a, float b)
{
	float c = a + b / 2;
	return (c - b * floor(c / b) - b / 2);
}

__device__ int o(int i, int j, int k)
{
	int M = params.meshSize;
	return i * M * M + j * M + k;  //calculate offset into mesh density array
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 get_gc(float *phi, int i, int j, int k, float s)
{
	int M = params.meshSize;
	float3 gc;
	// calculate accelerations from adjacent cell centres
	gc.x = (phi[o((i + 1) % M, j, k)] - phi[o((i - 1 + M) % M, j, k)]) / 2 * s;
	gc.y = (phi[o(i, (j + 1) % M, k)] - phi[o(i, (j - 1 + M) % M, k)]) / 2 * s;
	gc.z = (phi[o(i, j, (k + 1) % M)] - phi[o(i, j, (k - 1 + M) % M)]) / 2 * s;

	return gc;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void accumulateKernel(float3 *r, float *rho, int N)
{
	float dx, dy, dz;
	float tx, ty, tz;
	int i, j, k;  //mesh cell indices

	float m = params.particleMass; // particle mass
	int M = params.meshSize;
	float dr = params.meshCellSize;

	//For each particle, use cloud in cell methodology to apportion its mass between 8 neighbouring mesh points
	int p = blockIdx.x;

	if (p < N)
	{
		// calculate indices of cell containing particle at position r
		i = floor(r[p].x / dr) + M / 2;
		j = floor(r[p].y / dr) + M / 2;
		k = floor(r[p].z / dr) + M / 2;

		// calculate distances betweeen particle and cell centres
		dx = r[p].x - (i - M / 2) * dr;
		dy = r[p].y - (j - M / 2) * dr;
		dz = r[p].z - (k - M / 2) * dr;
		tx = 1 - dx;
		ty = 1 - dy;
		tz = 1 - dz;

		//rho[o(i, j, k)] += m;

		rho[o(i, j, k)] += m*tx*ty*tz;
		rho[o((i + 1) % M, j, k)] += m*dx*ty*tz;
		rho[o(i, (j + 1) % M, k)] += m*tx*dy*tz;
		rho[o((i + 1) % M, (j + 1) % M, k)] += m*dx*dy*tz;
		rho[o(i, j, (k + 1) % M)] += m*tx*ty*dz;
		rho[o((i + 1) % M, j, (k + 1) % M)] += m*dx*ty*dz;
		rho[o(i, (j + 1) % M, (k + 1) % M)] += m*tx*dy*dz;
		rho[o((i + 1) % M, (j + 1) % M, (k + 1) % M)] += m*dx*dy*dz;
	}
}

__global__ void propagateKernel(float *phi,float *rho)
{
	float phi_av;
	float r;
	int M = params.meshSize;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	r = (i - M / 2, 2) * (i - M / 2, 2) + (j - M / 2, 2)*(j - M / 2, 2) + (k - M / 2, 2)*(k - M / 2, 2);
	if (r < params.potentialLimit)  
	{
		// solve poisson equation
		phi_av = (phi[o(i, j, (k + 1) % M)] + phi[o(i, j, (k + M - 1) % M)]
			+ phi[o(i, (j + 1) % M, k)] + phi[o(i, (j + M - 1) % M, k)]
			+ phi[o((i + 1) % M, j, k)] + phi[o((i + M - 1) % M, j, k)]) / 6;

		phi[o(i, j, k)] += rho[o(i, j, k)] * params.G + (phi_av - phi[o(i, j, k)]) * params.diffusionTimestep;
	}
	else {
		phi_av = 0;
		phi[o(i, j, k)] = 0;
	}

	// reset rho to zero
	rho[o(i, j, k)] = 0;
}

__global__ void integrateKernel(float3 *r, float3 *v, float *phi, int N)
{
	// r contains particle positions, g is accelerations

	float dx, dy, dz;
	float tx, ty, tz;
	int i, j, k;  //mesh cell indices
	float3 g; //acceleration
	float B =  params.meshSize * params.meshCellSize;
	float dt = params.integrateTimestep; // timestep
	int M = params.meshSize;
	float dr = params.meshCellSize;

	//For each particle, use cloud in cell methodology to apportion force exterted by 8 neighbouring mesh points
	int p = blockIdx.x;

	if (p < N)
	{
		// calculate indices of cell containing particle at position r
		i = floor(r[p].x / dr) + M / 2;
		j = floor(r[p].y / dr) + M / 2;
		k = floor(r[p].z / dr) + M / 2;

		// calculate distances betweeen particle and cell centres
		dx = r[p].x - (i - M / 2) * dr;
		dy = r[p].y - (j - M / 2) * dr;
		dz = r[p].z - (k - M / 2) * dr;
		tx = 1 - dx;
		ty = 1 - dy;
		tz = 1 - dz;

		//g[p] = get_gc(phi, i, j, k, 1);

		// calculate accelerations from eight neighbouring cells
		g = get_gc(phi, i, j, k, tx*ty*tz)
			+ get_gc(phi, (i + 1) % M, j, k, dx*ty*tz)
			+ get_gc(phi, i, (j + 1) % M, k, tx*dy*tz)
			+ get_gc(phi, (i + 1) % M, (j + 1) % M, k, dx*dy*tz)
			+ get_gc(phi, i, j, (k + 1) % M, tx*ty*dz)
			+ get_gc(phi, (i + 1) % M, j, (k + 1) % M, dx*ty*dz)
			+ get_gc(phi, i, (j + 1) % M, (k + 1) % M, tx*dy*dz)
			+ get_gc(phi, (i + 1) % M, (j + 1) % M, (k + 1) % M, dx*dy*dz);

		// integrate accelerations to get velocities
		v[p].x += g.x * dt;
		v[p].y += g.y * dt;
		v[p].z += g.z * dt;

		// integrate velocities to get x, y, z positions
		r[p].x += v[p].x * dt;
		r[p].y += v[p].y * dt;
		r[p].z += v[p].z * dt;

		// Constrain x, y, z to mesh cube
		r[p].x = floatMod(r[p].x, B);
		r[p].y = floatMod(r[p].y, B);
		r[p].z = floatMod(r[p].z, B);
	}
}

GpuInterface::GpuInterface()
{
}

GpuInterface::~GpuInterface()
{
}

cudaError_t GpuInterface::CUDA_setParameters(SimParams *hostParams)
{
	cudaError_t cudaStatus;
	// copy parameters to constant memory
	cudaStatus = cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams));  // params is correct format even though it shows as error - used as a character string
	if (cudaStatus != cudaSuccess) 	fprintf(stderr, "cudaMemcpyToSymbol returned error code %d ", cudaStatus);
	return cudaStatus;
}

cudaError_t GpuInterface::CUDA_getStatus(float3 *vel, PhiStats *stats, ParticleStats *vstats, cudaGraphicsResource *vbo_res)
{
	cudaError_t cudaStatus;
	// copy parameters to constant memory
	//cudaStatus = cudaMemcpy(vel, dev_v, N * sizeof(float3), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) 	fprintf(stderr, "cudaMemcpy returned error code %d ", cudaStatus);

	// Calculate maximum particle velocity
	float3 mv	= *thrust::max_element(thrust::device, thrust::device_ptr<float3>(dev_v), thrust::device_ptr<float3>(dev_v + N),comp());
	vstats->max_v = sqrt(mv.x*mv.x + mv.y*mv.y + mv.z*mv.z);

	// Calculate average particle velocity
	thrust::plus<float> sum;
	float init = 0;
	vstats->avg_v  = sqrt(thrust::transform_reduce(thrust::device_ptr<float3>(dev_v), thrust::device_ptr<float3>(dev_v + N), mag_v(), init, sum))/N;

	// map OpenGL buffer object for writing from CUDA to phi VBO
	float *phiptr;
	size_t num_bytes;
	cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&phiptr, &num_bytes, vbo_res);

	// Calculate phi statistics
	thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(phiptr);
	stats->max_phi = *thrust::max_element(thrust::device, thrust::device_ptr<float>(phiptr), thrust::device_ptr<float>(phiptr + M));
	stats->min_phi = *thrust::min_element(thrust::device, thrust::device_ptr<float>(phiptr), thrust::device_ptr<float>(phiptr + M));
	stats->avg_phi = thrust::reduce(thrust::device_ptr<float>(phiptr), thrust::device_ptr<float>(phiptr + M)) /M;

	//stats->max_phi = *(thrust::max_element(d_ptr, d_ptr + N));
	//stats->min_phi = *(thrust::min_element(d_ptr, d_ptr + N));
	//stats->avg_phi = thrust::reduce(d_ptr, d_ptr + N) / N;

	cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);

	return cudaStatus;
}


cudaError_t GpuInterface::CUDA_Init(float3* v, float* m, unsigned int num, int size)
{
	cudaError_t cudaStatus;

	N = num;
	M = size;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Set GL device to this GPU
	//cudaStatus = cudaGLSetGLDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetGLDevice failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors  .
	cudaStatus = cudaMalloc((void**)&dev_v, N * sizeof(float3));
	cudaStatus = cudaMalloc((void**)&dev_m, M * M * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_v, v, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_m, m, M * M * M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t GpuInterface::CUDA_Accumulate(cudaGraphicsResource *vbo_res)
{
	cudaError_t cudaStatus;

	// map OpenGL buffer object for writing from CUDA to pos VBO
	float3 *pos_ptr;
	size_t num_bytes;
	cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&pos_ptr, &num_bytes, vbo_res);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "accKernel resource mapping failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	accumulateKernel << <N, 1 >> >(pos_ptr, dev_m, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "accKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}


cudaError_t GpuInterface::CUDA_Propagate(cudaGraphicsResource *vbo_res)
{
	cudaError_t cudaStatus;

	// map OpenGL buffer object for writing from CUDA to phi VBO
	float *phiptr;
	size_t num_bytes;
	cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&phiptr, &num_bytes, vbo_res);

	// Launch a kernel on the GPU with one thread for each mesh cell.
	dim3 grids(M, M, M);
	dim3 threads(1);
	propagateKernel << <grids, threads >> >(phiptr, dev_m);

	cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "propKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t GpuInterface::CUDA_Integrate(cudaGraphicsResource *particle_vbo_res, cudaGraphicsResource *mesh_vbo_res)
//Interop version
{
	cudaError_t cudaStatus;

	// map OpenGL buffer objects for writing from CUDA
	float3 *pos_ptr;
	float *phi_ptr;
	size_t num_bytes;
	cudaStatus = cudaGraphicsMapResources(1, &particle_vbo_res, 0);
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&pos_ptr, &num_bytes, particle_vbo_res);
	cudaStatus = cudaGraphicsMapResources(1, &mesh_vbo_res, 0);
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&phi_ptr, &num_bytes, mesh_vbo_res);
	cudaStatus = cudaGetLastError();
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "intKernel resource mapping failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	integrateKernel <<< N, 1 >>>(pos_ptr, dev_v, phi_ptr, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "intKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaGraphicsUnmapResources(1, &particle_vbo_res, 0);
	cudaStatus = cudaGraphicsUnmapResources(1, &mesh_vbo_res, 0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "intKernel resource un-mapping failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

void GpuInterface::CUDA_Close()
{
	cudaFree(dev_v);
	cudaFree(dev_m);
}


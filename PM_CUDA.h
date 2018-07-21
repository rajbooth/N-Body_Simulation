#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include "SimParams.h"

class GpuInterface
{
public:
	GpuInterface();
	~GpuInterface();

	cudaError_t CUDA_Init(float3*, float*, unsigned int size, int spacing);
	cudaError_t CUDA_Integrate(cudaGraphicsResource*,cudaGraphicsResource*); //Interop version
	cudaError_t CUDA_Accumulate(cudaGraphicsResource*); //Interop version
	cudaError_t CUDA_Propagate(cudaGraphicsResource*); //Interop version
	cudaError_t CUDA_setParameters(SimParams *hostParams);
	cudaError_t CUDA_getStatus(float3*, PhiStats*, ParticleStats*, cudaGraphicsResource*);

	void CUDA_Close();

private:
	float3 *dev_v;	// velocity
	float  *dev_m;	// mass - density

	unsigned int N; //number of particles
	unsigned int M; //mesh size
};

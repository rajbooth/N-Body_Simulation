#include "SineWave.h"

// Cuda function
void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

// constants
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

float g_fAnim = 0.0;

SineWave::SineWave()
{
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	createVBO(&vbo, size, cudaGraphicsMapFlagsWriteDiscard);
}

SineWave::~SineWave()
{
	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}

void SineWave::Draw()
{
// run CUDA kernel to generate vertex positions
SineWave::runCuda(&cuda_vbo_resource);

// render from the vbo
glBindBuffer(GL_ARRAY_BUFFER, vbo);
glVertexPointer(4, GL_FLOAT, 0, 0);

glEnableClientState(GL_VERTEX_ARRAY);
glColor3f(1.0, 0.0, 1.0);
glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
glDisableClientState(GL_VERTEX_ARRAY);

g_fAnim += 0.01f;
}

void SineWave::runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	//    dim3 block(8, 8, 1);
	//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

	launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

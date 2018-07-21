#pragma once
#include <vector>
#include <iostream>
#include <assert.h>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp> 
#include <glm/gtc/matrix_transform.hpp>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include "PM_CUDA.h"

class Element
{
public:
	Element();
	~Element();

	void Draw();
	void Update();
	void SetProgram(GLuint shaderName);
	void createVBO(GLuint*, size_t, unsigned int);
	void deleteVBO(GLuint *, struct cudaGraphicsResource *);

	GpuInterface* gpu;

protected:
	GLuint vao;
	GLuint program;
	std::vector<GLuint> vbos;
	std::vector<float4> vertices;
};

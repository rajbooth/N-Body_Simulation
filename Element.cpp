#include "Element.h"

Element::Element()
{
}

Element::~Element()
{
}

void Element::Draw()
{
}

void Element::Update()
{
}

void Element::SetProgram(GLuint shaderName)
{
}

void Element::createVBO(GLuint *vbo, size_t size, unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	SDK_CHECK_ERROR_GL();
}

void Element::deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}




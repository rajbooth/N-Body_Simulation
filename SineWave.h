#pragma once
#include "Element.h"
class SineWave :
	public Element
{
public:
	SineWave();
	~SineWave();

	void Draw();

private:
	GLuint vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	void *d_vbo_buffer = NULL;

	void runCuda(struct cudaGraphicsResource **);
};


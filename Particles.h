#pragma once
#include "Element.h"
#include "Mesh.h"
#include "Shader_Manager.h"

extern float sliceWidth;
extern int pixelSize;

class Particles :
	public Element
{
public:
	Particles(SimParams*, Mesh*);
	~Particles();

	void Draw(const glm::mat4&);
	void execCUDA();
	void calcStatistics(int frame);

	Mesh* mesh; // Mesh associated with this particle set
	ParticleStats vstats;
	GpuInterface* gpu;

private:
	GLuint vertexVBO;
	GLuint colVBO;
	struct cudaGraphicsResource *cuda_particleVBO_resource;


	std::vector<float3> accel;
	std::vector<float3> velocity;
	std::vector<float3> position;
	std::vector<float4> colour;

	int N; //number of particles
	int M; //mesh size
};


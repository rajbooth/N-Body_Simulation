#pragma once
#include "Element.h"
#include "Shader_Manager.h"

extern float meshAlpha;
extern float meshIntensity;
extern float sliceWidth;
extern float sliceZ;

class Mesh :
	public Element
{
public:
	Mesh(SimParams*);
	~Mesh();

	void Draw(const glm::mat4&);
	void SetVertices(float dr, int mode);

	std::vector<float> rho; // density
	std::vector<float> phi; // potential 

	struct cudaGraphicsResource *cuda_meshVBO_resource;
	PhiStats stats;

private:
	GLuint vertexVBO;
	GLuint phiVBO;

	GLuint program;
	std::vector<float4> vertices;

	int M; //mesh size
};


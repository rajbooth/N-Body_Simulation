#include "Particles.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <random>

extern float particleAlpha;
extern bool captureHisto;
extern bool displayHisto;
extern bool centreHisto;

Particles::Particles(SimParams* params, Mesh* m)
{
	N = params->numParticles;
	M = params->meshSize;
	float dr = params->meshCellSize;
	float v_scale = params->velocityScaling;

	mesh = m;
	gpu = new GpuInterface();

	GLuint vao;
	float4 col;
	float3 pos;
	float r;

	//Set simulation parameters
	gpu->CUDA_setParameters(params);

	/* initialize random seed: */
	srand(time(NULL));

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 2.0);

	int i = 0;
	
	while (i<N)
	{
		//Create random starting positions and velocities
		pos.x = (float)(rand() % (M * (int)(dr * 1000)) - (M / 2 * dr * 1000)) / 1000;
		pos.y = (float)(rand() % (M * (int)(dr * 1000)) - (M / 2 * dr * 1000)) / 1000;
		pos.z = (float)(rand() % (M * (int)(dr * 1000)) - (M / 2 * dr * 1000)) / 1000;
		r = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z;

		if (r < params->initialSphere || (i % 10 >= params->overDensity * 10 && r < params->potentialLimit))
		{
			i++;
			position.push_back(pos);
			velocity.push_back(make_float3(distribution(generator) * v_scale, distribution(generator) * v_scale, distribution(generator) * v_scale));
			accel.push_back(make_float3(0, 0, 0));
			if (r > 0.6)
				col = make_float4(0.0, 0.0, 1.0, 1.0);
			else col = make_float4(1.0, 0.5, 0.0, 1.0);
			colour.push_back(col);
		}
	}

	// Initialise velocity statistics
	vstats.avg_v = 0;
	vstats.max_v = 0;
	vstats.min_v = 0;

	for (i = 0; i < 56; i++)
	{
		vstats.density[i] = 0;
		vstats.velocity[i] = 0;
	}

	//Initialise CUDA buffers
	cudaError_t cudaStatus = gpu->CUDA_Init((float3*)&velocity[0], &mesh->rho[0], N, M);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "initCUDA failed!");
	}

	// Create shader program
	Managers::Shader_Manager* sm = new Managers::Shader_Manager();
	sm->CreateProgram("particleShader", "Particle_Vertex_Shader.glsl", "Fragment_Shader.glsl");
	program = Managers::Shader_Manager::GetShader("particleShader");

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * N, &position[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(&cuda_particleVBO_resource, vertexVBO, cudaGraphicsMapFlagsNone);

	//Add colour array
	glGenBuffers(1, &colVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * N, &colour[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);

	glBindVertexArray(0);
}


Particles::~Particles()
{
	if (vertexVBO)
	{
		deleteVBO(&vertexVBO, cuda_particleVBO_resource);
	}
}

void Particles::Draw(const glm::mat4& MVP_matrix)
{ 
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);

	glUseProgram(program);
	glUniform4f(glGetUniformLocation(program, "vert_color"), 1.0, 1.0, 0.0, particleAlpha);
	glUniform1f(glGetUniformLocation(program, "sliceWidth"), sliceWidth);
	glUniform1f(glGetUniformLocation(program, "sliceZ"), sliceZ);
	GLuint MatrixID = glGetUniformLocation(program, "MVP");
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP_matrix[0][0]);

	glBindVertexArray(vao);

	glPointSize(pixelSize);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawArrays(GL_POINTS, 0, N);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void Particles::execCUDA()
{
	cudaError_t cudaStatus;

	cudaStatus = gpu->CUDA_Accumulate(cuda_particleVBO_resource);
	cudaStatus = gpu->CUDA_Propagate(mesh->cuda_meshVBO_resource);
	cudaStatus = gpu->CUDA_Integrate(cuda_particleVBO_resource, mesh->cuda_meshVBO_resource);
	cudaStatus = gpu->CUDA_getStatus(&velocity[0], &mesh->stats, &vstats, mesh->cuda_meshVBO_resource);
}

void Particles::calcStatistics(int frame)
{
	// Calculate radial position and velocity of each particle and add to bins
	int i;
	int r;
	int n;
	float dens[56];
	float vel[56];

	cudaError_t cudaStatus;

	n = frame < 5 ? frame : 5;

	captureHisto = false;
	for (i = 0; i < 56; i++)
	{
		dens[i] = 0;
		vel[i] = 0;
	}
	// Get velocity and position arrays from device
	glGetNamedBufferSubData(vertexVBO, 0, N * sizeof(float3), &position[0]);

	for (i = 0; i < N; i++)
	{
		r = floor(sqrt(position[i].x*position[i].x + position[i].y*position[i].y + position[i].z*position[i].z) * 32);
		if (r < 56)
		{
			dens[r] += 1;
			vel[r] += sqrt(velocity[i].x*velocity[i].x + velocity[i].y*velocity[i].y + velocity[i].z*velocity[i].z) / dens[r];
		}
	}
	for (i = 0; i < 56; i++)
	{
		vstats.velocity[i] += (vel[i] - vstats.velocity[i]) / n;
		vstats.density[i] = dens[i];
		//printf("i=%d, count = %d, vel = %f \n", i, vstats.density[i], vstats.velocity[i]);
	}
}
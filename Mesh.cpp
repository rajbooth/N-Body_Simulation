#include "Mesh.h"
#include <time.h>       /* time */
#include <random>

Mesh::Mesh(SimParams* params)
{
	GLuint vao;
	float potential = 0.0f;

	M = params->meshSize;
	float dr = params->meshCellSize;

	// Initialise mesh phi statistics
	stats.avg_phi = 0;
	stats.max_phi = 0;
	stats.min_phi = 0;

	/* initialize random seed: */
	srand(time(NULL));

	//create a M*M*M mesh
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			for (int k = 0; k < M; k++)
			{
				float4 vert = make_float4(i * dr - (M / 2 * dr), j * dr - (M / 2 * dr), k * dr - (M / 2 * dr), 1.0);
				if (params->meshMode > 0)
				{
					vert.x += dr / 2 - (float)(rand() % 100) / 100 * dr;
					vert.y += dr / 2 - (float)(rand() % 100) / 100 * dr;
					vert.z += dr / 2 - (float)(rand() % 100) / 100 * dr;
				}

				vertices.push_back(vert);
				potential = 0.0f;

				phi.push_back(potential);
				rho.push_back(potential);
			}
		}
	}

	// Create shader program
	Managers::Shader_Manager* sm = new Managers::Shader_Manager();
	sm->CreateProgram("meshShader", "Mesh_Vertex_Shader.glsl", "Fragment_Shader.glsl");
	program = Managers::Shader_Manager::GetShader("meshShader");

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * M*M*M, &vertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);

	//Add phi array
	glGenBuffers(1, &phiVBO);
	glBindBuffer(GL_ARRAY_BUFFER, phiVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * M*M*M, &phi[0], GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(&cuda_meshVBO_resource, phiVBO, cudaGraphicsMapFlagsNone);

	glBindVertexArray(0);
}


Mesh::~Mesh()
{
	if (phiVBO)
	{
		deleteVBO(&phiVBO, cuda_meshVBO_resource);
	}
}

void Mesh::Draw(const glm::mat4 &MVP_matrix)
{
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, phiVBO);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT,  GL_FALSE, sizeof(float), (void*)0);

	glUseProgram(program);
	glUniform4f(glGetUniformLocation(program, "vert_color"), meshIntensity, meshIntensity, meshIntensity, meshAlpha);
	glUniform1f(glGetUniformLocation(program, "sliceWidth"), sliceWidth);
	glUniform1f(glGetUniformLocation(program, "sliceZ"), sliceZ);
	glUniform1f(glGetUniformLocation(program, "max_phi"), stats.max_phi);
	glUniform1f(glGetUniformLocation(program, "min_phi"), stats.min_phi);
	glUniform1f(glGetUniformLocation(program, "avg_phi"), stats.avg_phi);
	GLuint MatrixID = glGetUniformLocation(program, "MVP");
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP_matrix[0][0]);

	glBindVertexArray(vao);
		
	glEnableClientState(GL_VERTEX_ARRAY);
	glPointSize(1.0f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawArrays(GL_POINTS, 0, M*M*M);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void Mesh::SetVertices(float dr, int mode)
{
	int m;
	float noise; 

	vertices.clear();

	//create a M*M*M mesh
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			for (int k = 0; k < M; k++)
			{
				float4 vert = make_float4(i * dr - (M / 2 * dr), j * dr - (M / 2 * dr), k * dr - (M / 2 * dr), 1.0);
				if (mode > 0)
				{
					vert.x += dr / 2 - (float)(rand() % 100) / 100 * dr;
					vert.y += dr / 2 - (float)(rand() % 100) / 100 * dr;
					vert.z += dr / 2 - (float)(rand() % 100) / 100 * dr;
				}

				vertices.push_back(vert);
			}
		}
	}

	// Update vertex VBO
	glNamedBufferSubData(vertexVBO, 0, sizeof(float4)*M*M*M, &vertices[0]);
}
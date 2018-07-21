#pragma once

#include "vector_types.h"

// simulation parameters
struct SimParams
{
	int meshSize;
	float meshCellSize;
	int numParticles;
	int meshMode;
	float particleMass;
	float G;
	float integrateTimestep;
	float diffusionTimestep;

	float potentialLimit;
	float particleRadius;
	float massScaling;
	float velocityScaling;
	float initialSphere;
	float overDensity;

	float3 worldOrigin;
	float3 cellSize;
};

// simulation statistics
struct PhiStats
{
	float max_phi;
	float min_phi;
	float avg_phi;
};

struct ParticleStats {
	float max_v;
	float min_v;
	float avg_v;
	float density[56];
	float velocity[56];
};

struct comp
{
	template <typename T>
	__host__ __device__
		bool operator()(T &t1, T &t2) {
		return ((t1.x*t1.x + t1.y*t1.y + t1.z*t1.z) < (t2.x*t2.x + t2.y*t2.y + t2.z*t2.z));
	}
};

struct mag_v : public std::unary_function<float3, float>
{
	__host__ __device__
		float operator()(float3 v) { return (v.x*v.x + v.y*v.y + v.z*v.z); }
};
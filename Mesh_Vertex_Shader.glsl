#version 450 core

layout(location = 0) in vec4 in_position;
layout(location = 1) in float phi;

uniform vec4 vert_color;
uniform mat4 MVP;
uniform float sliceWidth;
uniform float sliceZ;
uniform float max_phi;
uniform float min_phi;
uniform float avg_phi;

out vec4 color;

void main()
{
	if (sliceWidth != 0 && (in_position.z < sliceZ - sliceWidth || in_position.z > sliceZ + sliceWidth) || vert_color.a == 0)
	{
		// Hide mesh
		color = vec4(0.0, 0.0, 0.0, 0.0);
		gl_PointSize = 0;
	}
	else
	{
		float r, b, g, alpha;
		float m = 2.0;
		float p = 8.0;
		if (phi >= max_phi * p)
		{
			r = 1.0;
			b = 1.0;
			g = (phi - p * max_phi) / (max_phi * p );
			alpha = 0.8;
		}
		else if (phi > avg_phi * m) 
		{
			r = 1.0;
			b = (phi - avg_phi*m) / (max_phi * p - avg_phi*m);
			g = 0.0;
			alpha = 0.6;
		}	
		else
		{
			r = (phi - min_phi) / (avg_phi * m - min_phi);
			b = 0.0;
			g = 0.0;
			alpha = 0.4;
		}
		if (r < vert_color.r) r = vert_color.r;
		if (g < vert_color.g) g = vert_color.g;
		if (b < vert_color.b) b = vert_color.b;
		color = vec4(r, g , b, alpha);
		gl_PointSize = log(phi);
	}

	gl_Position = MVP * in_position;
}
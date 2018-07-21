#version 450 core

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

uniform vec4 vert_color;
uniform mat4 MVP;
uniform float sliceWidth;
uniform float sliceZ;

out vec4 color;

void main()
{
	if (sliceWidth != 0 && (in_position.z < sliceZ - sliceWidth || in_position.z > sliceZ + sliceWidth) || vert_color.a == 0)
	{
		// Hide particles
		color = vec4(0.0, 0.0, 0.0, 0.0);
		gl_PointSize = 0;
	}
	else 
		color = vert_color;

	gl_Position = MVP * in_position;
}
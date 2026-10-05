#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(std140, set = 0, binding = 0) uniform ObjectUniforms {
	mat4 mvp;
	vec4 tint;
} object_uniforms;

void main() {
	vec4 position = object_uniforms.mvp * vec4(in_position, 1.0);
	position.z -= 1.0e-4 * position.w;
	gl_Position = position;
}

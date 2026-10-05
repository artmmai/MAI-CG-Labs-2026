#include "application.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "octahedron.hpp"
#include "math.hpp"

namespace application {

namespace {

// Макеты данных, используемые совместно с шейдерами

// Vertex attributes: location 0 = position, location 1 = colour.
struct Vertex {
	float position[3];
	float color[3];
};

// Единый блок для каждого объекта. Соответствует "ObjectUniforms` в шейдерах (std140):
struct ObjectUniforms {
	float mvp[16];
	float tint[4];
};
static_assert(sizeof(ObjectUniforms) == 80, "ObjectUniforms must match the std140 layout");

// Описание сцены

constexpr uint32_t max_objects = 4;

enum Trajectory : int {
	trajectory_circle = 0,
	trajectory_lissajous,
	trajectory_trefoil,
};

const char* const trajectory_names[] = {
	"Circle (tilted)",
	"Lissajous figure",
	"Trefoil knot",
};

struct SceneObject {
	// Параметры аффинного преобразования
	float position[3] = {0.0f, 0.0f, 4.5f};
	float rotation_deg[3] = {0.0f, 0.0f, 0.0f};
	float scale[3] = {1.0f, 1.0f, 1.0f};

	// Оттенок, умноженный на цвета каждой вершины
	float tint[3] = {1.0f, 1.0f, 1.0f};

	// Анимация по траектории
	bool animated = true;
	int trajectory = trajectory_trefoil;
	float radius = 1.2f;     // размер траектории в плоскости XY
	float depth = 0.8f;      // размер траектории вдоль Z
	float spin_deg = 40.0f;  // скорость самоповорота, градусы в секунду анимации
	float phase = 0.0f;      // смещение по времени, чтобы объекты не перемещались синхронно
};

struct Projection {
	bool perspective = true;
	float fov_deg = 60.0f;      // vertical FOV
	float ortho_height = 5.0f;  // высота видимого поля для орфографического режима
	float near_plane = 0.1f;
	float far_plane = 100.0f;
};

std::array<SceneObject, max_objects> objects;
std::array<ObjectUniforms, max_objects> object_uniforms;
int object_count = 3;
int selected_object = 0;

Projection projection;

bool animation_playing = true;
float animation_speed = 1.0f;
double animation_time = 0.0;
double previous_time = -1.0;

SceneObject makeDefaultObject(uint32_t index) {
	SceneObject o;
	o.phase = 1.7f * float(index);

	switch (index) {
	case 0:
		break; // defaults: centre, trefoil knot
	case 1:
		o.position[0] = -2.6f; o.position[1] = -0.6f; o.position[2] = 6.0f;
		o.tint[0] = 1.0f; o.tint[1] = 0.65f; o.tint[2] = 0.4f;
		o.trajectory = trajectory_circle;
		o.radius = 0.6f; o.depth = 0.0f;
		o.spin_deg = -60.0f;
		break;
	case 2:
		o.position[0] = 2.6f; o.position[1] = 0.6f; o.position[2] = 6.0f;
		o.tint[0] = 0.5f; o.tint[1] = 0.8f; o.tint[2] = 1.0f;
		o.trajectory = trajectory_lissajous;
		o.radius = 0.8f; o.depth = 0.8f;
		o.spin_deg = 90.0f;
		break;
	default:
		o.position[0] = 0.0f; o.position[1] = 2.0f; o.position[2] = 7.0f;
		o.tint[0] = 0.6f; o.tint[1] = 1.0f; o.tint[2] = 0.6f;
		o.animated = false;
		o.scale[0] = o.scale[1] = o.scale[2] = 0.7f;
		break;
	}

	return o;
}

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr; // постоянно отображаемый указатель хоста
};

Buffer vertex_buffer;
Buffer index_buffer;      // треугольники (грани)
Buffer edge_index_buffer; // список линий (ребер)
uint32_t index_count = 0;
uint32_t edge_index_count = 0;

std::array<Buffer, max_objects> uniform_buffers; // по одному на объект

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
std::array<VkDescriptorSet, max_objects> descriptor_sets{}; // по одному на объект

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline_faces = VK_NULL_HANDLE;
VkPipeline pipeline_edges = VK_NULL_HANDLE;

bool show_edges = true;

bool createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo info{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate Vulkan buffer\n";
		return false;
	}

	out.mapped = info.pMappedData;
	return out.mapped != nullptr;
}

bool uploadToBuffer(Buffer& buffer, const void* data, size_t size) {
	std::memcpy(buffer.mapped, data, size);
	return vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation,
	                          0, VK_WHOLE_SIZE) == VK_SUCCESS;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

VkShaderModule loadShader(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader '" << path
		          << "'. Is the working directory the project root and are shaders compiled?\n";
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % 4 != 0) {
		std::cerr << "Shader '" << path << "' is not a valid SPIR-V file\n";
		return VK_NULL_HANDLE;
	}

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from '" << path << "'\n";
		return VK_NULL_HANDLE;
	}

	return module;
}

bool createPipeline(const char* vertex_path, const char* fragment_path,
                    VkPrimitiveTopology topology, VkPipeline& out) {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = loadShader(vertex_path);
	VkShaderModule fragment_shader = loadShader(fragment_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_bindings[] = {
		{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		},
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo raster_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,          // back-face culling
		.frontFace = VK_FRONT_FACE_CLOCKWISE,       // вращение по часовой стрелке
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo blend_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input_state,
		.pInputAssemblyState = &assembly_state,
		.pViewportState = &viewport_state,
		.pRasterizationState = &raster_state,
		.pMultisampleState = &multisample_state,
		.pDepthStencilState = &depth_state,
		.pColorBlendState = &blend_state,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &out);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

void makeVertexColor(const std::array<float, 3>& p, float out[3]) {
	for (int i = 0; i < 3; ++i) {
		out[i] = 0.5f + 0.5f * p[i];
	}
}

void destroyResources() {
	auto& context = graphics::internal::context;

	vkDestroyPipeline(context.device, pipeline_edges, nullptr);
	vkDestroyPipeline(context.device, pipeline_faces, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	// Уничтожение пула также освобождает все выделенные из него наборы дескрипторов
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (Buffer& b : uniform_buffers) {
		destroyBuffer(b);
	}
	destroyBuffer(edge_index_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	pipeline_edges = pipeline_faces = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	descriptor_set_layout = VK_NULL_HANDLE;
	descriptor_sets.fill(VK_NULL_HANDLE);
}

// Анимация / матрицы
void trajectoryOffset(const SceneObject& o, float t, float out[3]) {
	switch (o.trajectory) {
	case trajectory_circle:
		out[0] = o.radius * std::cos(t);
		out[1] = o.radius * std::sin(t);
		out[2] = o.depth * std::sin(t);
		break;
	case trajectory_lissajous:
		out[0] = o.radius * std::sin(3.0f * t);
		out[1] = o.radius * std::sin(2.0f * t);
		out[2] = o.depth * std::cos(t);
		break;
	case trajectory_trefoil:
	default:
		out[0] = o.radius * (std::sin(t) + 2.0f * std::sin(2.0f * t)) / 3.0f;
		out[1] = o.radius * (std::cos(t) - 2.0f * std::cos(2.0f * t)) / 3.0f;
		out[2] = o.depth * -std::sin(3.0f * t);
		break;
	}
}

// model = Translation * (animated spin) * (user rotation) * Scale
math::Mat4 makeModelMatrix(const SceneObject& o, double time) {
	float offset[3] = {0.0f, 0.0f, 0.0f};
	math::Mat4 spin = math::Mat4::identity();

	if (o.animated) {
		const float t = float(time) + o.phase;
		trajectoryOffset(o, t, offset);

		const float angle = math::radians(o.spin_deg) * float(time);
		spin = math::rotationY(angle) * math::rotationX(0.5f * angle);
	}

	const math::Mat4 translate = math::translation(o.position[0] + offset[0],
	                                               o.position[1] + offset[1],
	                                               o.position[2] + offset[2]);

	const math::Mat4 rotate = math::rotationZ(math::radians(o.rotation_deg[2])) *
	                          math::rotationY(math::radians(o.rotation_deg[1])) *
	                          math::rotationX(math::radians(o.rotation_deg[0]));

	const math::Mat4 scale = math::scaling(o.scale[0], o.scale[1], o.scale[2]);

	return translate * spin * rotate * scale;
}

// Переключаемая проекция. Камера устанавливается в начале координат и смотрит вдоль +Z.
math::Mat4 makeProjectionMatrix(float aspect) {
	if (projection.perspective) {
		return math::perspective(math::radians(projection.fov_deg), aspect,
		                         projection.near_plane, projection.far_plane);
	}

	const float half_height = 0.5f * projection.ortho_height;
	const float half_width = half_height * aspect;
	// Vulkan: Y указывает вниз, так что "top" имеет отрицательный знак.
	return math::orthographic(-half_width, half_width, -half_height, half_height,
	                          projection.near_plane, projection.far_plane);
}

// UI
void drawUI() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 1: Octahedron");

	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));

	// Проекция
	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::RadioButton("Perspective", projection.perspective)) {
			projection.perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !projection.perspective)) {
			projection.perspective = false;
		}

		if (projection.perspective) {
			ImGui::SliderFloat("Vertical FOV (deg)", &projection.fov_deg, 20.0f, 120.0f);
		} else {
			ImGui::SliderFloat("View height", &projection.ortho_height, 1.0f, 20.0f);
		}

		ImGui::SliderFloat("Near", &projection.near_plane, 0.01f, 5.0f, "%.2f");
		ImGui::SliderFloat("Far", &projection.far_plane, 10.0f, 200.0f, "%.0f");
	}

	// Элементы управления анимацией
	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation_time = 0.0;
		}
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f, "%.2fx");
	}

	// Несколько объектов, каждый из которых имеет свой собственный набор дескрипторов
	if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderInt("Objects", &object_count, 1, int(max_objects));
		if (object_count < 1) object_count = 1;
		if (object_count > int(max_objects)) object_count = int(max_objects);

		if (selected_object >= object_count) selected_object = object_count - 1;
		if (object_count > 1) {
			ImGui::SliderInt("Selected", &selected_object, 0, object_count - 1);
		}

		ImGui::Checkbox("Show edges", &show_edges);
	}

	SceneObject& o = objects[size_t(selected_object)];

	ImGui::Separator();
	ImGui::Text("Object #%d", selected_object);

	// position / rotation / scale
	if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position (Y down)", o.position, 0.02f);
		ImGui::DragFloat3("Rotation (deg)", o.rotation_deg, 0.5f);
		ImGui::DragFloat3("Scale", o.scale, 0.01f, 0.05f, 10.0f);

		float uniform_scale = (o.scale[0] + o.scale[1] + o.scale[2]) / 3.0f;
		if (ImGui::SliderFloat("Uniform scale", &uniform_scale, 0.05f, 4.0f)) {
			o.scale[0] = o.scale[1] = o.scale[2] = uniform_scale;
		}

		if (ImGui::Button("Reset object")) {
			o = makeDefaultObject(uint32_t(selected_object));
		}
	}

	// colour 
	if (ImGui::CollapsingHeader("Colour", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Tint", o.tint);
		ImGui::TextDisabled("Tint is multiplied by the per-vertex colour");
	}

	// параметры траектории 
	if (ImGui::CollapsingHeader("Trajectory", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Animate this object", &o.animated);
		ImGui::Combo("Type", &o.trajectory, trajectory_names,
		             int(sizeof(trajectory_names) / sizeof(trajectory_names[0])));
		ImGui::SliderFloat("Radius (XY)", &o.radius, 0.0f, 4.0f);
		ImGui::SliderFloat("Depth (Z)", &o.depth, 0.0f, 3.0f);
		ImGui::SliderFloat("Spin (deg/s)", &o.spin_deg, -360.0f, 360.0f);
		ImGui::SliderFloat("Phase", &o.phase, 0.0f, 2.0f * math::pi);
	}

	ImGui::End();
}

} // namespace


// Application interface
bool initialize() {
	auto& context = graphics::internal::context;

	for (uint32_t i = 0; i < max_objects; ++i) {
		objects[i] = makeDefaultObject(i);
	}

	// Geometry: vertex buffer + index buffers
	const geometry::Mesh mesh = geometry::makeOctahedron();

	std::vector<Vertex> vertices(mesh.positions.size());
	for (size_t i = 0; i < vertices.size(); ++i) {
		std::memcpy(vertices[i].position, mesh.positions[i].data(), sizeof(vertices[i].position));
		makeVertexColor(mesh.positions[i], vertices[i].color);
	}

	index_count = uint32_t(mesh.triangles.size());
	edge_index_count = uint32_t(mesh.edges.size());

	const VkDeviceSize vertices_size = sizeof(Vertex) * vertices.size();
	const VkDeviceSize indices_size = sizeof(uint32_t) * mesh.triangles.size();
	const VkDeviceSize edges_size = sizeof(uint32_t) * mesh.edges.size();

	if (!createMappedBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createMappedBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
	    !createMappedBuffer(edges_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, edge_index_buffer)) {
		destroyResources();
		return false;
	}

	if (!uploadToBuffer(vertex_buffer, vertices.data(), size_t(vertices_size)) ||
	    !uploadToBuffer(index_buffer, mesh.triangles.data(), size_t(indices_size)) ||
	    !uploadToBuffer(edge_index_buffer, mesh.edges.data(), size_t(edges_size))) {
		std::cerr << "Failed to flush geometry buffers\n";
		destroyResources();
		return false;
	}

	//  Uniform buffers: one per object 
	for (Buffer& b : uniform_buffers) {
		if (!createMappedBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, b)) {
			destroyResources();
			return false;
		}
	}

	// Descriptor set layout: binding 0 = uniform buffer
	const VkDescriptorSetLayoutBinding descriptor_set_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(descriptor_set_bindings) / sizeof(descriptor_set_bindings[0]),
		.pBindings = descriptor_set_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &descriptor_set_layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		destroyResources();
		return false;
	}

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		destroyResources();
		return false;
	}

    // Максимальное количество наборов дескрипторов, которые можно выделить из пула
	//  Descriptor pool: N descriptors per set * S sets
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 * max_objects,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_objects,
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		destroyResources();
		return false;
	}

	// Allocate one descriptor set per object (all of the same layout)
	std::array<VkDescriptorSetLayout, max_objects> set_layouts;
	set_layouts.fill(descriptor_set_layout);

	const VkDescriptorSetAllocateInfo descriptor_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = set_layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &descriptor_set_info,
	                             descriptor_sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		destroyResources();
		return false;
	}

	// Bind every uniform buffer to its own descriptor set
	std::array<VkDescriptorBufferInfo, max_objects> buffer_infos;
	std::array<VkWriteDescriptorSet, max_objects> writes;

	for (uint32_t i = 0; i < max_objects; ++i) {
		buffer_infos[i] = {
			.buffer = uniform_buffers[i].buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_sets[i],
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}

	vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);

	//  Pipelines: filled triangles + edges (line list) 
	if (!createPipeline("shaders/object.vert.spv", "shaders/object.frag.spv",
	                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, pipeline_faces) ||
	    !createPipeline("shaders/edge.vert.spv", "shaders/edge.frag.spv",
	                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST, pipeline_edges)) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	destroyResources();
}

void update(double time) {
    // Время с разницей. Сжато, чтобы при длительной остановке (перетаскивание окна, точка останова) объекты не телепортировались.
	double dt = previous_time < 0.0 ? 0.0 : time - previous_time;
	previous_time = time;
	if (dt > 0.1) dt = 0.1;

	if (animation_playing) {
		animation_time += dt * double(animation_speed);
	}

	drawUI();

	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	if (projection.far_plane < projection.near_plane + 1.0f) {
		projection.far_plane = projection.near_plane + 1.0f;
	}

	const math::Mat4 proj = makeProjectionMatrix(aspect);

    // Камера зафиксирована в исходной точке (матрица просмотра = идентификация), поэтому MVP = P * M.
	for (int i = 0; i < object_count; ++i) {
		const SceneObject& o = objects[size_t(i)];
		const math::Mat4 mvp = proj * makeModelMatrix(o, animation_time);

		ObjectUniforms& u = object_uniforms[size_t(i)];
		std::memcpy(u.mvp, mvp.e, sizeof(u.mvp));
		u.tint[0] = o.tint[0];
		u.tint[1] = o.tint[1];
		u.tint[2] = o.tint[2];
		u.tint[3] = 1.0f;
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

    // Завершено выделение предыдущего кадра, графический процессор обработал его с помощью
    // однородных буферов: их можно безопасно обновлять.
	for (int i = 0; i < object_count; ++i) {
		uploadToBuffer(uniform_buffers[size_t(i)], &object_uniforms[size_t(i)], sizeof(ObjectUniforms));
	}

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	VkClearValue clear_values[2]{};

	clear_values[0].color.float32[0] = 0.10f;
	clear_values[0].color.float32[1] = 0.11f;
	clear_values[0].color.float32[2] = 0.14f;
	clear_values[0].color.float32[3] = 1.0f;

	clear_values[1].depthStencil.depth = 1.0f;
	clear_values[1].depthStencil.stencil = 0;

	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {.extent = context.swapchain_extent},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = {.extent = context.swapchain_extent};

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer, &vertex_buffer_offset);

	// Заполнение граней. Та же геометрия, но для каждого объекта задан другой дескриптор
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_faces);
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 1, &descriptor_sets[size_t(i)], 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	if (show_edges) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_edges);
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
		vkCmdBindIndexBuffer(fd.command_buffer, edge_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		for (int i = 0; i < object_count; ++i) {
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                        pipeline_layout, 0, 1, &descriptor_sets[size_t(i)], 0, nullptr);
			vkCmdDrawIndexed(fd.command_buffer, edge_index_count, 1, 0, 0, 0);
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
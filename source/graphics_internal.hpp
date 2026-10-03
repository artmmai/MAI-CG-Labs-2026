#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <array>
#include <vector>

struct GLFWwindow;

namespace graphics::internal {

struct Context {
	VkPhysicalDevice physical_device;
	VkDevice device;

	VmaAllocator allocator;

	VkQueue graphics_queue;
	uint32_t graphics_queue_index;

	VkFormat swapchain_format;
	VkExtent2D swapchain_extent;

	VkRenderPass render_pass;
};

struct FrameData {
	VkFramebuffer framebuffer;
	VkCommandBuffer command_buffer;
};

// Структуры для рендеринга октаэдров
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        
        // Атрибуты позиции
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        
        // Атрибут цвета
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        
        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

extern Context context;
extern uint32_t vk_swapchain_current_image;
extern VkBuffer vertexBuffer;
extern VmaAllocation vertexBufferAllocation;
extern VkBuffer indexBuffer;
extern VmaAllocation indexBufferAllocation;
extern VkBuffer uniformBuffers[3];
extern VmaAllocation uniformBuffersAllocations[3];
extern void* uniformBuffersMapped[3];
extern VkDescriptorSetLayout descriptorSetLayout;
extern VkDescriptorPool descriptorPool;
extern VkDescriptorSet descriptorSets[3];
extern VkPipelineLayout pipelineLayout;
extern VkPipeline graphicsPipeline;
extern std::vector<Vertex> vertices;
extern std::vector<uint32_t> indices;


bool initialize(GLFWwindow* const window);
void shutdown();

void resize(uint32_t width, uint32_t height);

FrameData prepare();
void submitAndPresent();

} // namespace graphics::internal
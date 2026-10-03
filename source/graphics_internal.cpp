#include "graphics_internal.hpp"

#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <array>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100 4189 4324)
#endif // _MSC_VER
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <backends/imgui_impl_vulkan.h>

namespace graphics::internal {

namespace {

VkInstance vk_instance;
uint32_t vk_api_version;
VkSurfaceKHR vk_surface;

VkSwapchainKHR vk_swapchain;
std::vector<VkImage> vk_swapchain_images;
std::vector<VkImageView> vk_swapchain_image_views;

uint32_t vk_swapchain_resize_width;
uint32_t vk_swapchain_resize_height;
bool vk_swapchain_resize_require;

VkFormat vk_depth_buffer_format = VK_FORMAT_UNDEFINED;
VkImage vk_image_depth_buffer;
VmaAllocation vma_allocation_depth_buffer;
VkImageView vk_image_view_depth_buffer;

std::vector<VkFramebuffer> vk_framebuffers;

VkSemaphore vk_semaphore_image_available;
std::vector<VkSemaphore> vk_semaphores_image_finished;
VkFence vk_fence_frame_in_flight;

VkCommandPool vk_command_pool;
VkCommandBuffer vk_command_buffer;

VkDescriptorPool vk_imgui_descriptor_pool;
VkRenderPass vk_imgui_render_pass;
std::vector<VkFramebuffer> vk_imgui_framebuffers;
VkCommandPool vk_imgui_command_pool;
VkCommandBuffer vk_imgui_command_buffer;

VkFormat selectDepthFormat(VkPhysicalDevice physical_device) {
	// Prefer the original format and preserve stencil support in the fallback.
	const VkFormat candidates[] = {
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
	};

	for (VkFormat format : candidates) {
		VkFormatProperties properties{};
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
		if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			return format;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

bool initializeImGUI() {
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLER,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = uint32_t(vk_swapchain_images.size()),
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
							   &vk_imgui_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool for ImGUI rendering\n";
		return false;
	}

	// Создание отдельного рендер-паса для ImGUI
	const VkAttachmentDescription imgui_attachment = {
		.format = context.swapchain_format,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	};

	const VkAttachmentReference imgui_color_attachment = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription imgui_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &imgui_color_attachment,
	};

	const VkRenderPassCreateInfo imgui_render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &imgui_attachment,
		.subpassCount = 1,
		.pSubpasses = &imgui_subpass,
	};

	if (vkCreateRenderPass(context.device, &imgui_render_pass, nullptr, &vk_imgui_render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass for ImGUI\n";
		return false;
	}
	


	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr,
							&vk_imgui_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool for ImGUI rendering\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffer = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_imgui_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffer,
								 &vk_imgui_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command buffer for ImGUI rendering\n";
		return false;
	}

	ImGui_ImplVulkan_InitInfo init = {
		.ApiVersion = vk_api_version,
		.Instance = vk_instance,
		.PhysicalDevice = context.physical_device,
		.Device = context.device,
		.QueueFamily = context.graphics_queue_index,
		.Queue = context.graphics_queue,
		.DescriptorPool = vk_imgui_descriptor_pool,
		.MinImageCount = swapchain_images_count,
		.ImageCount = swapchain_images_count,
		.PipelineInfoMain = {
			.RenderPass = vk_imgui_render_pass,
		},
	};

	bool imgui_init_result = ImGui_ImplVulkan_Init(&init);
	if (!imgui_init_result) {
		std::cerr << "Failed to initialize ImGUI Vulkan backend\n";
	}
	
	return imgui_init_result;
}

void drawImGUI() {
	vkResetCommandBuffer(vk_imgui_command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(vk_imgui_command_buffer, &command_buffer_begin);

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = vk_imgui_render_pass,
		.framebuffer = vk_imgui_framebuffers[graphics::internal::vk_swapchain_current_image],
		.renderArea = { .extent = context.swapchain_extent },
	};

	vkCmdBeginRenderPass(vk_imgui_command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk_imgui_command_buffer);

	vkCmdEndRenderPass(vk_imgui_command_buffer);

	vkEndCommandBuffer(vk_imgui_command_buffer);
}

bool rebuildSwapchain(uint32_t width, uint32_t height) {
	vkQueueWaitIdle(context.graphics_queue);

	vkb::SwapchainBuilder sb(context.physical_device, context.device, vk_surface,
	                         context.graphics_queue_index, context.graphics_queue_index);

	auto sb_result = sb.set_desired_extent(width, height)
	                   .use_default_format_selection()
					   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
					   .use_default_image_usage_flags()
					   .set_old_swapchain(vk_swapchain)
					   .build();
	if (!sb_result) {
		std::cerr << sb_result.error().message() << '\n';
	}

	auto vkb_swapchain = sb_result.value();

	for (size_t i = 0, n = vk_swapchain_image_views.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}

	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vk_swapchain = vkb_swapchain.swapchain;
	context.swapchain_format = vkb_swapchain.image_format;
	context.swapchain_extent = vkb_swapchain.extent;

	auto swapchain_images = vkb_swapchain.get_images().value();
	auto swapchain_image_views = vkb_swapchain.get_image_views().value();

	vk_swapchain_images = std::move(swapchain_images);
	vk_swapchain_image_views = std::move(swapchain_image_views);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer,
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo imgui_framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		imgui_framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &imgui_framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	vk_swapchain_resize_require = false;

	return true;
}

void createOctahedronGeometry() {
    // Вершины правильного октаэдра
    graphics::internal::vertices = {
        {{ 1.0f,  0.0f,  0.0f}, {1.0f, 0.0f, 0.0f}},
        {{-1.0f,  0.0f,  0.0f}, {0.0f, 1.0f, 0.0f}},
        {{ 0.0f,  1.0f,  0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.0f, -1.0f,  0.0f}, {1.0f, 1.0f, 0.0f}},
        {{ 0.0f,  0.0f,  1.0f}, {1.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.0f, -1.0f}, {0.0f, 1.0f, 1.0f}}
    };
    
    // Индексы для 8 треугольников
    graphics::internal::indices = {
        2, 4, 0,  2, 0, 5,  2, 5, 1,  2, 1, 4,
        3, 0, 4,  3, 5, 0,  3, 1, 5,  3, 4, 1
    };
}

void createVertexBuffer() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(graphics::internal::vertices[0]) * graphics::internal::vertices.size();
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo,
                       &graphics::internal::vertexBuffer, &graphics::internal::vertexBufferAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }
    
    void* data;
    vmaMapMemory(context.allocator, graphics::internal::vertexBufferAllocation, &data);
    memcpy(data, graphics::internal::vertices.data(), (size_t)bufferInfo.size);
    vmaUnmapMemory(context.allocator, graphics::internal::vertexBufferAllocation);
}

void createIndexBuffer() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(graphics::internal::indices[0]) * graphics::internal::indices.size();
    bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo,
                       &graphics::internal::indexBuffer, &graphics::internal::indexBufferAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("failed to create index buffer!");
    }
    
    void* data;
    vmaMapMemory(context.allocator, graphics::internal::indexBufferAllocation, &data);
    memcpy(data, graphics::internal::indices.data(), (size_t)bufferInfo.size);
    vmaUnmapMemory(context.allocator, graphics::internal::indexBufferAllocation);
}

void createUniformBuffers() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(UniformBufferObject);
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    for (size_t i = 0; i < 3; i++) {
        if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo,
                           &graphics::internal::uniformBuffers[i], &graphics::internal::uniformBuffersAllocations[i], nullptr) != VK_SUCCESS) {
            throw std::runtime_error("failed to create uniform buffer!");
        }
        
        vmaMapMemory(context.allocator, graphics::internal::uniformBuffersAllocations[i], &graphics::internal::uniformBuffersMapped[i]);
    }
}

void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;
    
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &graphics::internal::descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(3);
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(3);
    
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &graphics::internal::descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(3, graphics::internal::descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = graphics::internal::descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocInfo.pSetLayouts = layouts.data();
    
    if (vkAllocateDescriptorSets(context.device, &allocInfo, graphics::internal::descriptorSets) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
    
    for (size_t i = 0; i < 3; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = graphics::internal::uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);
        
        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = graphics::internal::descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;
        
        vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

static VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    
    return shaderModule;
}

void createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/octahedron.vert.spv");
    auto fragShaderCode = readFile("shaders/octahedron.frag.spv");
    
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
    
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    // Настройки вершинного ввода
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
    // Настройки входной сборки
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    // Настройки вьюпорта и ножниц
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)context.swapchain_extent.width;
    viewport.height = (float)context.swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = context.swapchain_extent;
    
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    
    // Настройки растеризации
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    // Настройки мультисэмплинга
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Настройки смешивания цветов
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;
    
    // Настройки глубины
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    
    // layout пайплайн
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &graphics::internal::descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    
    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &graphics::internal::pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    
    // Графический пайплайн
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = graphics::internal::pipelineLayout;
    pipelineInfo.renderPass = context.render_pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    
    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphics::internal::graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    
    // Удаляем шейдерные модули
    vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
}

} // namespace

// Переменные рендеринга октаэдра
VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;
VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;

VkBuffer uniformBuffers[3];
VmaAllocation uniformBuffersAllocations[3];
void* uniformBuffersMapped[3];

VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
VkDescriptorSet descriptorSets[3];

VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

uint32_t vk_swapchain_current_image;

Context context;

bool initialize(GLFWwindow* const window) {
    // Выбор NVIDIA GPU 
    putenv((char*)"VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json");
    putenv((char*)"NODEVICE_SELECT=1");
    
    vkb::InstanceBuilder ib;

    auto ibr = ib.require_api_version(VK_MAKE_VERSION(1, 1, 0))
                 .request_validation_layers()
                 .build();

    auto vkb_instance = ibr.value();
    vk_instance = vkb_instance.instance;
    vk_api_version = vkb_instance.api_version;

    if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
        const char *message = nullptr;
        glfwGetError(&message);
        std::cerr << message << '\n';
        return false;
    }

    vkb::PhysicalDeviceSelector pds(vkb_instance, vk_surface);

    auto pds_result = pds.set_required_features({})
                         .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                         .require_present()
                         .select();
    
    if (!pds_result) {
        std::cerr << "Failed to select physical device: " << pds_result.error().message() << '\n';
        return false;
    }

    auto vkb_physical_device = pds_result.value();
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vkb_physical_device.physical_device, &props);
    std::cout << "Selected device: " << props.deviceName << "\n";
    


    vk_depth_buffer_format = selectDepthFormat(vkb_physical_device.physical_device);
    if (vk_depth_buffer_format == VK_FORMAT_UNDEFINED) {
        std::cerr << "No supported depth/stencil attachment format found\n";
        return false;
    }

    vkb::DeviceBuilder db(vkb_physical_device);

    auto db_result = db.build();
    if (!db_result) {
        std::cerr << db_result.error().message() << '\n';
        return false;
    }

    auto vkb_device = db_result.value();

    context.physical_device = vkb_device.physical_device;
    context.device = vkb_device.device;

    if (auto result = vkb_device.get_queue(vkb::QueueType::graphics); result) {
        context.graphics_queue = result.value();
    } else {
        std::cerr << result.error().message() << '\n';
        return false;
    }

    if (auto result = vkb_device.get_queue_index(vkb::QueueType::graphics); result) {
        context.graphics_queue_index = result.value();
    } else {
        std::cerr << result.error().message() << '\n';
        return false;
    }

    const VmaAllocatorCreateInfo allocator = {
        .physicalDevice = context.physical_device,
        .device = context.device,
        .instance = vk_instance,
        .vulkanApiVersion = vk_api_version,
    };

    if (vmaCreateAllocator(&allocator, &context.allocator) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan Memory Allocator\n";
        return false;
    }

    vkb::SwapchainBuilder sb(vkb_device);

    auto sb_result = sb.use_default_format_selection()
                       .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                       .use_default_image_usage_flags()
                       .build();
    if (!sb_result) {
        std::cerr << "Failed to create swapchain: " << sb_result.error().message() << '\n';
        return false;
    }

    auto vkb_swapchain = sb_result.value();

    vk_swapchain = vkb_swapchain.swapchain;
    context.swapchain_format = vkb_swapchain.image_format;
    context.swapchain_extent = vkb_swapchain.extent;
    vk_swapchain_images = vkb_swapchain.get_images().value();
    vk_swapchain_image_views = vkb_swapchain.get_image_views().value();
    vk_swapchain_current_image = UINT32_MAX;

    const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());
    


    // Создание буфера глубины
    const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const VkAttachmentDescription render_pass_attachments[] = {
		{
			.format = context.swapchain_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		},
		{
			.format = vk_depth_buffer_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		},
	};

	const VkAttachmentReference render_pass_color_attachment = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkAttachmentReference render_pass_depth_attachment = {
		.attachment = 1,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription render_pass_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &render_pass_color_attachment,
		.pDepthStencilAttachment = &render_pass_depth_attachment,
	};

	const VkRenderPassCreateInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = sizeof(render_pass_attachments) / sizeof(render_pass_attachments[0]),
		.pAttachments = render_pass_attachments,
		.subpassCount = 1,
		.pSubpasses = &render_pass_subpass,
	};

	if (vkCreateRenderPass(context.device, &render_pass, nullptr, &context.render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass\n";
		return false;
	}

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	const VkSemaphoreCreateInfo semaphore = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

	const VkFenceCreateInfo fence = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT,
	};

	vk_semaphores_image_finished.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		if (vkCreateSemaphore(context.device, &semaphore, nullptr,
							  &vk_semaphores_image_finished[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan semaphore #" << i << " for finished image\n";
			return false;
		}
	}

	if (vkCreateSemaphore(context.device, &semaphore, NULL,
						  &vk_semaphore_image_available) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for available image\n";
		return false;
	}

	if (vkCreateFence(context.device, &fence, nullptr, &vk_fence_frame_in_flight) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for image in flight\n";
		return false;
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffers = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffers, &vk_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan command buffer\n";
		return false;
	}

	if (!initializeImGUI()) {
		std::cerr << "Failed to initialize ImGUI Vulkan rendering backend\n";
		return false;
	}
    // Инициализация рендеринга октаэдра
    try {
        createOctahedronGeometry();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createGraphicsPipeline();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    
    return true;

}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	ImGui_ImplVulkan_Shutdown();

	vkDestroyCommandPool(context.device, vk_imgui_command_pool, nullptr);
	for (size_t i = 0, n = vk_imgui_framebuffers.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, vk_imgui_render_pass, nullptr);
	vkDestroyDescriptorPool(context.device, vk_imgui_descriptor_pool, nullptr);

	vkDestroyCommandPool(context.device, vk_command_pool, nullptr);

	vkDestroyFence(context.device, vk_fence_frame_in_flight, nullptr);
	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroySemaphore(context.device, vk_semaphores_image_finished[i], nullptr);
	}
	vkDestroySemaphore(context.device, vk_semaphore_image_available, nullptr);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, context.render_pass, nullptr);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}
	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vmaDestroyAllocator(context.allocator);

	vkDestroyDevice(context.device, nullptr);

	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkDestroyInstance(vk_instance, nullptr);
	
    // Очистка ресурсов октаэдра
    vkDestroyPipeline(context.device, graphics::internal::graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, graphics::internal::pipelineLayout, nullptr);
    vkDestroyDescriptorPool(context.device, graphics::internal::descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, graphics::internal::descriptorSetLayout, nullptr);
    
    for (size_t i = 0; i < 3; i++) {
        vmaUnmapMemory(context.allocator, graphics::internal::uniformBuffersAllocations[i]);
        vmaDestroyBuffer(context.allocator, graphics::internal::uniformBuffers[i], graphics::internal::uniformBuffersAllocations[i]);
    }
    
    vmaDestroyBuffer(context.allocator, graphics::internal::indexBuffer, graphics::internal::indexBufferAllocation);
    vmaDestroyBuffer(context.allocator, graphics::internal::vertexBuffer, graphics::internal::vertexBufferAllocation);

}

void resize(uint32_t width, uint32_t height) {
	if (width == 0 || height == 0) {
		return;
	}

	vk_swapchain_resize_width = width;
	vk_swapchain_resize_height = height;

	vk_swapchain_resize_require = true;
}

FrameData prepare() {
	vkWaitForFences(context.device, 1, &vk_fence_frame_in_flight, VK_TRUE, UINT64_MAX);

retry_acquire:
	switch (vkAcquireNextImageKHR(context.device, vk_swapchain, UINT64_MAX,
								  vk_semaphore_image_available, VK_NULL_HANDLE,
								  &graphics::internal::vk_swapchain_current_image)) {
	case VK_SUCCESS:
		break;

	case VK_ERROR_OUT_OF_DATE_KHR:
		std::cout << "Swapchain out of date in acquire, rebuilding...\n";
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
		goto retry_acquire;

	case VK_SUBOPTIMAL_KHR:
		std::cout << "Swapchain is suboptimal for rendering!\n";
		break;

	default:
		std::cerr << "Failed to acquire next image.\n";
		return {};
	}

	vkResetFences(context.device, 1, &vk_fence_frame_in_flight);

	vkResetCommandBuffer(vk_command_buffer, 0);

	VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(vk_command_buffer, &begin_info);

	return {
		.framebuffer = vk_framebuffers[graphics::internal::vk_swapchain_current_image],
		.command_buffer = vk_command_buffer,
	};
}

void submitAndPresent() {

	
	drawImGUI();

	const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	const VkCommandBuffer command_buffers[] = {
		vk_command_buffer,
		vk_imgui_command_buffer,
	};

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphore_image_available,
		.pWaitDstStageMask = &stage,
		.commandBufferCount = sizeof(command_buffers) / sizeof(command_buffers[0]),
		.pCommandBuffers = command_buffers,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &vk_semaphores_image_finished[graphics::internal::vk_swapchain_current_image],
	};

	VkResult submit_result = vkQueueSubmit(context.graphics_queue, 1, &submit, vk_fence_frame_in_flight);
	if (submit_result != VK_SUCCESS) {
		std::cerr << "Failed to submit command buffers. Error code: " << submit_result << "\n";
	}

	const VkPresentInfoKHR present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphores_image_finished[graphics::internal::vk_swapchain_current_image],
		.swapchainCount = 1,
		.pSwapchains = &vk_swapchain,
		.pImageIndices = &graphics::internal::vk_swapchain_current_image,
	};

	VkResult result = vkQueuePresentKHR(context.graphics_queue, &present);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		std::cout << "Swapchain out of date, rebuilding...\n";
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
	} else if (result == VK_SUBOPTIMAL_KHR) {
		std::cout << "Swapchain suboptimal, continuing...\n";
	} else if (vk_swapchain_resize_require) {
		std::cout << "Swapchain resize required, rebuilding...\n";
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
	} else if (result != VK_SUCCESS) {
		std::cerr << "Failed to present Vulkan swapchain image. Error code: " << result << "\n";
	}
}

} // namespace graphics::internal



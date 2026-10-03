#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include "graphics_internal.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

bool initialize() {
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
}

void update([[maybe_unused]] double time) {
    ImGui::Begin("Octahedron Controls");
    
    ImGui::Text("Perspective Projection (active)");
    ImGui::Text("Orthographic projection - coming soon");
    
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    
    ImGui::End();
    
    ImGui::ShowDemoWindow();
}


void render(const graphics::internal::FrameData& fd) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    
    // Обновление uniform буфера
    graphics::internal::UniformBufferObject ubo{};
    
    // Модельная матрица - вращение
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Видовая матрица - камера смотрит на объект
    glm::vec3 cameraPos = glm::vec3(2.0f, 2.0f, 2.0f);
    
    ubo.view = glm::lookAt(cameraPos,
                          glm::vec3(0.0f, 0.0f, 0.0f),  // Смотрим на центр
                          glm::vec3(0.0f, 0.0f, 1.0f)); // Вверх по оси Z
    
    float aspectRatio = graphics::internal::context.swapchain_extent.width / 
                       (float)graphics::internal::context.swapchain_extent.height;
    
    ubo.proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1; // Vulkan использует инвертированную Y-координату
    
    // Копируем данные в uniform буфер
    memcpy(graphics::internal::uniformBuffersMapped[graphics::internal::vk_swapchain_current_image], &ubo, sizeof(ubo));
    
    VkClearValue clear_values[2] = {};
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};
    
    VkRenderPassBeginInfo render_pass_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = graphics::internal::context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {
            .offset = {0, 0},
            .extent = graphics::internal::context.swapchain_extent,
        },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    
    vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
    
    // Отрисовываем октаэдр
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics::internal::graphicsPipeline);
    
    VkBuffer vertexBuffers[] = {graphics::internal::vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, graphics::internal::indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                           graphics::internal::pipelineLayout, 0, 1, 
                           &graphics::internal::descriptorSets[graphics::internal::vk_swapchain_current_image], 0, nullptr);
    
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(graphics::internal::indices.size()), 1, 0, 0, 0);
    
    vkCmdEndRenderPass(fd.command_buffer);
    
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
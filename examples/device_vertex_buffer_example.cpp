/*

MIT License

Copyright (c) 2022 Benjamin Frech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <iostream>
#include <fstream>

#include <VkBootstrap.h>
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "vku.h"
#include <glm/glm.hpp>

const int WINDOW_WIDTH = 1440;
const int WINDOW_HEIGHT = 1080;
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::string EXAMPLE_SHADER_DIRECTORY = "resources/shaders";


struct Vertex
{
	glm::vec3 position{};
	glm::vec3 color{};

	static VkVertexInputBindingDescription get_binding_description()
	{
		VkVertexInputBindingDescription bindingDescription{};

		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::vector<VkVertexInputAttributeDescription> get_attribute_descriptions() {
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return std::move(attributeDescriptions);
	}
};


const std::vector<Vertex> vertices{
	{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}
};


inline std::vector<char> read_file(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.good())
	{
		return {};
	}

	size_t file_size = (size_t)file.tellg();
	std::vector<char> buffer(file_size);

	file.seekg(0);
	file.read(buffer.data(), static_cast<std::streamsize> (file_size));

	file.close();

	return buffer;
}


class ExampleApplication
{
public:
	inline bool create_window()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "vku Examples", nullptr, nullptr);

		return nullptr != window;
	}

	inline void destroy_window()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	inline bool create_instance()
	{
		vkb::InstanceBuilder instance_builder;

		// TODO: debug flag around this
		instance_builder
			.request_validation_layers()
			.use_default_debug_messenger()
			.set_app_name("VKU Example Application")
			.set_engine_name("No Engine")
			.require_api_version(1, 0, 0);

		auto instance_builder_return = instance_builder.build();

		if (!instance_builder_return)
		{
			std::cerr << "Failed to create Vulkan instance. Error: " << instance_builder_return.error().message() << "\n";
			return false;
		}

		vkb_instance = instance_builder_return.value();

		return true;
	}

	inline void destroy_instance()
	{
		vkb::destroy_instance(vkb_instance);
	}

	inline bool create_surface()
	{
		VkResult err = glfwCreateWindowSurface(vkb_instance, window, NULL, &surface);
		return err == VK_SUCCESS;
	}

	inline void destroy_surface()
	{
		vkDestroySurfaceKHR(vkb_instance, surface, nullptr);
	}

	inline bool select_physical_device()
	{
		vkb::PhysicalDeviceSelector phys_device_selector(vkb_instance);

		auto physical_device_selector_return = phys_device_selector
			.set_surface(surface)
			.select();

		if (!physical_device_selector_return) {
			// Handle error
			return false;
		}

		vkb_physical_device = physical_device_selector_return.value();

		return true;
	}

	inline bool create_device()
	{
		vkb::DeviceBuilder device_builder{ vkb_physical_device };
		auto dev_ret = device_builder.build();
		if (!dev_ret) {
			// error
			std::cerr << "failed to create device" << std::endl;
			return false;
		}
		vkb_device = dev_ret.value();

		return true;
	}

	inline void destroy_device()
	{
		vkb::destroy_device(vkb_device);
	}

	inline bool get_queues()
	{
		auto queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
		if (!queue_ret)
		{
			// handle error
			return false;
		}
		graphics_queue = queue_ret.value();

		auto pq = vkb_device.get_queue(vkb::QueueType::present);
		if (!pq.has_value())
		{
			std::cout << "failed to get present queue: " << pq.error().message() << "\n";
			return false;
		}
		present_queue = pq.value();

		return true;
	}

	inline bool create_swapchain()
	{
		vkb::SwapchainBuilder swapchain_builder{ vkb_device };
		swapchain_builder.set_old_swapchain(vkb_swapchain);
		auto swap_ret = swapchain_builder.build();
		if (!swap_ret)
		{
			return false;
		}
		vkb::destroy_swapchain(vkb_swapchain);
		vkb_swapchain = swap_ret.value();
		return true;
	}

	inline void destroy_swapchain()
	{
		vkb::destroy_swapchain(vkb_swapchain);
	}

	inline bool create_render_pass()
	{
		VkAttachmentDescription color_attachment = {};
		color_attachment.format = vkb_swapchain.image_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;

		if (vkCreateRenderPass(vkb_device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS)
		{
			std::cout << "failed to create render pass\n";
			return false; // failed to create render pass!
		}

		return true;
	}

	inline void destroy_render_pass()
	{
		vkDestroyRenderPass(vkb_device, render_pass, nullptr);
	}

	inline bool create_framebuffers()
	{
		// TODO: unwrap/check for errors
		swapchain_images = vkb_swapchain.get_images().value();
		swapchain_image_views = vkb_swapchain.get_image_views().value();

		framebuffers.resize(swapchain_image_views.size());

		for (size_t i = 0; i < swapchain_image_views.size(); i++) {
			VkImageView attachments[] = { swapchain_image_views[i] };

			VkFramebufferCreateInfo framebuffer_info = {};
			framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer_info.renderPass = render_pass;
			framebuffer_info.attachmentCount = 1;
			framebuffer_info.pAttachments = attachments;
			framebuffer_info.width = vkb_swapchain.extent.width;
			framebuffer_info.height = vkb_swapchain.extent.height;
			framebuffer_info.layers = 1;

			if (vkCreateFramebuffer(vkb_device, &framebuffer_info, nullptr, &framebuffers[i]) != VK_SUCCESS)
			{
				return false; // failed to create framebuffer
			}
		}
		return true;
	}

	inline void destroy_framebuffers()
	{
		for (size_t i = 0; i < swapchain_image_views.size(); ++i)
		{
			vkDestroyFramebuffer(vkb_device, framebuffers[i], nullptr);
		}
		framebuffers.clear();

		vkb_swapchain.destroy_image_views(swapchain_image_views);
		swapchain_image_views.clear();	
	}

	inline bool create_command_pool()
	{
		VkCommandPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

		// TODO: error handling
		pool_info.queueFamilyIndex = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

		if (vkCreateCommandPool(vkb_device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
		{
			std::cout << "failed to create command pool\n";
			return false; // failed to create command pool
		}
		return true;
	}

	inline void destroy_command_pool()
	{
		vkDestroyCommandPool(vkb_device, command_pool, nullptr);
	}

	inline VkShaderModule create_shader_module(const std::vector<char> &code)
	{
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*> (code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(vkb_device, &create_info, nullptr, &shaderModule) != VK_SUCCESS)
		{
			return VK_NULL_HANDLE; // failed to create shader module
		}

		return shaderModule;
	}

	inline bool create_graphics_pipeline()
	{
		// create vertex shader
		std::vector<char> vert_code = read_file(std::string(EXAMPLE_SHADER_DIRECTORY) + "/triangle_buffer.vert.spv");
		if (vert_code.empty())
		{
			std::cout << "failed to load vertex shader\n";
			return false; // failed to create shader modules
		}

		auto vert_module_result = vku::create_shader_module(vkb_device, vert_code);
		if (!vert_module_result)
		{
			std::cout << "failed to create shader module\n";
			return false; // failed to create shader modules
		}
		vku::ShaderModule vert_module = vert_module_result.get_value();


		// create fragment shader
		std::vector<char> frag_code = read_file(std::string(EXAMPLE_SHADER_DIRECTORY) + "/triangle_buffer.frag.spv");
		if (frag_code.empty())
		{
			std::cout << "failed to load fragment shader\n";
			return false; // failed to create shader modules
		}

		auto frag_module_result = vku::create_shader_module(vkb_device, frag_code);
		if (!frag_module_result)
		{
			std::cout << "failed to create shader module\n";
			return false; // failed to create shader modules
		}
		vku::ShaderModule frag_module = frag_module_result.get_value();


		// create pipeline layout
		vku::PipelineLayoutBuilder pipeline_layout_builder(vkb_device);
		auto pipeline_layout_result = pipeline_layout_builder.build();
		if (!pipeline_layout_result)
		{
			return false;
		}
		pipeline_layout = pipeline_layout_result.get_value();


		vku::GraphicsPipelineBuilder pipeline_builder(vkb_device);

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		pipeline_builder
			.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vert_module)
			.add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module)
		    // .add_viewport(0.0f, 0.0f, (float)vkb_swapchain.extent.width, (float)vkb_swapchain.extent.height, 0.0f, 1.0f)
		    // .add_scissor(0, 0, vkb_swapchain.extent.width, vkb_swapchain.extent.height)
			.set_viewport_count(1)
			.set_scissor_count(1)
			.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
			.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR)
			.set_pipeline_layout(pipeline_layout)
			.set_render_pass(render_pass)
			.add_color_blend_attachment(colorBlendAttachment)
			.add_vertex_binding(Vertex::get_binding_description())
			.add_vertex_attributes(Vertex::get_attribute_descriptions());

		auto pipeline_result = pipeline_builder.build();
		if (!pipeline_result)
		{
			return false;
		}
		graphics_pipeline = pipeline_result.get_value();

		return true;
	}

	inline void destroy_graphics_pipeline()
	{
		graphics_pipeline.destroy();
		pipeline_layout.destroy();
	}

	/* 
		This waits for the queue to drain then deletes the command buffer in the call to SingleTimeCommandExecutor::execute. It's less code, but performance suffers.
	*/
	inline bool create_vertex_buffer()
	{
		vku::BufferFactory factory(vkb_device, vkb_physical_device);

		auto staging_buffer_result = factory.build(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!staging_buffer_result)
		{
			std::cout << "failed to create staging buffer" << std::endl;
			return false;
		}

		// this will be destroyed when it goes out of scope at the end of the functin.
		auto staging_buffer = staging_buffer_result.get_value();

		auto vb_result = factory.build(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!vb_result)
		{
			std::cout << "failed to create vertex buffer" << std::endl;
			return false;
		}

		vertex_buffer = vb_result.get_value();

		VkResult map_result = staging_buffer.map_memory();

		if (map_result != VK_SUCCESS)
		{
			std::cout << "error mapping staging buffer" << std::endl;
			return false;
		}

		memcpy(staging_buffer.get_mapped(), vertices.data(), staging_buffer.get_size());

		staging_buffer.unmap_memory();

		// copy the staging buffer to the vertex buffer.
		vku::SingleTimeCommandExecutor executor(vkb_device, command_pool, graphics_queue);

		vku::Error error{};
		bool execute_result = executor.execute([&](VkCommandBuffer commandBuffer) -> bool {
			VkBufferCopy copyRegion{};
			copyRegion.size = staging_buffer.get_size();
			vkCmdCopyBuffer(commandBuffer, staging_buffer, vertex_buffer, 1, &copyRegion);

			return true;
		});

		if (!execute_result)
		{
			// TODO: report contents of error.
			return false;
		}

		// because we didn't pass in a fence, execute_result should contain a VK_NULL_HANDLE result.

		return true;
	}

	/*
		This creates a fence and passes it to SingleTimeCommandExecutor::execute. This allows you to perform other operations while the transfer is happening, then wait
		for the fence before using the transfered data.
	*/
	inline bool create_vertex_buffer_fence()
	{
		vku::BufferFactory factory(vkb_device, vkb_physical_device);

		auto staging_buffer_result = factory.build(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!staging_buffer_result)
		{
			std::cout << "failed to create staging buffer" << std::endl;
			return false;
		}

		// this will be destroyed when it goes out of scope at the end of the functin.
		staging_buffer = staging_buffer_result.get_value();

		auto vb_result = factory.build(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!vb_result)
		{
			std::cout << "failed to create vertex buffer" << std::endl;
			return false;
		}

		vertex_buffer = vb_result.get_value();

		VkResult map_result = staging_buffer.map_memory();

		if (map_result != VK_SUCCESS)
		{
			std::cout << "error mapping staging buffer" << std::endl;
			return false;
		}

		memcpy(staging_buffer.get_mapped(), vertices.data(), staging_buffer.get_size());

		staging_buffer.unmap_memory();

		// copy the staging buffer to the vertex buffer.
		vku::SingleTimeCommandExecutor executor(vkb_device, command_pool, graphics_queue);

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		vkCreateFence(vkb_device, &fence_create_info, nullptr, &vertex_buffer_fence);

		vku::Error error{};
		auto execute_result = executor.execute([&](VkCommandBuffer commandBuffer) -> bool {
			VkBufferCopy copyRegion{};
			copyRegion.size = staging_buffer.get_size();
			vkCmdCopyBuffer(commandBuffer, staging_buffer, vertex_buffer, 1, &copyRegion);

			return true;
		}, vertex_buffer_fence);

		if (!execute_result)
		{
			// TODO: report contents of error.
			return false;
		}

		// because we DID pass in a fence, execute_result contains a command buffer, and we need to delete it after the fence is signalled.
		vertex_command_buffer = execute_result.get_value();

		return true;
	}

	inline void destroy_vertex_buffer()
	{
		vertex_buffer.destroy();
	}

	inline bool create_command_buffers()
	{
		command_buffers.resize(framebuffers.size());

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = command_pool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

		if (vkAllocateCommandBuffers(vkb_device, &allocInfo, command_buffers.data()) != VK_SUCCESS)
		{
			return false; // failed to allocate command buffers;
		}

		for (size_t i = 0; i < command_buffers.size(); i++)
		{
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			if (vkBeginCommandBuffer(command_buffers[i], &begin_info) != VK_SUCCESS)
			{
				return false; // failed to begin recording command buffer
			}

			VkRenderPassBeginInfo render_pass_info = {};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			render_pass_info.renderPass = render_pass;
			render_pass_info.framebuffer = framebuffers[i];
			render_pass_info.renderArea.offset = { 0, 0 };
			render_pass_info.renderArea.extent = vkb_swapchain.extent;
			VkClearValue clearColor{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };
			render_pass_info.clearValueCount = 1;
			render_pass_info.pClearValues = &clearColor;

			VkViewport viewport = {};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)vkb_swapchain.extent.width;
			viewport.height = (float)vkb_swapchain.extent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {};
			scissor.offset = { 0, 0 };
			scissor.extent = vkb_swapchain.extent;

			vkCmdSetViewport(command_buffers[i], 0, 1, &viewport);
			vkCmdSetScissor(command_buffers[i], 0, 1, &scissor);

			vkCmdBeginRenderPass(command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

			VkBuffer vertexBuffers[] = { vertex_buffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertexBuffers, offsets);

			vkCmdDraw(command_buffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);

			// vkCmdDraw(command_buffers[i], 3, 1, 0, 0);

			vkCmdEndRenderPass(command_buffers[i]);

			if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS)
			{
				std::cout << "failed to record command buffer\n";
				return false; // failed to record command buffer!
			}
		}
		return true;
	}

	inline bool create_sync_objects() 
	{
		available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		finished_semaphore.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
		image_in_flight.resize(vkb_swapchain.image_count, VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphore_info = {};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info = {};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(vkb_device, &semaphore_info, nullptr, &available_semaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(vkb_device, &semaphore_info, nullptr, &finished_semaphore[i]) != VK_SUCCESS ||
				vkCreateFence(vkb_device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
				std::cout << "failed to create sync objects\n";
				return false; // failed to create synchronization objects for a frame
			}
		}
		return true;
	}

	inline void destroy_sync_objects()
	{
		for (const auto& semaphore : available_semaphores)
		{
			vkDestroySemaphore(vkb_device, semaphore, nullptr);
		}

		for (const auto& semaphore : finished_semaphore)
		{
			vkDestroySemaphore(vkb_device, semaphore, nullptr);
		}

		for (const auto& fence : in_flight_fences)
		{
			vkDestroyFence(vkb_device, fence, nullptr);
		}
	}

	inline bool recreate_swapchain()
	{
		vkDeviceWaitIdle(vkb_device);

		destroy_command_pool();
		destroy_framebuffers();

		if (!create_swapchain())
		{
			return false;
		}

		if (!create_framebuffers())
		{
			return false;
		}

		if (!create_command_pool())
		{
			return false;
		}

		if (!create_command_buffers())
		{
			return false;
		}

		return true;
	}

	inline bool create()
	{
		// TODO: we should be returning error codes to indicate where the error happened for reporting elsewhere.
		if (!create_window())
		{
			return false;
		}

		if (!create_instance())
		{
			return false;
		}

		if (!create_surface())
		{
			return false;
		}

		if (!select_physical_device())
		{
			return false;
		}

		if (!create_device())
		{
			return false;
		}

		if (!get_queues())
		{
			return false;
		}

		if (!create_swapchain())
		{
			return false;
		}

		if (!create_render_pass())
		{
			return false;
		}

		if (!create_framebuffers())
		{
			return false;
		}

		if (!create_sync_objects())
		{
			return false;
		}

		if (!create_graphics_pipeline())
		{
			return false;
		}

		if (!create_command_pool())
		{
			return false;
		}

		if (!create_vertex_buffer_fence())
		{
			return false;
		}

		// you could do other transfer operations here, then wait for the fence.

		if (vertex_buffer_fence != VK_NULL_HANDLE)
		{
			vkWaitForFences(vkb_device, 1, &vertex_buffer_fence, true, UINT64_MAX);
			vkDestroyFence(vkb_device, vertex_buffer_fence, nullptr);
			vkFreeCommandBuffers(vkb_device, command_pool, 1, &vertex_command_buffer);
			staging_buffer.destroy(); // we need to make sure the staging buffer remains live as well.
		}

		if (!create_command_buffers())
		{
			return false;
		}

		return true;
	}

	inline void destroy()
	{
		vkDeviceWaitIdle(vkb_device);

		destroy_vertex_buffer();
		destroy_graphics_pipeline();
		destroy_command_pool();
		destroy_framebuffers();
		destroy_sync_objects();
		destroy_render_pass();
		destroy_swapchain();
		destroy_device();
		destroy_surface();
		destroy_instance();
		destroy_window();
	}

	inline bool draw_frame()
	{
		vkWaitForFences(vkb_device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index = 0;
		VkResult result = vkAcquireNextImageKHR(vkb_device,
			vkb_swapchain,
			UINT64_MAX,
			available_semaphores[current_frame],
			VK_NULL_HANDLE,
			&image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			return recreate_swapchain();
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			std::cout << "failed to acquire swapchain image. Error " << result << "\n";
			return false;
		}

		if (image_in_flight[image_index] != VK_NULL_HANDLE)
		{
			vkWaitForFences(vkb_device, 1, &image_in_flight[image_index], VK_TRUE, UINT64_MAX);
		}
		image_in_flight[image_index] = in_flight_fences[current_frame];

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore wait_semaphores[] = { available_semaphores[current_frame] };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = wait_semaphores;
		submitInfo.pWaitDstStageMask = wait_stages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &command_buffers[image_index];

		VkSemaphore signal_semaphores[] = { finished_semaphore[current_frame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signal_semaphores;

		vkResetFences(vkb_device, 1, &in_flight_fences[current_frame]);

		if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS)
		{
			std::cout << "failed to submit draw command buffer\n";
			return false; //"failed to submit draw command buffer
		}

		VkPresentInfoKHR present_info = {};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		VkSwapchainKHR swapChains[] = { vkb_swapchain };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapChains;

		present_info.pImageIndices = &image_index;

		result = vkQueuePresentKHR(present_queue, &present_info);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		{
			return recreate_swapchain();
		}
		else if (result != VK_SUCCESS)
		{
			std::cout << "failed to present swapchain image\n";
			return false;
		}

		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
		return true;
	}

	inline bool main_loop()
	{
		// TODO: assertions to make sure everything is initialized.
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();

			if (!draw_frame())
			{
				std::cout << "failed to draw frame \n";
				return false;
			}
		}

		return true;
	}

private:
	GLFWwindow* window{ nullptr };
	vkb::Instance vkb_instance{ };
	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	vkb::PhysicalDevice vkb_physical_device{};
	vkb::Device vkb_device{};
	VkQueue graphics_queue{ VK_NULL_HANDLE };
	VkQueue present_queue{ VK_NULL_HANDLE };
	vkb::Swapchain vkb_swapchain{};
	VkRenderPass render_pass{};
	std::vector<VkImage> swapchain_images{};
	std::vector<VkImageView> swapchain_image_views{};
	std::vector<VkFramebuffer> framebuffers{};

	// TODO: initialize these
	VkCommandPool command_pool{ VK_NULL_HANDLE  };
	std::vector<VkCommandBuffer> command_buffers{};

	std::vector<VkSemaphore> available_semaphores{};
	std::vector<VkSemaphore> finished_semaphore{};
	std::vector<VkFence> in_flight_fences{};
	std::vector<VkFence> image_in_flight{};
	size_t current_frame{ 0 };

	vku::PipelineLayout  pipeline_layout{};
	vku::Pipeline graphics_pipeline{};

	vku::Buffer vertex_buffer{};
	vku::Buffer staging_buffer{};
	VkFence vertex_buffer_fence{ VK_NULL_HANDLE };
	VkCommandBuffer vertex_command_buffer{ VK_NULL_HANDLE };
};

int main()
{
	ExampleApplication example_application{};

	if (!example_application.create())
	{
		std::cerr << "failed to create example application" << std::endl;
	}
	else
	{
		example_application.main_loop();
	}
	
	example_application.destroy();

	return 0;
}

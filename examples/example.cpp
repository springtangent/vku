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

const int WINDOW_WIDTH = 1440;
const int WINDOW_HEIGHT = 1080;
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::string EXAMPLE_SHADER_DIRECTORY = "resources/shaders";


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
		// glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
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
		std::vector<char> vert_code = read_file(std::string(EXAMPLE_SHADER_DIRECTORY) + "/triangle.vert.spv");
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
		std::vector<char> frag_code = read_file(std::string(EXAMPLE_SHADER_DIRECTORY) + "/triangle.frag.spv");
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


		// create pipeline
		VkPipelineShaderStageCreateInfo vert_stage_info = {};
		vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_stage_info.module = vert_module;
		vert_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_stage_info = {};
		frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_stage_info.module = frag_module;
		frag_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage_info, frag_stage_info };

		VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 0;
		vertex_input_info.vertexAttributeDescriptionCount = 0;

		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

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

		VkPipelineViewportStateCreateInfo viewport_state = {};
		viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo color_blending = {};
		color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blending.logicOpEnable = VK_FALSE;
		color_blending.logicOp = VK_LOGIC_OP_COPY;
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &colorBlendAttachment;
		color_blending.blendConstants[0] = 0.0f;
		color_blending.blendConstants[1] = 0.0f;
		color_blending.blendConstants[2] = 0.0f;
		color_blending.blendConstants[3] = 0.0f;

		std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamic_info = {};
		dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_info.dynamicStateCount = static_cast<uint32_t> (dynamic_states.size());
		dynamic_info.pDynamicStates = dynamic_states.data();

		VkGraphicsPipelineCreateInfo pipeline_info = {};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.pDynamicState = &dynamic_info;
		pipeline_info.layout = pipeline_layout;
		pipeline_info.renderPass = render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(
			vkb_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
			std::cout << "failed to create pipline\n";
			return false; // failed to create graphics pipeline
		}

		return true;
	}

	inline void destroy_graphics_pipeline()
	{
		vkDestroyPipeline(vkb_device, graphics_pipeline, nullptr);
		pipeline_layout.destroy();
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

			vkCmdDraw(command_buffers[i], 3, 1, 0, 0);

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

		if (!create_command_buffers())
		{
			return false;
		}

		return true;
	}

	inline void destroy()
	{
		vkDeviceWaitIdle(vkb_device);

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
	VkPipeline graphics_pipeline{ VK_NULL_HANDLE };
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

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

#pragma once

#include <array>
#include <string>

const std::string EXAMPLE_SHADER_DIRECTORY = "resources/shaders/";
const int WINDOW_WIDTH = 1440;
const int WINDOW_HEIGHT = 1080;
const int MAX_FRAMES_IN_FLIGHT = 2;


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


inline vku::ShaderModule read_example_shader(VkDevice device, const std::string& filename)
{
	vku::ShaderModule result{};
	auto file_path = std::string(EXAMPLE_SHADER_DIRECTORY) + filename;
	std::cout << "loading: " << file_path << std::endl;
	// create vertex shader
	std::vector<char> code = read_file(file_path);
	if (code.empty())
	{
		std::cerr << "failed to load shader: " << filename << std::endl;;
		return result; // failed to create shader modules
	}

	auto shader_module_result = vku::create_shader_module(device, code);
	if (!shader_module_result)
	{
		std::cerr << "failed to create shader module for file: " << filename << std::endl;
		return result; // failed to create shader modules
	}
	result = shader_module_result.get_value();

	return result;
}


class ExampleApplication
{
public:
	ExampleApplication() = default;
	virtual ~ExampleApplication() { }

	virtual bool record_command_buffer_render_pass(uint32_t i)
	{
		return true;
	}

	virtual bool record_command_buffer(uint32_t image_index)
	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		auto command_buffer = command_buffers[current_frame];

		if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
		{
			return false; // failed to begin recording command buffer
		}

		VkRenderPassBeginInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_info.renderPass = render_pass;
		render_pass_info.framebuffer = framebuffers[image_index];
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

		vkCmdSetViewport(command_buffer, 0, 1, &viewport);
		vkCmdSetScissor(command_buffer, 0, 1, &scissor);

		vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		if (!record_command_buffer_render_pass(current_frame))
		{
			return false;
		}

		vkCmdEndRenderPass(command_buffer);

		if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
		{
			std::cout << "failed to record command buffer\n";
			return false; // failed to record command buffer!
		}

		return true;
	}

	inline bool create_command_buffers()
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = command_pool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

		if (vkAllocateCommandBuffers(vkb_device, &allocInfo, command_buffers.data()) != VK_SUCCESS)
		{
			std::cerr << "failed to allocate command buffers" << std::endl;
			return false; // failed to allocate command buffers;
		}

		return true;
	}

	inline vku::ShaderModule read_example_shader(const std::string& filename)
	{
		return std::move(::read_example_shader(vkb_device, filename));
	}

	inline void set_framebuffer_resized(bool value)
	{
		framebuffer_resized = value;
	}

	inline static void framebuffer_resized_callback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<ExampleApplication*>(glfwGetWindowUserPointer(window));
		app->set_framebuffer_resized(true);
	}

	inline bool create_window()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "vku Examples", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebuffer_resized_callback);

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

		// TODO: debug flag around validation layers/debug messenger
		// TODO: customizable application name.
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
		if (!dev_ret)
		{
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
			std::cerr << "failed to get graphics queue" << queue_ret.error().message() << "\n";
			// handle error
			return false;
		}
		graphics_queue = queue_ret.value();

		auto pq = vkb_device.get_queue(vkb::QueueType::present);
		if (!pq.has_value())
		{
			std::cerr << "failed to get present queue: " << pq.error().message() << "\n";
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
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

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

	inline bool create_sync_objects()
	{
		VkSemaphoreCreateInfo semaphore_info = {};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info = {};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(vkb_device, &semaphore_info, nullptr, &available_semaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(vkb_device, &semaphore_info, nullptr, &finished_semaphore[i]) != VK_SUCCESS ||
				vkCreateFence(vkb_device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
				std::cerr << "failed to create sync objects\n";
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

	virtual bool update_frame_data(uint32_t current_frame)
	{
		return true;
	}

	inline bool draw_frame()
	{
		vkWaitForFences(vkb_device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(vkb_device, vkb_swapchain, UINT64_MAX, available_semaphores[current_frame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreate_swapchain();
			return true;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		if (!update_frame_data(current_frame))
		{
			return false;
		}

		vkResetFences(vkb_device, 1, &in_flight_fences[current_frame]);

		vkResetCommandBuffer(command_buffers[current_frame], /*VkCommandBufferResetFlagBits*/ 0);
		record_command_buffer(imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { available_semaphores[current_frame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &command_buffers[current_frame];

		VkSemaphore signalSemaphores[] = { finished_semaphore[current_frame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS)
		{
			// TODO: print to cerr, return false.
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { vkb_swapchain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(present_queue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
		{
			framebuffer_resized = false;
			recreate_swapchain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
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
				std::cerr << "failed to draw frame \n";
				return false;
			}
		}

		return true;
	}

	virtual bool on_create()
	{
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

		if (!create_command_pool())
		{
			return false;
		}

		if (!on_create())
		{
			return false;
		}

		if (!create_command_buffers())
		{
			return false;
		}

		return true;
	}

	virtual void on_destroy()
	{
	}

	inline void destroy()
	{
		vkDeviceWaitIdle(vkb_device);

		on_destroy();

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
protected:
	GLFWwindow* window{ nullptr };

	vkb::Instance vkb_instance{ };
	vkb::PhysicalDevice vkb_physical_device{};
	vkb::Device vkb_device{};

	VkQueue graphics_queue{ VK_NULL_HANDLE };
	VkQueue present_queue{ VK_NULL_HANDLE };

	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	vkb::Swapchain vkb_swapchain{};
	std::vector<VkImage> swapchain_images{};
	std::vector<VkImageView> swapchain_image_views{};
	std::vector<VkFramebuffer> framebuffers{};

	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> available_semaphores{};
	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> finished_semaphore{};
	std::array<VkFence, MAX_FRAMES_IN_FLIGHT> in_flight_fences{};

	VkRenderPass render_pass{};

	VkCommandPool command_pool{ VK_NULL_HANDLE };
	std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers{};
	size_t current_frame{ 0 };
	bool framebuffer_resized = false;
};

template<typename A>
inline int run_application()
{
	A application{};

	if (!application.create())
	{
		std::cerr << "failed to create example application" << std::endl;
	}
	else
	{
		application.main_loop();
	}

	application.destroy();

	return EXIT_SUCCESS;
}

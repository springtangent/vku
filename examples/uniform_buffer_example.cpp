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

/*
	TODO: SharedContext / FrameData aren't great names
*/

#include <iostream>
#include <fstream>
#include <chrono>

#include <VkBootstrap.h>
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "vku.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "example_application.h"


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

	static std::vector<VkVertexInputAttributeDescription> get_attribute_descriptions()
	{
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


struct UniformData
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};


const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}}
};


const std::vector<uint16_t> indices = {
	0, 1, 2, 0, 2, 3
};


class SharedContext
{
public:
	VkDevice device{ VK_NULL_HANDLE };
	VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
	VkRenderPass render_pass{ VK_NULL_HANDLE };
	VkCommandPool command_pool{ VK_NULL_HANDLE };
	VkQueue graphics_queue{ VK_NULL_HANDLE };

	SharedContext() { }
	~SharedContext()
	{
		destroy();
	}

	inline void destroy()
	{
		staging_buffer.destroy();
		vertex_buffer.destroy();
		index_buffer.destroy();
		graphics_pipeline.destroy();
		graphics_pipeline_layout.destroy();
		destroy_descriptor_set_layouts();

		device = VK_NULL_HANDLE;
		physical_device = VK_NULL_HANDLE;
		render_pass = VK_NULL_HANDLE;
		command_pool = VK_NULL_HANDLE;
		graphics_queue = VK_NULL_HANDLE;

	}
	
	inline bool init(VkDevice d, VkPhysicalDevice pd, VkRenderPass rp, VkCommandPool cp, VkQueue gq)
	{
		device = d;
		physical_device = pd;
		render_pass = rp;
		command_pool = cp;
		graphics_queue = gq;

		if (!create_buffers())
		{
			return false;
		}

		if (!create_descriptor_sets_layouts())
		{
			return false;
		}

		if (!create_pipeline_layouts())
		{
			return false;
		}

		if (!create_pipelines())
		{
			return false;
		}

		if (buffer_fence != VK_NULL_HANDLE)
		{
			auto vk_result = vkWaitForFences(device, 1, &buffer_fence, true, UINT64_MAX);
			// TODO: handle error
			vkDestroyFence(device, buffer_fence, nullptr);
			vkFreeCommandBuffers(device, command_pool, 1, &copy_buffer_command_buffer);
			staging_buffer.destroy(); // we need to make sure the staging buffer remains live as well.
		}

		return true;
	}

	vku::Pipeline graphics_pipeline{};
	vku::PipelineLayout graphics_pipeline_layout{};
	vku::DescriptorSetLayout descriptor_set_layout{};

	vku::Buffer vertex_buffer{};
	vku::Buffer index_buffer{};

	// temporary stuff used when populating the vertex and index buffers.
	vku::Buffer staging_buffer{};
	VkFence buffer_fence{ VK_NULL_HANDLE }; // TODO: this and copy_buffer_command_buffer aren't destroyed in some error cases.
	VkCommandBuffer copy_buffer_command_buffer{ VK_NULL_HANDLE };

	inline bool create_buffers()
	{
		vku::BufferFactory factory(device, physical_device);

		size_t vertices_size = sizeof(Vertex) * vertices.size();
		size_t indices_size = sizeof(uint16_t) * indices.size();

		auto staging_buffer_result = factory.build(vertices_size + indices_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!staging_buffer_result)
		{
			std::cout << "failed to create staging buffer" << std::endl;
			return false;
		}

		// this will be destroyed when it goes out of scope at the end of the functin.
		staging_buffer = staging_buffer_result.get_value();

		auto vb_result = factory.build(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!vb_result)
		{
			std::cout << "failed to create vertex buffer" << std::endl;
			return false;
		}

		vertex_buffer = vb_result.get_value();

		auto ib_result = factory.build(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!ib_result)
		{
			std::cout << "failed to index vertex buffer" << std::endl;
			return false;
		}

		index_buffer = ib_result.get_value();

		VkResult map_result = staging_buffer.map_memory();

		if (map_result != VK_SUCCESS)
		{
			std::cout << "error mapping staging buffer" << std::endl;
			return false;
		}

		memcpy(staging_buffer.get_mapped(), vertices.data(), vertices_size);
		memcpy(static_cast<char*>(staging_buffer.get_mapped()) + vertices_size, indices.data(), indices_size);

		staging_buffer.unmap_memory();

		// copy the staging buffer to the vertex buffer.
		vku::SingleTimeCommandExecutor executor(device, command_pool, graphics_queue);

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		vkCreateFence(device, &fence_create_info, nullptr, &buffer_fence);

		vku::Error error{};
		auto execute_result = executor.execute([&](VkCommandBuffer commandBuffer) -> bool {
			VkBufferCopy copyRegion{};
			copyRegion.dstOffset = 0;
			copyRegion.srcOffset = 0;
			copyRegion.size = vertices_size;
			vkCmdCopyBuffer(commandBuffer, staging_buffer, vertex_buffer, 1, &copyRegion);

			copyRegion.dstOffset = 0;
			copyRegion.srcOffset = vertices_size;
			copyRegion.size = indices_size;
			vkCmdCopyBuffer(commandBuffer, staging_buffer, index_buffer, 1, &copyRegion);

			return true;
		}, buffer_fence);

		if (!execute_result)
		{
			// TODO: report contents of error.
			return false;
		}

		// because we DID pass in a fence, execute_result contains a command buffer, and we need to delete it after the fence is signaled.
		copy_buffer_command_buffer = execute_result.get_value();

		return true;
	}

	bool create_descriptor_sets_layouts()
	{
		vku::DescriptorSetLayoutBuilder layout_builder(device);

		layout_builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);

		auto layout_builder_result = layout_builder.build();

		if (!layout_builder_result)
		{
			std::cerr << "failed to create descriptor set layout." << std::endl;
			return false;
		}

		descriptor_set_layout = layout_builder_result.get_value();

		return true;
	}

	void destroy_descriptor_set_layouts()
	{
		descriptor_set_layout.destroy();
	}

	bool create_pipeline_layouts()
	{
		// create pipeline layout
		vku::PipelineLayoutBuilder pipeline_layout_builder(device);

		pipeline_layout_builder.add_descriptor_set(descriptor_set_layout);

		auto pipeline_layout_result = pipeline_layout_builder.build();
		if (!pipeline_layout_result)
		{
			return false;
		}
		graphics_pipeline_layout = pipeline_layout_result.get_value();

		return true;
	}

	bool create_pipelines()
	{
		vku::ShaderModule vert_module = read_example_shader(device, "uniform_example.vert.spv");

		if (!vert_module.is_valid())
		{
			return false;
		}

		vku::ShaderModule frag_module = read_example_shader(device, "uniform_example.frag.spv");

		if (!frag_module.is_valid())
		{
			return false;
		}

		vku::GraphicsPipelineBuilder pipeline_builder(device);

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		pipeline_builder
			.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vert_module)
			.add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module)
			.set_viewport_count(1)
			.set_scissor_count(1)
			.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
			.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR)
			.set_pipeline_layout(graphics_pipeline_layout)
			.set_render_pass(render_pass)
			.add_color_blend_attachment(colorBlendAttachment)
			.add_vertex_binding(Vertex::get_binding_description())
			.add_vertex_attributes(Vertex::get_attribute_descriptions())
			.set_cull_mode(VK_CULL_MODE_BACK_BIT)
			.set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE);

		auto pipeline_result = pipeline_builder.build();
		if (!pipeline_result)
		{
			return false;
		}
		graphics_pipeline = pipeline_result.get_value();

		return true;
	}
};


class FrameData
{
public:
	FrameData() { }

	FrameData(FrameData &&other)
	{
		device = other.device;
		physical_device = other.physical_device;
		uniform_buffer = std::move(other.uniform_buffer);
	}

	~FrameData() { }

	inline void destroy()
	{
		uniform_buffer.destroy();
		// detroying the pool without freeing the sets it contains does not cause validation errors...
		// auto vk_result = vkFreeDescriptorSets(device, descriptor_pool, 1, &descriptor_set);
		// not sure what we can do here aside from log...
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	}

	inline bool create_descriptor_pool(const SharedContext &shared_context)
	{
		vku::DescriptorPoolBuilder descriptor_pool_builder(device);

		descriptor_pool_builder.add_descriptor_sets(shared_context.descriptor_set_layout);
		
		auto result = descriptor_pool_builder.build();

		if (!result)
		{
			std::cerr << "failed to create descriptor pool" << std::endl;
			return false;
		}

		descriptor_pool = result.get_value();

		return true;
	}

	inline bool create_descriptor_set(const SharedContext &shared_context)
	{
		vku::DescriptorSetBuilder builder(device, descriptor_pool, shared_context.descriptor_set_layout);

		builder.write_uniform_buffer(0, 0, uniform_buffer, 0, uniform_buffer.get_size());

		auto result = builder.build();

		if (!result)
		{
			std::cerr << "failed to create descriptor set" << std::endl;
			return false;
		}

		descriptor_set = result.get_value();

		return true;
	}

	inline bool init(const SharedContext &shared_context)
	{
		device = shared_context.device;
		physical_device = shared_context.physical_device;

		if (!create_buffers())
		{
			return false;
		}

		if (!create_descriptor_pool(shared_context))
		{
			return false;
		}

		if (!create_descriptor_set(shared_context))
		{
			return false;
		}

		return true;
	}

	bool create_buffers()
	{
		vku::BufferFactory buffer_factory(device, physical_device);

		auto uniform_buffer_result = buffer_factory.build(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!uniform_buffer_result)
		{
			std::cerr << "failed to create uniform buffer" << std::endl;
			return false;
		}

		uniform_buffer = uniform_buffer_result.get_value();

		if (VK_SUCCESS != uniform_buffer.map_memory())
		{
			std::cerr << "failed to map uniform buffer" << std::endl;
			return false;
		}

		// we keep it mapped; mapping is expensive and it's sometimes a good idea to not map/unmap each frame.
		// TODO: reference for limited size of mapped memory.

		return true;
	}

	inline void update(VkExtent2D swapchain_extent)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformData ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapchain_extent.width / (float)swapchain_extent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data = uniform_buffer.get_mapped();
		memcpy(data, &ubo, sizeof(ubo));
	}

	bool record_command_buffer_render_pass(VkCommandBuffer command_buffer, SharedContext &shared_context)
	{
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shared_context.graphics_pipeline);

		VkBuffer vertexBuffers[] = { shared_context.vertex_buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(command_buffer, 0, 1, vertexBuffers, offsets);

		vkCmdBindIndexBuffer(command_buffer, shared_context.index_buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shared_context.graphics_pipeline.get_layout(), 0, 1, &descriptor_set, 0, nullptr);

		vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		return true;
	}
private:
	VkDevice device{ VK_NULL_HANDLE };
	VkPhysicalDevice physical_device{ VK_NULL_HANDLE };

	vku::Buffer uniform_buffer;
	VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
	VkDescriptorSet descriptor_set{ VK_NULL_HANDLE };
};


class UniformBufferExampleApplication : public ExampleApplication
{
public:
	virtual bool update_frame_data(uint32_t current_frame) override
	{
		frame_data[current_frame].update(vkb_swapchain.extent);

		return true;
	}

	virtual bool record_command_buffer_render_pass(uint32_t i) override
	{
		return frame_data[current_frame].record_command_buffer_render_pass(command_buffers[i], shared_context);
	}

	virtual bool on_create()
	{
		if (!shared_context.init(vkb_device, vkb_physical_device, render_pass, command_pool, graphics_queue))
		{
			return false;
		}

		for (auto &fd : frame_data)
		{
			if (!fd.init(shared_context))
			{
				return false;
			}
		}

		return true;
	}

	virtual void on_destroy() override
	{
		for (auto& fd : frame_data)
		{
			fd.destroy();
		}
		shared_context.destroy();
	}

private:
	SharedContext shared_context{};
	std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frame_data{};
};

int main()
{
	return run_application<UniformBufferExampleApplication>();
}

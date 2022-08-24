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


const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}}
};


const std::vector<uint16_t> indices = {
	0, 1, 2, 0, 2, 3
};



class DeviceVertexBufferExampleApplication : public ExampleApplication
{
public:
	bool create_pipeline_layouts()
	{
		// create pipeline layout
		vku::PipelineLayoutBuilder pipeline_layout_builder(vkb_device);

		auto pipeline_layout_result = pipeline_layout_builder.build();
		if (!pipeline_layout_result)
		{
			return false;
		}
		pipeline_layout = pipeline_layout_result.get_value();

		return true;
	}

	bool create_pipelines()
	{
		vku::ShaderModule vert_module = read_example_shader("triangle_buffer.vert.spv");

		if (!vert_module.is_valid())
		{
			return false;
		}

		vku::ShaderModule frag_module = read_example_shader("triangle_buffer.frag.spv");

		if (!frag_module.is_valid())
		{
			return false;
		}

		vku::GraphicsPipelineBuilder pipeline_builder(vkb_device);

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

	/*
		This creates a fence and passes it to SingleTimeCommandExecutor::execute. This allows you to perform other operations while the transfer is happening, then wait
		for the fence before using the transfered data.
	*/
	inline bool create_buffers()
	{
		vku::BufferFactory factory(vkb_device, vkb_physical_device);

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
		memcpy(static_cast<char *>(staging_buffer.get_mapped()) + vertices_size, indices.data(), indices_size);

		staging_buffer.unmap_memory();

		// copy the staging buffer to the vertex buffer.
		vku::SingleTimeCommandExecutor executor(vkb_device, command_pool, graphics_queue);

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		vkCreateFence(vkb_device, &fence_create_info, nullptr, &buffer_fence);

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

	inline void destroy_buffers()
	{
		vertex_buffer.destroy();
		index_buffer.destroy();
	}

	virtual bool populate_command_buffer_render_pass(uint32_t i)
	{
		VkCommandBuffer command_buffer = command_buffers[i];

		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkBuffer vertexBuffers[] = { vertex_buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(command_buffer, 0, 1, vertexBuffers, offsets);

		vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		return true;
	}


	virtual bool on_create()
	{
		if (!create_buffers())
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

		// you could do other transfer operations here, then wait for the fence.

		if (buffer_fence != VK_NULL_HANDLE)
		{
			vkWaitForFences(vkb_device, 1, &buffer_fence, true, UINT64_MAX);
			vkDestroyFence(vkb_device, buffer_fence, nullptr);
			vkFreeCommandBuffers(vkb_device, command_pool, 1, &copy_buffer_command_buffer);
			staging_buffer.destroy(); // we need to make sure the staging buffer remains live as well.
		}

		return true;
	}

	virtual void on_destroy() override
	{
		destroy_buffers();
	}

private:
	vku::PipelineLayout pipeline_layout{};
	vku::Pipeline graphics_pipeline{};

	vku::Buffer vertex_buffer{};
	vku::Buffer index_buffer{};
	vku::Buffer staging_buffer{};
	VkFence buffer_fence{ VK_NULL_HANDLE };
	VkCommandBuffer copy_buffer_command_buffer{ VK_NULL_HANDLE };
};

int main()
{
	return run_application<DeviceVertexBufferExampleApplication>();
}

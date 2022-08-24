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


const std::vector<Vertex> vertices{
	{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}
};


class BufferExampleApplication : public ExampleApplication
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

		/*
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
		*/

		vku::GraphicsPipelineBuilder pipeline_builder(vkb_device);

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		pipeline_builder
			.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vert_module)
			.add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module)
		    .add_viewport(0.0f, 0.0f, (float)vkb_swapchain.extent.width, (float)vkb_swapchain.extent.height, 0.0f, 1.0f)
		    .add_scissor(0, 0, vkb_swapchain.extent.width, vkb_swapchain.extent.height)
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

	inline bool create_vertex_buffer()
	{
		vku::BufferFactory factory(vkb_device, vkb_physical_device);

		auto result = factory.build(sizeof(Vertex) * 3, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (!result)
		{
			std::cout << "failed to create vertex buffer" << std::endl;
			return false;
		}

		vertex_buffer = result.get_value();

		VkResult map_result = vertex_buffer.map_memory();

		if (map_result != VK_SUCCESS)
		{
			std::cout << "error mapping vertex buffer" << std::endl;
			return false;
		}

		memcpy(vertex_buffer.get_mapped(), vertices.data(), vertex_buffer.get_size());

		vertex_buffer.unmap_memory();

		return true;
	}

	inline void destroy_buffers()
	{
		vertex_buffer.destroy();
	}

	virtual bool record_command_buffer_render_pass(uint32_t i) override
	{
		vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkBuffer vertexBuffers[] = { vertex_buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertexBuffers, offsets);

		vkCmdDraw(command_buffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);

		return true;
	}

	virtual bool on_create() override
	{
		if (!create_vertex_buffer())
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

		return true;
	}

	virtual void on_destroy() override
	{
		destroy_buffers();
	}
private:
	vku::Buffer vertex_buffer{};
	vku::PipelineLayout pipeline_layout{};
	vku::Pipeline graphics_pipeline{};
};

int main()
{
	BufferExampleApplication example_application{};

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

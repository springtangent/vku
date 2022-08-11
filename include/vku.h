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

#include <vector>
#include <system_error>
#include <limits>

namespace vku
{
	struct Error
	{
		std::error_code type;
		VkResult result{ VK_SUCCESS };
	};

	template<typename T>
	class Result
	{
	public:
		Result(const T &v) : value{ v }, has_value{ true } { }
		Result(T &&v) : value{ std::move(v) }, has_value{ true } { }
		Result(Error e) : error(e), has_value{ false } { }
		Result(std::error_code error_code, VkResult result = VK_SUCCESS) : error{ error_code, result }, has_value{ false } { }
		Result(Result<T>&& r) : has_value(r.has_value)
		{
			if (has_value)
			{
				new (&value) T{ std::move(r.value) };
			}
			else
			{
				error = r.error;
			}
		}

		Result(const Result<T>&& r) = delete;

		~Result()
		{
			destroy();
		}

		inline bool full_error(Error &result) const
		{
			if(!has_value)
			{
				return false;
			}
			result = error;
			return true;
		}

		inline bool vk_result(VkResult &vk_result) const
		{
			if(has_value)
			{
				return false;
			}
			result = error.vk_result;
			return true;
		}

		inline bool error_code(std::error_code &result) const
		{
			if(has_value)
			{
				return false;
			}
			result = error.error_code;
			return true;
			
		}

		// returns true if success, false if error, allowing
		// auto result = returns_a_result();
		// if(!result)
		// {
		//     // handle failure case
		// } 
		// // handle success case
		//
		inline operator bool() const
		{
			return has_value;
		}

		inline T get_value()
		{
			if (has_value)
			{
				return std::move(value);
			}
			return std::move(T());
		}

		inline Result& operator=(Result&& result)
		{
			has_value = result.has_value;
			if (has_value)
				new (&value) T{ result.value };
			else
				error = result.error;
			result.clear();
			return *this;
		}
	private:
		inline void destroy()
		{
			if(has_value) { value.~T(); }
		}

		union
		{
			T value;
			Error error;
		};

		bool has_value;
	};

	namespace detail
	{
		enum class ShaderError {
			device_not_provided,
			failed_create_shader
		};

		struct ShaderErrorCategory : std::error_category
		{
			const char* name() const noexcept override
			{
				return "vku_shader";
			}

			std::string message(int err) const override
			{
				switch (static_cast<ShaderError>(err))
				{
					case ShaderError::device_not_provided:
						return "device_not_provided";
					case ShaderError::failed_create_shader:
						return "failed_create_shader";
					default:
						return "unknown";
				}
			}
		};

		const ShaderErrorCategory shader_error_category;

		std::error_code make_error_code(ShaderError shader_error)
		{
			return { static_cast<int>(shader_error), detail::shader_error_category };
		}


		enum class PipelineLayoutError {
			device_not_provided,
			failed_create_pipeline_layout
		};

		struct PipelineLayoutErrorCategory : std::error_category
		{
			const char* name() const noexcept override
			{
				return "vku_pipeline_layout";
			}

			std::string message(int err) const override
			{
				switch (static_cast<PipelineLayoutError>(err))
				{
				case PipelineLayoutError::device_not_provided:
					return "device_not_provided";
				case PipelineLayoutError::failed_create_pipeline_layout:
					return "failed_create_pipeline_layout";
				default:
					return "unknown";
				}
			}
		};

		const PipelineLayoutErrorCategory pipeline_layout_error_category;

		std::error_code make_error_code(PipelineLayoutError pipeline_layout_error)
		{
			return { static_cast<int>(pipeline_layout_error), detail::pipeline_layout_error_category };
		}


		enum class PipelineError {
			device_not_provided,
			failed_create_pipeline
		};

		struct PipelineErrorCategory : std::error_category
		{
			const char* name() const noexcept override
			{
				return "vku_pipeline";
			}

			std::string message(int err) const override
			{
				switch (static_cast<PipelineError>(err))
				{
				case PipelineError::device_not_provided:
					return "device_not_provided";
				case PipelineError::failed_create_pipeline:
					return "failed_create_pipeline_layout";
				default:
					return "unknown";
				}
			}
		};

		const PipelineErrorCategory pipeline_error_category;

		std::error_code make_error_code(PipelineError pipeline_error)
		{
			return { static_cast<int>(pipeline_error), detail::pipeline_error_category };
		}
	};


	// RAII shader module class
	class ShaderModule
	{
	public:
		ShaderModule() : device(VK_NULL_HANDLE), shader_module(VK_NULL_HANDLE) { }

		ShaderModule(VkDevice d, VkShaderModule s) : device(d), shader_module(s) { }

		ShaderModule(ShaderModule &&s) noexcept : device(s.device), shader_module(s.shader_module)
		{
			s.clear();
		}

		ShaderModule(const ShaderModule& s) = delete;

		~ShaderModule()
		{
			destroy();
		}

		inline ShaderModule& operator=(ShaderModule&& s)
		{
			device = s.device;
			shader_module = s.shader_module;
			s.clear();
			return *this;
		}

		inline operator VkShaderModule() const
		{
			return shader_module;
		}
	private:
		inline void clear() noexcept
		{
			device = VK_NULL_HANDLE;
			shader_module = VK_NULL_HANDLE;
		}

		inline void destroy() noexcept
		{
			if(VK_NULL_HANDLE != device && VK_NULL_HANDLE != shader_module)
			{
				vkDestroyShaderModule(device, shader_module, nullptr);
			}
			clear();
		}

		VkDevice device{ VK_NULL_HANDLE };
		VkShaderModule shader_module{ VK_NULL_HANDLE };
	};


	inline Result<ShaderModule> create_shader_module(VkDevice device, const std::vector<char>& code)
	{
		if (VK_NULL_HANDLE == device)
		{
			return Result<ShaderModule>(detail::make_error_code(detail::ShaderError::device_not_provided));
		}
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*> (code.data());

		VkShaderModule shader_module{ VK_NULL_HANDLE };
		VkResult vk_result = vkCreateShaderModule(device, &create_info, nullptr, &shader_module);

		if (vk_result != VK_SUCCESS)
		{
			return Result<ShaderModule>(detail::make_error_code(detail::ShaderError::failed_create_shader), vk_result);
		}

		return Result<ShaderModule>(ShaderModule(device, shader_module));
	}


	class PipelineLayout
	{
	public:
		PipelineLayout() : device(VK_NULL_HANDLE), pipeline_layout(VK_NULL_HANDLE) { }

		PipelineLayout(VkDevice d, VkPipelineLayout s) : device(d), pipeline_layout(s) { }

		PipelineLayout(PipelineLayout&& p) noexcept : device(p.device), pipeline_layout(p.pipeline_layout)
		{
			p.clear();
		}

		PipelineLayout(const PipelineLayout& s) = delete;

		~PipelineLayout() { destroy(); }

		inline PipelineLayout& operator=(PipelineLayout&& p)
		{
			device = p.device;
			pipeline_layout = p.pipeline_layout;
			p.clear();
			return *this;
		}

		inline operator VkPipelineLayout() const
		{
			return pipeline_layout;
		}

		inline void destroy()
		{
			if (device != VK_NULL_HANDLE && pipeline_layout != VK_NULL_HANDLE)
			{
				vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
			}
			clear();
		}
	private:

		inline void clear()
		{
			device = VK_NULL_HANDLE;
			pipeline_layout = VK_NULL_HANDLE;
		}

		VkDevice device{ VK_NULL_HANDLE };
		VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
	};


	class PipelineLayoutBuilder
	{
	public:
		PipelineLayoutBuilder(VkDevice d) : device(d) { }

		Result<PipelineLayout> build() const
		{
			VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };

			VkPipelineLayoutCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

			// TODO: descriptor set layouts, push constant ranges.

			VkResult vk_result = vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout);
			if (vk_result != VK_SUCCESS)
			{
				return Result<PipelineLayout>(detail::make_error_code(detail::PipelineLayoutError::failed_create_pipeline_layout), vk_result);
			}
			return  Result<PipelineLayout>(PipelineLayout(device, pipeline_layout));
		}
	private:
		VkDevice device{ VK_NULL_HANDLE };
	};


	class Pipeline
	{
	public:
		Pipeline() : device(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE) { }
		Pipeline(VkDevice d, VkPipeline p) : device(d), pipeline(p) { }
		Pipeline(Pipeline&& p) noexcept : device(p.device), pipeline(p.pipeline) { p.clear(); }
		~Pipeline() { destroy(); }

		inline bool is_valid() const
		{
			return device != VK_NULL_HANDLE && pipeline != VK_NULL_HANDLE;
		}

		inline void destroy()
		{
			if (is_valid())
			{
				vkDestroyPipeline(device, pipeline, nullptr);
			}
			clear();
		}

		inline Pipeline& operator=(Pipeline&& p)
		{
			// TODO: if we're valid, should we destroy ourselves?
			device = p.device;
			pipeline = p.pipeline;
			p.clear();
			return *this;
		}

		inline operator VkPipeline() const
		{
			return pipeline;
		}
	private:
		inline void clear()
		{
			device = VK_NULL_HANDLE;
			pipeline = VK_NULL_HANDLE;
		}

		VkDevice device{ VK_NULL_HANDLE };
		VkPipeline pipeline{ VK_NULL_HANDLE };
	};


	class GraphicsPipelineBuilder
	{
	public:
		GraphicsPipelineBuilder(VkDevice d) : device(d) { }
		~GraphicsPipelineBuilder() { }

		inline GraphicsPipelineBuilder& add_shader_stage(VkShaderStageFlagBits stage, VkShaderModule module, const char *name="main")
		{
			VkPipelineShaderStageCreateInfo stage_info = {};
			stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stage_info.stage = stage;
			stage_info.module = module;
			stage_info.pName = name;
			shader_stages.push_back(stage_info);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_viewport(float x, float y, float width, float height, float min_depth, float max_depth)
		{
			VkViewport viewport = {};
			viewport.x = x;
			viewport.y = y;
			viewport.width = width;
			viewport.height = height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			viewports.push_back(viewport);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_scissor(int32_t offsetx, int32_t offsety, uint32_t extentx, uint32_t extenty)
		{
			VkRect2D scissor{};
			scissor.offset = { offsetx, offsety };
			scissor.extent = { extentx, extenty };
			scissors.push_back(scissor);
			return *this;
		}

		inline GraphicsPipelineBuilder& set_cull_mode(VkCullModeFlagBits c)
		{
			cull_mode = c;
			return *this;
		}

		inline GraphicsPipelineBuilder& add_dynamic_state(VkDynamicState s)
		{
			dynamic_states.push_back(s);
			return *this;
		}

		inline GraphicsPipelineBuilder& set_render_pass(VkRenderPass rp, uint32_t sp = 0)
		{
			render_pass = rp;
			subpass = sp;
			return *this;
		}

		inline GraphicsPipelineBuilder& set_pipeline_layout(VkPipelineLayout pl)
		{
			pipeline_layout = pl;
			return *this;
		}
		
		inline GraphicsPipelineBuilder& add_color_blend_attachment(const VkPipelineColorBlendAttachmentState &a)
		{
			color_blend_attachments.push_back(a);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_vertex_binding(const VkVertexInputBindingDescription &binding)
		{
			vertex_bindings.push_back(binding);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_vertex_attributes(const std::vector<VkVertexInputAttributeDescription> &attributes)
		{
			vertex_attributes.insert(vertex_attributes.end(), attributes.begin(), attributes.end());
			return *this;
		}

		inline Result<Pipeline> build() const
		{
			if (device == VK_NULL_HANDLE)
			{
				// TODO: set error
			}

			VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
			vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			
			vertex_input_info.vertexBindingDescriptionCount = vertex_bindings.size();
			vertex_input_info.pVertexBindingDescriptions = vertex_bindings.data();

			vertex_input_info.vertexAttributeDescriptionCount = vertex_attributes.size();
			vertex_input_info.pVertexAttributeDescriptions = vertex_attributes.data();

			VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
			input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			input_assembly.primitiveRestartEnable = VK_FALSE;

			VkPipelineViewportStateCreateInfo viewport_state = {};
			viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewport_state.viewportCount = viewports.size();
			viewport_state.pViewports = viewports.data();
			viewport_state.scissorCount = scissors.size();
			viewport_state.pScissors = scissors.data();

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

			VkPipelineColorBlendStateCreateInfo color_blending = {};
			color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			color_blending.logicOpEnable = VK_FALSE;
			color_blending.logicOp = VK_LOGIC_OP_COPY;
			color_blending.attachmentCount = color_blend_attachments.size();
			color_blending.pAttachments = color_blend_attachments.data();
			color_blending.blendConstants[0] = 0.0f;
			color_blending.blendConstants[1] = 0.0f;
			color_blending.blendConstants[2] = 0.0f;
			color_blending.blendConstants[3] = 0.0f;

			VkPipelineDynamicStateCreateInfo dynamic_info = {};
			dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamic_info.dynamicStateCount = static_cast<uint32_t> (dynamic_states.size());
			dynamic_info.pDynamicStates = dynamic_states.data();

			VkGraphicsPipelineCreateInfo create_info{};

			create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			create_info.stageCount = shader_stages.size();
			create_info.pStages = shader_stages.data();
			create_info.pVertexInputState = &vertex_input_info;
			create_info.pInputAssemblyState = &input_assembly;
			create_info.pViewportState = &viewport_state;
			create_info.pRasterizationState = &rasterizer;
			create_info.pMultisampleState = &multisampling;
			create_info.pColorBlendState = &color_blending;
			create_info.pDynamicState = &dynamic_info;
			create_info.layout = pipeline_layout;
			create_info.renderPass = render_pass;
			create_info.subpass = subpass;
			create_info.basePipelineHandle = VK_NULL_HANDLE;

			VkPipeline graphics_pipeline{ VK_NULL_HANDLE };
			VkResult vk_result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &graphics_pipeline);
			if (vk_result != VK_SUCCESS)
			{
				return Result<Pipeline>(detail::make_error_code(detail::PipelineError::failed_create_pipeline), vk_result);
			}
			return  Result<Pipeline>(Pipeline(device, graphics_pipeline));
		}
	private:
		std::vector<VkDynamicState> dynamic_states{};
		VkCullModeFlagBits cull_mode{ VK_CULL_MODE_BACK_BIT  };
		std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
		std::vector<VkViewport> viewports{};
		std::vector<VkRect2D> scissors{};
		std::vector< VkPipelineColorBlendAttachmentState> color_blend_attachments{};

		VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
		VkRenderPass render_pass{ VK_NULL_HANDLE };
		uint32_t subpass{ 0 };

		VkDevice device{ VK_NULL_HANDLE };

		std::vector< VkVertexInputBindingDescription> vertex_bindings{};
		std::vector<VkVertexInputAttributeDescription> vertex_attributes{};
	};


	struct Buffer
	{
	public:
		Buffer(VkDevice d=VK_NULL_HANDLE, VkBuffer b = VK_NULL_HANDLE, VkDeviceMemory m = VK_NULL_HANDLE, VkDeviceSize s=0, void *map=nullptr) : device(d), buffer(b), memory(m), size(s), mapped(map) { }
		Buffer(Buffer&& d) : device(d.device), buffer(d.buffer), memory(d.memory), size(d.size), mapped(d.mapped)
		{ 
			d.clear();
		}
		~Buffer() { destroy(); }

		inline Buffer& operator=(Buffer&& b) noexcept
		{
			// TODO: if we're valid, should we destroy ourselves? It seems like we leak if we don't.
			device = b.device;
			buffer = b.buffer;
			memory = b.memory;
			size = b.size;
			mapped = b.mapped;
			b.clear();
			return *this;
		}

		inline operator VkBuffer() const
		{
			return buffer;
		}

		inline bool is_valid() const
		{
			return device != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE;
		}

		inline void destroy()
		{
			if (is_valid())
			{
				vkDestroyBuffer(device, buffer, nullptr);
				vkFreeMemory(device, memory, nullptr);
			}
			clear();
		}

		inline VkResult map_memory(VkDeviceSize s=VK_WHOLE_SIZE, VkDeviceSize offset=0)
		{
			// TODO: test for validity, convey non-vk errors.
			return vkMapMemory(device, memory, offset, s, offset, &mapped);
		}

		inline void* get_mapped() const
		{
			return mapped;
		}

		inline VkDeviceSize get_size() const
		{
			return size;
		}

		inline void unmap_memory()
		{
			vkUnmapMemory(device, memory);
			mapped = nullptr;
		}
	private:
		inline void clear()
		{
			device = VK_NULL_HANDLE;
			buffer = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;
			size = 0;
			mapped = nullptr;
		}

		VkDevice device{ VK_NULL_HANDLE };
		VkBuffer buffer{ VK_NULL_HANDLE };
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		VkDeviceSize size{ 0 };
		void* mapped{ nullptr };
	};

	const uint32_t MEMORY_TYPE_NOT_FOUND = std::numeric_limits<uint32_t>::max();
	const uint32_t PHYSICAL_DEVICE_NOT_PROVIDED = MEMORY_TYPE_NOT_FOUND - 1;

	inline uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags  properties)
	{
		if (physical_device == VK_NULL_HANDLE)
		{
			return PHYSICAL_DEVICE_NOT_PROVIDED;
		}

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((type_filter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		return MEMORY_TYPE_NOT_FOUND;
	}


	class BufferFactory
	{
	public:
		BufferFactory(VkDevice d, VkPhysicalDevice pd) : device(d), physical_device(pd) { }

		inline Result<Buffer> build(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_properties) const
		{
			if (device == VK_NULL_HANDLE)
			{
				// TODO: error case.
			}

			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VkBuffer buffer;
			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			{
				// TODO: error
				throw std::runtime_error("failed to create vertex buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			uint32_t type_index = find_memory_type(physical_device, memRequirements.memoryTypeBits, memory_properties);

			if (type_index == PHYSICAL_DEVICE_NOT_PROVIDED)
			{
				// TODO: error
			}
			
			if (type_index == MEMORY_TYPE_NOT_FOUND)
			{
				// TODO: error
			}

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = type_index;

			VkDeviceMemory memory{ VK_NULL_HANDLE };
			if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
			{
				// TODO: error handling
				throw std::runtime_error("failed to allocate vertex buffer memory!");
			}

			vkBindBufferMemory(device, buffer, memory, 0);

			/*
			void* data;
			vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &data);
			memcpy(data, vertices.data(), (size_t)bufferInfo.size);
			vkUnmapMemory(device, vertexBufferMemory);
			*/

			return Result<Buffer>(Buffer(device, buffer, memory, size, nullptr));
		}
	private:


		VkDevice device{ VK_NULL_HANDLE };
		VkPhysicalDevice physical_device;
	};


	class DeviceBufferBuilder
	{

	};
};
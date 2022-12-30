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
#include <array>
#include <optional>

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

		enum class PipelineError
		{
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

		// CommandBufferExecutorError
		enum class CommandBufferExecutorError {
			device_not_provided,
			command_pool_not_provided,
			queue_not_provided,
			allocate_command_buffer_failed,
			begin_command_buffer_failed,
			callback_failed,
			end_command_buffer_failed,
			queue_submit_failed,
			wait_idle_failed,
			already_in_progress,
			create_fence_failed,
			wait_failed
		};

		struct CommandBufferExecutorErrorCategory : std::error_category
		{
			const char* name() const noexcept override
			{
				return "vku_command_buffer_executor";
			}

			std::string message(int err) const override
			{
				switch (static_cast<CommandBufferExecutorError>(err))
				{
				case CommandBufferExecutorError::device_not_provided:
					return "device_not_provided";
				case CommandBufferExecutorError::command_pool_not_provided:
					return "failed_create_pipeline_layout";
				case CommandBufferExecutorError::queue_not_provided:
					return "queue_not_provided";
				case CommandBufferExecutorError::allocate_command_buffer_failed:
					return "allocate_command_buffer_failed";
				case CommandBufferExecutorError::begin_command_buffer_failed:
					return "begin_command_buffer_failed";
				case CommandBufferExecutorError::queue_submit_failed:
					return "queue_submit_failed";
				case CommandBufferExecutorError::wait_idle_failed:
					return "wait_idle_failed";
				case CommandBufferExecutorError::end_command_buffer_failed:
					return "end_command_buffer_failed";
				case CommandBufferExecutorError::already_in_progress:
					return "already_in_progress";
				case CommandBufferExecutorError::create_fence_failed:
					return "create_fence_failed";
				case CommandBufferExecutorError::wait_failed:
					return "wait_failed";
				default:
					return "unknown";
				}
			}
		};

		const CommandBufferExecutorErrorCategory command_buffer_executor_error_category;

		std::error_code make_error_code(CommandBufferExecutorError command_buffer_executor_error)
		{
			return { static_cast<int>(command_buffer_executor_error), detail::command_buffer_executor_error_category };
		}


		// DescriptorSetLayoutError
		enum class DescriptorSetLayoutError {
			device_not_provided,
			create_failed
		};

		struct DescriptorSetLayoutErrorCategory : std::error_category
		{
			const char* name() const noexcept override
			{
				return "vku_descriptor_set_layout";
			}

			std::string message(int err) const override
			{
				switch (static_cast<DescriptorSetLayoutError>(err))
				{
				case DescriptorSetLayoutError::device_not_provided:
					return "device_not_provided";
				case DescriptorSetLayoutError::create_failed:
					return "create_failed";
				default:
					return "unknown";
				}
			}
		};

		const DescriptorSetLayoutErrorCategory descriptor_set_layout_error_category;

		std::error_code make_error_code(DescriptorSetLayoutError descriptor_set_layout_error)
		{
			return { static_cast<int>(descriptor_set_layout_error), detail::descriptor_set_layout_error_category };
		}
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


	template<typename T>
	class VkHandle
	{
	public:
		VkHandle() : device(VK_NULL_HANDLE), handle(VK_NULL_HANDLE) { }

		VkHandle(VkDevice d, T h) : device(d), handle(h) { }

		inline operator T() const
		{
			return handle;
		}

		inline VkDevice get_device() const
		{
			return device;
		}

		inline void clear()
		{
			device = VK_NULL_HANDLE;
			handle = VK_NULL_HANDLE;
		}

		inline bool is_valid() const
		{
			return handle != VK_NULL_HANDLE;
		}
	protected:
		VkDevice device{ VK_NULL_HANDLE };
		T handle{ VK_NULL_HANDLE };
	};

	/*
	class Queue : public VkHandle<VkQueue> { };
	class Surface : public VkHandle<VkSurfaceKHR> { };
	class ImageView : public VkHandle<VkImageView> { };
	class RenderPass : public VkHandle<VkRenderPass> { };
	class Framebuffer : public VkHandle<VkFramebuffer> { };
	class CommandPool : public VkHandle<VkCommandPool> { };
	class Semaphore : public VkHandle<VkSemaphore> { };
	class DescriptorSetLayout : public VkHandle<VkDescriptorSetLayout> { };
	*/

	class DescriptorPool : public VkHandle<VkDescriptorPool>
	{
	public:
		DescriptorPool() : VkHandle<VkDescriptorPool>(VK_NULL_HANDLE, VK_NULL_HANDLE) {}

		DescriptorPool(VkDevice d, VkDescriptorPool m) : VkHandle<VkDescriptorPool>(d, m) {}

		inline void destroy() noexcept
		{
			if (is_valid())
			{
				vkDestroyDescriptorPool(device, handle, nullptr);
			}
			clear();
		}
	};


	class DescriptorSet : public VkHandle<VkDescriptorSet>
	{
	public:
		DescriptorSet() : VkHandle<VkDescriptorSet>(VK_NULL_HANDLE, VK_NULL_HANDLE), descriptor_pool(VK_NULL_HANDLE) {}

		DescriptorSet(VkDevice d, VkDescriptorSet m, VkDescriptorPool dp) : VkHandle<VkDescriptorSet>(d, m), descriptor_pool(dp) {}

		inline void destroy() noexcept
		{
			if (is_valid())
			{
				vkFreeDescriptorSets(device, descriptor_pool, 1, &handle);
			}
			clear();
		}

		inline void clear()
		{
			VkHandle<VkDescriptorSet>::clear();
			descriptor_pool = VK_NULL_HANDLE;
		}
	protected:
		VkDescriptorPool descriptor_pool;
	};

	class Image : public VkHandle<VkImage>
	{
	public:
		Image() : VkHandle<VkImage>(), memory(VK_NULL_HANDLE), source_stage(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) { }
		Image(
			VkDevice d, 
			VkImage i, 
			VkDeviceMemory m, 
			VkPipelineStageFlagBits s= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) : 
			VkHandle<VkImage>(d, i), 
			memory(m),
			source_stage(s)
		{ }

		inline void destroy()
		{
			if (handle != VK_NULL_HANDLE)
			{
				vkDestroyImage(device, handle, nullptr);
			}

			if (memory != VK_NULL_HANDLE)
			{
				vkFreeMemory(device, memory, nullptr);
			}

			clear();
		}

		inline void clear()
		{
			VkHandle<VkImage>::clear();
			memory = VK_NULL_HANDLE;
			source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		}

		VkPipelineStageFlagBits source_stage{ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };
		VkAccessFlagBits src_access_mask{ VK_ACCESS_NONE };
	protected:
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		
	};


	class ImageFactory
	{
	public:
		ImageFactory(VkPhysicalDevice pd, VkDevice d) : physical_device(pd), device(d) { }

		Result<Image> build_image(VkImageCreateInfo create_info, VkMemoryPropertyFlags properties)
		{
			VkImage image{ VK_NULL_HANDLE };
			auto vk_create_image_result = vkCreateImage(device, &create_info, nullptr, &image);
			if (vk_create_image_result != VK_SUCCESS)
			{
				// TODO: error handling.
			}

			VkMemoryRequirements mem_reqs;
			vkGetImageMemoryRequirements(device, image, &mem_reqs);

			VkMemoryAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
			alloc_info.allocationSize = mem_reqs.size;
			alloc_info.memoryTypeIndex = find_memory_type(physical_device, mem_reqs.memoryTypeBits, properties);

			VkDeviceMemory image_memory{ VK_NULL_HANDLE };
			auto vk_alloc_result = vkAllocateMemory(device, &alloc_info, nullptr, &image_memory);

			if (vk_alloc_result != VK_SUCCESS)
			{
				// TODO: error handling
			}

			vkBindImageMemory(device, image, image_memory, 0);

			return Image(device, image, image_memory);
		}
		
		Result<Image> build_image_2d(VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags properties)
		{
			VkImageCreateInfo image_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
			image_info.pNext = nullptr;
			image_info.imageType = VK_IMAGE_TYPE_2D;
			image_info.extent = { extent.width, extent.height, 1 };
			image_info.mipLevels = 1;
			image_info.arrayLayers = 1;
			image_info.format = format;
			image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
			image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			image_info.samples = VK_SAMPLE_COUNT_1_BIT;
			image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			image_info.usage = usage;

			return build_image(image_info, properties);
		}
	private:
		VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
		VkDevice device{ VK_NULL_HANDLE };
	};


	class ShaderModule : public VkHandle<VkShaderModule>
	{
	public:
		ShaderModule() : VkHandle<VkShaderModule>() {}

		ShaderModule(VkDevice d, VkShaderModule m) : VkHandle<VkShaderModule>(d, m) {}

		inline void destroy() noexcept
		{
			if (is_valid())
			{
				vkDestroyShaderModule(device, handle, nullptr);
			}
			clear();
		}
	};


	class PipelineLayout : public VkHandle<VkPipelineLayout>
	{
	public:
		PipelineLayout() : VkHandle<VkPipelineLayout>() { }

		PipelineLayout(VkDevice d, VkPipelineLayout s) : VkHandle<VkPipelineLayout>(d, s) { }

		inline void destroy()
		{
			if (is_valid())
			{
				vkDestroyPipelineLayout(device, handle, nullptr);
			}
			clear();
		}
	};


	class Pipeline : public VkHandle<VkPipeline>
	{
	public:
		Pipeline() : VkHandle<VkPipeline>(), layout(VK_NULL_HANDLE) { }

		Pipeline(VkDevice d, VkPipeline p, VkPipelineLayout pl) : VkHandle<VkPipeline>(d, p), layout(pl) { }

		inline void destroy()
		{
			if (is_valid())
			{
				vkDestroyPipeline(device, handle, nullptr);
			}
			clear();
		}

		inline VkPipelineLayout get_layout() const
		{
			return layout;
		}
	protected:
		VkPipelineLayout layout{ VK_NULL_HANDLE };
	};


	class Buffer : public VkHandle<VkBuffer>
	{
	public:
		Buffer(VkDevice d = VK_NULL_HANDLE, VkBuffer b = VK_NULL_HANDLE, VkDeviceMemory m = VK_NULL_HANDLE, VkDeviceSize s = 0, void* map = nullptr) : VkHandle<VkBuffer>(d, b), memory(m), size(s), mapped(map) { }

		inline operator VkBuffer() const
		{
			return handle;
		}

		inline bool is_valid() const
		{
			return VkHandle<VkBuffer>::is_valid() && memory != VK_NULL_HANDLE;
		}

		inline void destroy()
		{
			if (handle != VK_NULL_HANDLE)
			{
				vkDestroyBuffer(device, handle, nullptr);
			}

			if (memory != VK_NULL_HANDLE)
			{
				vkFreeMemory(device, memory, nullptr);
			}

			clear();
		}

		inline VkResult map_memory(VkDeviceSize s = VK_WHOLE_SIZE, VkDeviceSize offset = 0)
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

		inline void clear()
		{
			VkHandle<VkBuffer>::clear();
			memory = VK_NULL_HANDLE;
			size = 0;
			mapped = nullptr;
		}
	protected:
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		VkDeviceSize size{ 0 };
		void* mapped{ nullptr };
	};


	class Fence : public VkHandle<VkFence>
	{
	public:
		Fence() : VkHandle<VkFence>() { }
		Fence(VkDevice d, VkFence f) : VkHandle<VkFence>(d, f) { }

		void destroy()
		{
			if (is_valid())
			{
				vkDestroyFence(device, handle, nullptr);
			}
			clear();
		}
	};


	class CommandBuffer : public VkHandle<VkCommandBuffer>
	{
	public:
		CommandBuffer() : VkHandle<VkCommandBuffer>(), command_pool(VK_NULL_HANDLE)  {}
		CommandBuffer(VkDevice d, VkCommandBuffer c, VkCommandPool p) : VkHandle<VkCommandBuffer>(d, c), command_pool(p) { }

		bool is_valid() const
		{
			return VkHandle<VkCommandBuffer>::is_valid() && command_pool != VK_NULL_HANDLE;
		}

		void destroy()
		{
			if (is_valid())
			{
				vkFreeCommandBuffers(device, command_pool, 1, &handle);
			}
			clear();
		}
	protected:
		VkCommandPool command_pool{ VK_NULL_HANDLE };
	};


	enum class DescriptorType
	{
		SAMPLER = 0,				// VK_DESCRIPTOR_TYPE_SAMPLER = 0,
		COMBINED_IMAGE_SAMPLER = 1,	// VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
		SAMPLED_IMAGE = 2,			// VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2,
		STORAGE_IMAGE = 3,			// VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
		UNIFORM_TEXEL_BUFFER = 4,	// VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER = 4,
		STORAGE_TEXEL_BUFFER = 5,	// VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER = 5,
		UNIFORM_BUFFER = 6,			// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6,
		STORAGE_BUFFER = 7,			// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
		UNIFORM_BUFFER_DYNAMIC = 8, // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8,
		STORAGE_BUFFER_DYNAMIC = 9, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9,
		INPUT_ATTACHMENT = 10,		// VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT = 10,

		/*
		// Provided by VK_VERSION_1_3
		INLINE_UNIFORM_BLOCK = 11,  // VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK = 1000138000,
		// Provided by VK_KHR_acceleration_structure
		ACCELERATION_STRUCTURE_KHR = 12, // VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000,
		// Provided by VK_NV_ray_tracing
		ACCELERATION_STRUCTURE_NV = 13, // VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV = 1000165000,
		// Provided by VK_VALVE_mutable_descriptor_type
		MUTABLE_VALVE = 14, // VK_DESCRIPTOR_TYPE_MUTABLE_VALVE = 1000351000,
		// Provided by VK_QCOM_image_processing
		SAMPLE_WEIGHT_IMAGE_QCOM = 15, // VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM = 1000440000,
		// Provided by VK_QCOM_image_processing
		BLOCK_MATCH_IMAGE_QCOM = 16, // VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM = 1000440001,
		// Provided by VK_EXT_inline_uniform_block
		INLINE_UNIFORM_BLOCK_EXT = INLINE_UNIFORM_BLOCK,
		*/
		MAX = INPUT_ATTACHMENT + 1,
		UNKNOWN = MAX + 1
	};

	DescriptorType from_vk(VkDescriptorType t)
	{
		switch (t)
		{
		case(VK_DESCRIPTOR_TYPE_SAMPLER):
			return DescriptorType::SAMPLER;
		case(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER):
			return DescriptorType::COMBINED_IMAGE_SAMPLER;
		case(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE):
			return DescriptorType::SAMPLED_IMAGE;
		case(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE):
			return DescriptorType::STORAGE_IMAGE;
		case(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER):
			return DescriptorType::UNIFORM_TEXEL_BUFFER;
		case(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER):
			return DescriptorType::STORAGE_TEXEL_BUFFER;
		case(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER):
			return DescriptorType::UNIFORM_BUFFER;
		case(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER):
			return DescriptorType::STORAGE_BUFFER;
		case(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC):
			return DescriptorType::UNIFORM_BUFFER_DYNAMIC;
		case(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC):
			return DescriptorType::STORAGE_BUFFER_DYNAMIC;
		case(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT):
			return DescriptorType::INPUT_ATTACHMENT;
		}

		return DescriptorType::UNKNOWN;
	}

	VkDescriptorType to_vk(DescriptorType t)
	{
		switch (t)
		{
		case(DescriptorType::SAMPLER):
			return VK_DESCRIPTOR_TYPE_SAMPLER;
		case(DescriptorType::COMBINED_IMAGE_SAMPLER):
			return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		case(DescriptorType::SAMPLED_IMAGE):
			return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		case(DescriptorType::STORAGE_IMAGE):
			return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		case(DescriptorType::UNIFORM_TEXEL_BUFFER):
			return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
		case(DescriptorType::STORAGE_TEXEL_BUFFER):
			return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
		case(DescriptorType::UNIFORM_BUFFER):
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case(DescriptorType::STORAGE_BUFFER):
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		case(DescriptorType::UNIFORM_BUFFER_DYNAMIC):
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		case(DescriptorType::STORAGE_BUFFER_DYNAMIC):
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
		case(DescriptorType::INPUT_ATTACHMENT):
			return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		}

		return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}





	using DescriptorTypeCounts = std::array<uint16_t, static_cast<size_t>(DescriptorType::MAX)>;

	class DescriptorSetLayout : public VkHandle<VkDescriptorSetLayout>
	{
	public:
		DescriptorSetLayout() : VkHandle<VkDescriptorSetLayout>(), descriptor_pool(VK_NULL_HANDLE), descriptor_type_counts() { }
		DescriptorSetLayout(VkDevice d, VkDescriptorSetLayout ds, const DescriptorTypeCounts& dsc) : VkHandle<VkDescriptorSetLayout>(d, ds), descriptor_type_counts(dsc) { }

		inline void destroy()
		{
			if (is_valid())
			{
				vkDestroyDescriptorSetLayout(device, handle, nullptr);
			}

			clear();
		}

	private:
		inline void clear()
		{
			VkHandle<VkDescriptorSetLayout>::clear();
			descriptor_pool = VK_NULL_HANDLE;
			descriptor_type_counts = {};
		}
		VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
		DescriptorTypeCounts descriptor_type_counts{};

		friend class DescriptorPoolBuilder; // allow access to descriptor_type_counts
	};



	inline Result<ShaderModule> create_shader_module(VkDevice device, const uint32_t *code, VkDeviceSize size)
	{
		if (VK_NULL_HANDLE == device)
		{
			return Result<ShaderModule>(detail::make_error_code(detail::ShaderError::device_not_provided));
		}
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = size;
		create_info.pCode = code;

		VkShaderModule shader_module{ VK_NULL_HANDLE };
		VkResult vk_result = vkCreateShaderModule(device, &create_info, nullptr, &shader_module);

		if (vk_result != VK_SUCCESS)
		{
			return Result<ShaderModule>(detail::make_error_code(detail::ShaderError::failed_create_shader), vk_result);
		}

		return Result<ShaderModule>(ShaderModule(device, shader_module));
	}


	inline Result<ShaderModule> create_shader_module(VkDevice device, const std::vector<char>& code)
	{
		return create_shader_module(device, reinterpret_cast<const uint32_t*>(code.data()), code.size());
	}


	class PipelineLayoutBuilder
	{
	public:
		PipelineLayoutBuilder(VkDevice d) : device(d) { }

		inline PipelineLayoutBuilder& add_descriptor_set(VkDescriptorSetLayout descriptor_set_layout)
		{
			descriptor_set_layouts.push_back(descriptor_set_layout);
			return *this;
		}

		Result<PipelineLayout> build() const
		{
			VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };

			VkPipelineLayoutCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

			// TODO: descriptor set layouts, push constant ranges.
			create_info.pSetLayouts = descriptor_set_layouts.data();
			create_info.setLayoutCount = descriptor_set_layouts.size();

			VkResult vk_result = vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout);
			if (vk_result != VK_SUCCESS)
			{
				return Result<PipelineLayout>(detail::make_error_code(detail::PipelineLayoutError::failed_create_pipeline_layout), vk_result);
			}
			return  Result<PipelineLayout>(PipelineLayout(device, pipeline_layout));
		}
	private:
		VkDevice device{ VK_NULL_HANDLE };
		std::vector<VkDescriptorSetLayout> descriptor_set_layouts{};
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

		inline GraphicsPipelineBuilder& add_viewport(VkViewport& viewport)
		{
			viewports.push_back(viewport);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_viewport(float x, float y, float width, float height, float min_depth, float max_depth)
		{
			VkViewport viewport = { x, y, width,height, min_depth, max_depth };
			return add_viewport(viewport);
		}

		inline GraphicsPipelineBuilder& set_viewport_count(uint32_t count)
		{
			viewport_count = count;
			return *this;
		}

		inline GraphicsPipelineBuilder& add_scissor(VkRect2D scissor)
		{
			scissors.push_back(scissor);
			return *this;
		}

		inline GraphicsPipelineBuilder& add_scissor(int32_t offsetx, int32_t offsety, uint32_t extentx, uint32_t extenty)
		{
			VkRect2D scissor{ { offsetx, offsety }, { extentx, extenty } };
			return add_scissor(scissor);
		}

		inline GraphicsPipelineBuilder& set_scissor_count(uint32_t count)
		{
			scissor_count = count;
			return *this;
		}

		inline GraphicsPipelineBuilder& set_cull_mode(VkCullModeFlags c)
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

		inline GraphicsPipelineBuilder& set_front_face(VkFrontFace value)
		{
			front_face = value;
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
			viewport_state.viewportCount = viewports.size() ? viewports.size() : viewport_count;
			viewport_state.pViewports = viewports.data();
			viewport_state.scissorCount = scissors.size() ? scissors.size() : scissor_count;
			viewport_state.pScissors = scissors.data();

			VkPipelineRasterizationStateCreateInfo rasterizer = {};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE;
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = cull_mode;
			rasterizer.frontFace = front_face;
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
			return  Result<Pipeline>(Pipeline(device, graphics_pipeline, pipeline_layout));
		}
	private:
		std::vector<VkDynamicState> dynamic_states{};
		VkCullModeFlags cull_mode{ VK_CULL_MODE_BACK_BIT  };
		VkFrontFace front_face{ VK_FRONT_FACE_CLOCKWISE };
		std::vector<VkPipelineShaderStageCreateInfo> shader_stages{};
		std::vector<VkViewport> viewports{};
		uint32_t viewport_count{ 0 };
		std::vector<VkRect2D> scissors{};
		uint32_t scissor_count{ 0 };
		std::vector< VkPipelineColorBlendAttachmentState> color_blend_attachments{};

		VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
		VkRenderPass render_pass{ VK_NULL_HANDLE };
		uint32_t subpass{ 0 };

		VkDevice device{ VK_NULL_HANDLE };

		std::vector< VkVertexInputBindingDescription> vertex_bindings{};
		std::vector<VkVertexInputAttributeDescription> vertex_attributes{};
	};


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

			return Result<Buffer>(Buffer(device, buffer, memory, size, nullptr));
		}
	private:
		VkDevice device{ VK_NULL_HANDLE };
		VkPhysicalDevice physical_device;
	};

	class SingleTimeCommandExecutor
	{
	public:
		SingleTimeCommandExecutor() : device(VK_NULL_HANDLE), command_pool(VK_NULL_HANDLE), queue(VK_NULL_HANDLE)
		{
		}

		SingleTimeCommandExecutor(VkDevice _device, VkCommandPool _command_pool, VkQueue _queue) : device(_device), command_pool(_command_pool), queue(_queue)
		{
		}

		void init(VkDevice _device, VkCommandPool _command_pool, VkQueue _queue)
		{
			device = _device;
			command_pool = _command_pool;
			queue = _queue;
		}

		inline bool is_valid() const
		{
			return device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE && queue != VK_NULL_HANDLE;
		}

		inline bool in_progress() const
		{
			return command_buffer != VK_NULL_HANDLE;
		}

		inline void destroy()
		{
			if (command_buffer != VK_NULL_HANDLE)
			{
				vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
			}

			if (fence != VK_NULL_HANDLE)
			{
				vkDestroyFence(device, fence, nullptr);
			}
		}

		Result<CommandBuffer> enter()
		{
			Error error{};

			if (in_progress())
			{
				error = Error{ detail::make_error_code(detail::CommandBufferExecutorError::already_in_progress) };
				return Result<CommandBuffer>(error);
			}

			if (!begin_single_time_commands(command_buffer, error))
			{
				destroy();
				return Result<CommandBuffer>(error);
			}

			return Result<CommandBuffer>(CommandBuffer(device, command_buffer, command_pool));
		}

		Result<Fence> exit()
		{
			Error error{};
			bool result = end_single_time_commands(command_buffer, error);

			if (!result)
			{
				// we have a valid command buffer, but there was an error ending the single time commands.
				destroy();
				return Result<Fence>(error);
			}

			return Result<Fence>(Fence(device, fence));
		}

		// waiting destroys the created command buffer and fence.
		std::optional<Error> wait()
		{
			auto vk_result = vkWaitForFences(device, 1, &fence, true, UINT64_MAX);

			Error error{};

			if (vk_result != VK_SUCCESS)
			{
				// we have a valid command buffer, but there was an error ending the single time commands.
				error = Error{ detail::make_error_code(detail::CommandBufferExecutorError::wait_failed) };
				return { error };
			}

			return {};
		}

	private:
		bool create_fence(Error &error)
		{
			VkFenceCreateInfo fence_create_info{};
			fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fence_create_info.pNext = nullptr;
			fence_create_info.flags = 0;

			auto vk_result = vkCreateFence(device, &fence_create_info, nullptr, &fence);

			if (vk_result != VK_SUCCESS)
			{
				error = { detail::make_error_code(detail::CommandBufferExecutorError::create_fence_failed), vk_result };
				return false;
			}

			return true;
		}

		bool begin_single_time_commands(VkCommandBuffer &commandBuffer, Error &error) const
		{
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = command_pool;
			allocInfo.commandBufferCount = 1;
			commandBuffer = VK_NULL_HANDLE;

			auto vk_result = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

			if (vk_result != VK_SUCCESS)
			{
				error = { detail::make_error_code(detail::CommandBufferExecutorError::allocate_command_buffer_failed), vk_result };
				return false;
			}

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vk_result = vkBeginCommandBuffer(commandBuffer, &beginInfo);

			if (vk_result != VK_SUCCESS)
			{
				error = { detail::make_error_code(detail::CommandBufferExecutorError::begin_command_buffer_failed), vk_result };
				return false;
			}

			return true;
		}

		// returns false if error occurred.
		bool end_single_time_commands(VkCommandBuffer commandBuffer, Error &error)
		{
			auto vk_result = vkEndCommandBuffer(commandBuffer);
			if (vk_result != VK_SUCCESS)
			{
				error = Error{ detail::make_error_code(detail::CommandBufferExecutorError::end_command_buffer_failed), vk_result };
				return false;
			}

			if (!create_fence(error))
			{
				error = Error{ detail::make_error_code(detail::CommandBufferExecutorError::create_fence_failed), vk_result };
				return false;
			}

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			vk_result = vkQueueSubmit(queue, 1, &submitInfo, fence);

			if (vk_result != VK_SUCCESS)
			{
				error = Error{ detail::make_error_code(detail::CommandBufferExecutorError::queue_submit_failed), vk_result };
				return false;
			}

			return true;
		}

		VkDevice device{ VK_NULL_HANDLE };
		VkQueue queue{ VK_NULL_HANDLE };
		VkCommandPool command_pool{ VK_NULL_HANDLE };
		VkCommandBuffer command_buffer{ VK_NULL_HANDLE };
		VkFence fence{ VK_NULL_HANDLE };
	};



	class DescriptorSetLayoutBuilder
	{
	public:
		DescriptorSetLayoutBuilder() = delete;
		DescriptorSetLayoutBuilder(VkDevice d) : device(d) { }
		~DescriptorSetLayoutBuilder() { }

		inline Result<DescriptorSetLayout> build() const
		{
			if (device == VK_NULL_HANDLE)
			{
				return Result<DescriptorSetLayout>(Error{detail::make_error_code(detail::DescriptorSetLayoutError::device_not_provided)});
			}

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = bindings.size();
			layoutInfo.pBindings = bindings.data();

			VkDescriptorSetLayout descriptor_set_layout{ VK_NULL_HANDLE };

			auto vk_result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptor_set_layout);
			if (vk_result != VK_SUCCESS)
			{
				return Result<DescriptorSetLayout>(Error{ detail::make_error_code(detail::DescriptorSetLayoutError::create_failed), vk_result });
			}

			return Result<DescriptorSetLayout>(std::move(DescriptorSetLayout(device, descriptor_set_layout, descriptor_type_counts)));
		}

		inline DescriptorSetLayoutBuilder& add_binding(uint32_t binding, VkDescriptorType descriptor_type, uint32_t descriptor_count, VkShaderStageFlags stage_flags)
		{
			VkDescriptorSetLayoutBinding b{ binding , descriptor_type, descriptor_count, stage_flags, nullptr};
			return add_binding(b);
		}

		inline DescriptorSetLayoutBuilder& add_binding(VkDescriptorSetLayoutBinding b)
		{
			bindings.push_back(b);
			auto t = from_vk(b.descriptorType);
			// TODO: check this, I don't think decriptorCount always works like this...
			descriptor_type_counts[static_cast<uint16_t>(t)] += b.descriptorCount;
			return *this;
		}
	private:
		VkDevice device{ VK_NULL_HANDLE };
		std::vector<VkDescriptorSetLayoutBinding> bindings{};
		DescriptorTypeCounts descriptor_type_counts{};
	};


	class DescriptorPoolBuilder
	{
	public:
		DescriptorPoolBuilder() = delete;
		DescriptorPoolBuilder(VkDevice d) : device(d) { }
		
		inline DescriptorPoolBuilder& add_descriptor_sets(const DescriptorSetLayout &set_layout, uint32_t count=1)
		{
			max_sets += count;
			for (uint16_t i = 0; i < static_cast<size_t>(DescriptorType::MAX); ++i)
			{
				descriptor_type_counts[i] += set_layout.descriptor_type_counts[i] * count;
			}
			return *this;
		}

		inline Result<DescriptorPool> build() const
		{
			VkDescriptorPool result{ VK_NULL_HANDLE };

			std::vector<VkDescriptorPoolSize> pool_sizes{};
			for (uint16_t i = 0; i < static_cast<size_t>(DescriptorType::MAX); ++i)
			{
				if (descriptor_type_counts[i] > 0)
				{
					VkDescriptorPoolSize pool_size{};
					pool_size.type = to_vk(static_cast<DescriptorType>(i));
					pool_size.descriptorCount = descriptor_type_counts[i];
					pool_sizes.push_back(pool_size);
				}
			}

			VkDescriptorPoolCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			create_info.maxSets = max_sets;
			create_info.poolSizeCount = pool_sizes.size();
			create_info.pPoolSizes = pool_sizes.data();

			auto vk_result = vkCreateDescriptorPool(device, &create_info, nullptr, &result);

			if (vk_result != VK_SUCCESS)
			{
				// TODO: error and stuff
			}

			return Result<DescriptorPool>(DescriptorPool(device, result));
		}
	private:
		VkDevice device{ VK_NULL_HANDLE };
		uint32_t max_sets{ 0 };
		DescriptorTypeCounts descriptor_type_counts{};
	};

	class DescriptorSetBuilder
	{
	public:
		DescriptorSetBuilder(VkDevice _device, VkDescriptorPool _descriptor_pool, VkDescriptorSetLayout _layout) : device(_device), descriptor_pool(_descriptor_pool), layout(_layout) { }
		~DescriptorSetBuilder() = default;

		inline bool is_valid() const { return device != VK_NULL_HANDLE; }

		inline DescriptorSetBuilder& write_uniform_buffer(uint32_t binding, uint32_t array_element, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
		{
			UniformBufferWrite write{};
			write.binding = binding;
			write.array_element = array_element;
			write.buffer = buffer;
			write.offset = offset;
			write.range = range;
			uniform_buffer_writes.push_back(write);
			return *this;
		}

		inline DescriptorSetBuilder& write_combined_image_sampler(uint32_t binding, uint32_t array_element, VkSampler sampler, VkImageView image_view, VkImageLayout image_layout)
		{
			CombinedImageSamplerWrite write{};
			write.binding = binding;
			write.array_element = array_element;
			write.sampler = sampler;
			write.image_view = image_view;
			write.image_layout = image_layout;
			combined_image_sampler_writes.push_back(write);
			return *this;
		}

		inline Result<DescriptorSet> build() const
		{
			// allocate the descriptor sets.
			VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };

			alloc_info.descriptorPool = descriptor_pool;
			alloc_info.descriptorSetCount = 1;
			alloc_info.pSetLayouts = &layout;

			VkDescriptorSet result{ VK_NULL_HANDLE };
			auto vk_result = vkAllocateDescriptorSets(device, &alloc_info, &result);

			if (vk_result != VK_SUCCESS)
			{
				// return res;
				// TODO: error case.
			}

			// write the descriptor sets
			VkWriteDescriptorSet w{};
			w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet = result;
			std::vector<VkWriteDescriptorSet> writes(uniform_buffer_writes.size() + combined_image_sampler_writes.size(), w);

			// populate the bufferinfos
			std::vector<VkDescriptorBufferInfo> buffer_infos(uniform_buffer_writes.size());

			// populate the buffer writes
			uint32_t write_index = 0;
			uint32_t buffer_index = 0;
			for (const auto &uniform_buffer_write: uniform_buffer_writes)
			{
				uniform_buffer_write.populate(writes[write_index], buffer_infos[buffer_index]);
				buffer_index += 1;
				write_index += 1;
			}

			// populate the image_infos
			std::vector<VkDescriptorImageInfo> image_infos(combined_image_sampler_writes.size());

			// populate the image writes
			uint32_t image_index = 0;
			for (const auto &combined_image_sampler_write: combined_image_sampler_writes)
			{
				combined_image_sampler_write.populate(writes[write_index], image_infos[image_index]);
				image_index += 1;
				write_index += 1;
			}

			// perform the writes
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

			return Result<DescriptorSet>(DescriptorSet(device, result, descriptor_pool));
		}
	private:
		struct UniformBufferWrite
		{
			uint32_t binding;
			uint32_t array_element;
			VkBuffer buffer;
			VkDeviceSize offset;
			VkDeviceSize range;

			inline void populate(VkWriteDescriptorSet& write, VkDescriptorBufferInfo& buffer_info) const
			{
				write.pBufferInfo = &buffer_info;
				write.descriptorCount = 1;
				write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				write.dstArrayElement = array_element;

				buffer_info.buffer = buffer;
				buffer_info.offset = offset;
				buffer_info.range = range;
			}
		};

		struct CombinedImageSamplerWrite
		{
			uint32_t binding;
			uint32_t array_element;
			VkSampler sampler;
			VkImageView image_view;
			VkImageLayout image_layout;

			inline void populate(VkWriteDescriptorSet& write, VkDescriptorImageInfo &image_info) const
			{
				write.pImageInfo = &image_info;
				write.descriptorCount = 1;
				write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				write.dstArrayElement = array_element;

				image_info.imageLayout = image_layout;
				image_info.imageView = image_view;
				image_info.sampler = sampler;
			}
		};

		VkDevice device{ VK_NULL_HANDLE };
		VkDescriptorSetLayout layout{ VK_NULL_HANDLE };
		VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };

		std::vector<UniformBufferWrite> uniform_buffer_writes{};
		std::vector<CombinedImageSamplerWrite> combined_image_sampler_writes{};
	};
};
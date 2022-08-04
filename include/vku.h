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
		// if(Result<Whatever> = returns_a_result())
		// {
		//     // handle success case
		// } else {
		//     // handle failure case
		// }
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

		return std::move(Result<ShaderModule>(ShaderModule(device, shader_module)));
	}
};
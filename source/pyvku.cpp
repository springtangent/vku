#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <vku.h>

#include <stdexcept>
#include <tuple>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace py = pybind11;


class ImageData
{
public:
    int width{0};
    int height{ 0 };
    int channels{ 0 };
    stbi_uc* pixels{ nullptr };

    ~ImageData()
    {
        if (pixels)
        {
            stbi_image_free(pixels);
        }

        pixels = nullptr;
        width = height = channels = 0;
    }

    bool is_valid()
    {
        return pixels != nullptr;
    }

    py::memoryview get_data()
    {
        return py::memoryview::from_memory(pixels, width * height * channels);
    }

};


ImageData load_image(const char *filename, int desired_channels = STBI_rgb_alpha)
{
    ImageData result{};
    result.pixels = stbi_load(filename, &result.width, &result.height, &result.channels, desired_channels);
    return result;
}




class SwapchainOutOfDateError : public std::exception {
private:
    char* message;

public:
    SwapchainOutOfDateError(char* msg) : message(msg) {}

    const char* what() const
    {
        return message;
    }
};


class Instance
{
public:
    Instance(const vkb::Instance& value) : handle(value) { }
    vkb::Instance handle;
};

template<typename T>
class VkHandle
{
public:
    T handle{ VK_NULL_HANDLE };

    operator T() const
    {
        return handle;
    }

    VkHandle& operator=(const T& value)
    {
        handle = value;
        return *this;
    }

    void clear()
    {
        handle = VK_NULL_HANDLE;
    }

    bool is_valid() const
    {
        return handle != VK_NULL_HANDLE;
    }
};


class Queue : public VkHandle<VkQueue> { };
class Surface : public VkHandle<VkSurfaceKHR> { };
class Image : public VkHandle<VkImage> { };
class ImageView : public VkHandle<VkImageView> { };
class Sampler : public VkHandle<VkImageView> { };
class RenderPass : public VkHandle<VkRenderPass> { };
class Framebuffer : public VkHandle<VkFramebuffer> { };
class CommandPool : public VkHandle<VkCommandPool> { };
class Semaphore : public VkHandle<VkSemaphore> { };
class Fence : public VkHandle<VkFence> { };


template<typename T>
class VKUWrapper
{
public:
    void clear()
    {
        value.clear();
    }

    void destroy()
    {
        value.destroy();
    }

    bool is_valid() const
    {
        return value.is_valid();
    }

    T value{};
};


/*
class Buffer : public VKUWrapper<vku::Buffer>
{
public:
    operator VkBuffer() const { return (VkBuffer)value; }

    void map_memory(VkDeviceSize size=VK_WHOLE_SIZE, VkDeviceSize offset=0)
    {
        auto vk_result = value.map_memory(size, offset);
        // TODO: handle error
    }

    void unmap_memory()
    {
        value.unmap_memory();
    }

    py::memoryview get_mapped()
    {
        return py::memoryview::from_memory(value.get_mapped(), value.get_size());
    }
};
*/

template<typename H, typename W>
std::vector<H> get_handles(const std::vector<W>& wrappers)
{
    std::vector<H> results(wrappers.size());
    for (uint32_t i = 0; i < wrappers.size(); ++i)
    {
        results[i] = wrappers[i];
    }
    return std::move(results);
}


template<typename H, typename W>
inline uint32_t accumulate_handles(const std::vector<W>& wrappers, std::vector<H>& handles)
{
    uint32_t result = handles.size();
    handles.resize(handles.size() + wrappers.size());
    for (uint32_t i = 0; i < handles.size(); i++)
    {
        handles[result + i] = wrappers[i];
    }

    return result;
}


template<typename W, typename H>
std::vector<W> get_wrappers(const std::vector<H> &handles)
{
    std::vector<W> results(handles.size());

    for (uint32_t i = 0; i < results.size(); ++i)
    {
        results[i].handle = (H)handles[i];
    }
    
    return std::move(results);
}



class PhysicalDevice 
{
public:
    PhysicalDevice(vkb::PhysicalDevice d) : physical_device(d) { }

    vkb::PhysicalDevice physical_device{};
};


void destroy_surface(Instance &instance, Surface &surface)
{
    vkDestroySurfaceKHR(instance.handle.instance, surface, nullptr);
}


class Window
{
public:
    Window(GLFWwindow* w) : window(w) { }

    GLFWwindow* window{ nullptr };

    void make_context_current()
    {
        glfwMakeContextCurrent(window);
    }

    Surface create_surface(Instance &instance)
    {
        VkSurfaceKHR result{ VK_NULL_HANDLE };

        auto vk_result = glfwCreateWindowSurface(instance.handle, window, nullptr, &result);

        return Surface{ result };
    }

    bool should_close()
    {
        return glfwWindowShouldClose(window);
    }

    void swap_buffers()
    {
        glfwSwapBuffers(window);
    }
};


// TODO: this should throw an exception which indicates exactly what the error is.
bool init()
{
    return glfwInit();
}


void _terminate()
{
    glfwTerminate();
}

void window_hint(int hint, int value)
{
    glfwWindowHint(hint, value);
}

void poll_events()
{
    glfwPollEvents();
}


Window create_window(int width, int height, const char *title)
{
    auto result = glfwCreateWindow(width, height, title, nullptr, nullptr);

    if (!result)
    {
        // TODO: error stuff.
    }

    return Window(result);
}



class InstanceBuilder
{
public:
    InstanceBuilder() : instance_builder() { }
    ~InstanceBuilder() { }

    vkb::InstanceBuilder instance_builder{};

    // Sets the name of the application. Defaults to "" if none is provided.
    InstanceBuilder& set_app_name(const char* app_name)
    {
        instance_builder.set_app_name(app_name);
        return *this;
    }

    // Sets the name of the engine. Defaults to "" if none is provided.
    InstanceBuilder& set_engine_name(const char* engine_name)
    {
        instance_builder.set_app_name(engine_name);
        return *this;
    }

    InstanceBuilder& request_validation_layers()
    {
        instance_builder.request_validation_layers();
        return *this;
    }

    InstanceBuilder& use_default_debug_messenger()
    {
        instance_builder.use_default_debug_messenger();
        return *this;
    }
    
    InstanceBuilder& require_api_version(uint32_t major, uint32_t minor, uint32_t patch)
    {
        instance_builder.require_api_version(major, minor, patch);
        return *this;
    }

    Instance build() const
    {
        auto instance_result = instance_builder.build();

        if (!instance_result)
        {
            throw std::runtime_error(instance_result.error().message());
        }

        return Instance(instance_result.value());
    }
};


void destroy_instance(Instance& instance)
{
    vkb::destroy_instance(instance.handle);
}



class PhysicalDeviceSelector
{
public:
    PhysicalDeviceSelector(Instance &instance) : physical_device_selector(instance.handle) { }

    vkb::PhysicalDeviceSelector physical_device_selector;

    PhysicalDeviceSelector& set_surface(Surface &surface)
    {
        physical_device_selector.set_surface(surface);
        return *this;
    }

    PhysicalDeviceSelector& prefer_gpu_device_type(vkb::PreferredDeviceType tp)
    {
        physical_device_selector.prefer_gpu_device_type(tp);
        return *this;
    }

    PhysicalDeviceSelector& allow_any_gpu_device_type(bool allow)
    {
        physical_device_selector.allow_any_gpu_device_type(allow);
        return *this;
    }

    PhysicalDevice select()
    {
        auto result = physical_device_selector.select();
        auto physical_device = result.value();
        return PhysicalDevice(physical_device);
    }

    std::vector<PhysicalDevice> select_devices()
    {
        auto result = physical_device_selector.select_devices();
        auto physical_devices = result.value();
        std::vector<PhysicalDevice> results;
        for (uint32_t i = 0; i < physical_devices.size(); ++i)
        {
            results.push_back(PhysicalDevice(physical_devices[i]));
        }
        return results;
    }
};


class Device
{
public:
    Device(const vkb::Device &d) : device(d) { }

    vkb::Device device;

    operator VkDevice() const { return (VkDevice)device; }

    void wait_idle()
    {
        auto vk_result = vkDeviceWaitIdle(device);
        // TODO: error handling.
    }

    Queue get_queue(vkb::QueueType queue_type)
    {
        auto res = device.get_queue(queue_type);

        // TODO: error handling.

        return Queue{ res.value() };
    }

    uint32_t get_queue_index(vkb::QueueType queue_type) const
    {
        auto res = device.get_queue_index(queue_type);

        // TODO: error handling
        return res.value();
    }
};

/*
template<typename W, typename H>
std::vector<W> get_vku_wrappers(const Device& device, ...ArgTypes, const std::vector<H>& handles)
{
    std::vector<W> results;
    results.reserve(handles.size());

    for (uint32_t i = 0; i < handles.size(); ++i)
    {
        results.push_back(W((VkDevice)device, handles[i]));
    }

    return std::move(results);
}
*/

class DeviceBuilder
{
public:
    DeviceBuilder(PhysicalDevice& physical_device) : device_builder(physical_device.physical_device) { }

    vkb::DeviceBuilder device_builder;

    Device build()
    {
        auto res = device_builder.build();
        // TODO: error handling.
        return Device(res.value());
    }
};


class Swapchain
{
public:
    Swapchain(vkb::Swapchain& s) : swapchain(s) { }

    vkb::Swapchain swapchain;

    operator VkSwapchainKHR() const { return swapchain.swapchain;  }

    VkFormat get_image_format() const
    {
        return swapchain.image_format;
    }

    void set_image_format(VkFormat value)
    {
        swapchain.image_format = value;
    }

    VkExtent2D get_extent()
    {
        return swapchain.extent;
    }

    std::vector<Image> get_images()
    {
        auto result = swapchain.get_images();
        // TODO: error handling
        auto images = result.value();
        return std::move(get_wrappers<Image>(images));
    }

    std::vector<ImageView> get_image_views()
    {
        auto result = swapchain.get_image_views();
        // TODO: error handling
        auto image_views = result.value();
        return std::move(get_wrappers<ImageView>(image_views));
    }

    void destroy_image_views(std::vector<ImageView> image_views)
    {
        std::vector<VkImageView> ivs = get_handles<VkImageView>(image_views);
        swapchain.destroy_image_views(ivs);
    }
};


class FramebufferCreateInfo
{
public:
    RenderPass render_pass{ VK_NULL_HANDLE };
    std::vector<ImageView> attachments{};
    uint32_t width{ 0 };
    uint32_t height{ 0 };
    uint32_t layers{ 0 };
};


Framebuffer create_framebuffer(Device &device, FramebufferCreateInfo &create_info)
{
    std::vector<VkImageView>  image_views = get_handles<VkImageView>(create_info.attachments);

    VkFramebufferCreateInfo ci{};

    ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.flags = 0;
    ci.attachmentCount = create_info.attachments.size();
   
    ci.pAttachments = image_views.data();
    ci.width = create_info.width;
    ci.height = create_info.height;
    ci.layers = create_info.layers;
    ci.renderPass = create_info.render_pass;

    Framebuffer result;
    auto vk_result = vkCreateFramebuffer(device.device, &ci, nullptr, &result.handle);

    // TODO: error handler.

    return result;
}

void destroy_swapchain(Device &device, Swapchain &swapchain)
{
    vkb::destroy_swapchain(swapchain.swapchain);
}


class SwapchainBuilder
{
public:
    SwapchainBuilder(const Device& d) : swapchain_builder(d.device) { }

    vkb::SwapchainBuilder swapchain_builder;

    SwapchainBuilder& set_old_swapchain(std::optional<Swapchain> s)
    {
        VkSwapchainKHR old_swapchain{ VK_NULL_HANDLE };
        if (s.has_value())
        {
            old_swapchain = s.value();
        }
        swapchain_builder.set_old_swapchain(old_swapchain);
        return *this;
    }

    Swapchain build()
    {
        auto res = swapchain_builder.build();
        return Swapchain(res.value());
    }

};


class AttachmentDescription
{
public:
    VkFormat format{ VK_FORMAT_A8B8G8R8_SRGB_PACK32 };
    VkSampleCountFlags samples{ VK_SAMPLE_COUNT_1_BIT };
    VkAttachmentLoadOp load_op{ VK_ATTACHMENT_LOAD_OP_DONT_CARE };
    VkAttachmentStoreOp store_op{ VK_ATTACHMENT_STORE_OP_DONT_CARE };
    VkAttachmentLoadOp stencil_load_op{ VK_ATTACHMENT_LOAD_OP_DONT_CARE };
    VkAttachmentStoreOp stencil_store_op{ VK_ATTACHMENT_STORE_OP_DONT_CARE };
    VkImageLayout initial_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VkImageLayout final_layout{ VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };

    void populate(VkAttachmentDescription& result)
    {
        result.flags = 0;
        result.format = format;
        result.samples = (VkSampleCountFlagBits)samples;
        result.loadOp = load_op;
        result.storeOp = store_op;
        result.stencilLoadOp = stencil_load_op;
        result.stencilStoreOp = stencil_store_op;
        result.initialLayout = initial_layout;
        result.finalLayout = final_layout;
    }
};


class AttachmentReference
{
public:
    uint32_t attachment{ 0 };
    VkImageLayout layout{ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    uint32_t index{ 0 };
};


class SubpassDescription
{
public:
    VkPipelineBindPoint pipeline_bind_point{ VK_PIPELINE_BIND_POINT_GRAPHICS };
    std::vector<AttachmentReference> color_attachments{};

    uint32_t color_attachments_start{ 0 };

    void populate(std::vector<VkAttachmentReference> &refs)
    {
        for (auto& a : color_attachments)
        {
            a.index = refs.size();
            VkAttachmentReference attachment_reference{};
            attachment_reference.attachment = a.attachment;
            attachment_reference.layout = a.layout;
            refs.push_back(attachment_reference);
        }
    }

    void populate(VkSubpassDescription& result, std::vector<VkAttachmentReference>& refs)
    {
        result.flags = 0;
        result.pipelineBindPoint = pipeline_bind_point;
        result.colorAttachmentCount = color_attachments.size();
        if (color_attachments.size())
        {
            result.pColorAttachments = &refs[color_attachments[0].index];
        }
    }
};


class SubpassDependency
{
public:
    uint32_t src_subpass{ 0 };
    uint32_t dst_subpass{ 0 };
    VkPipelineStageFlags src_stage_mask{ 0 };
    VkPipelineStageFlags dst_stage_mask{ 0 };
    VkAccessFlags src_access_mask{ 0 };
    VkAccessFlags dst_access_mask{ 0 };
    VkDependencyFlags dependency_flags{ 0 };

    void populate(VkSubpassDependency& result)
    {
        result.srcSubpass = src_subpass;
        result.dstSubpass = dst_subpass;
        result.srcStageMask = src_stage_mask;
        result.dstStageMask = dst_stage_mask;
        result.srcAccessMask = src_access_mask;
        result.dstAccessMask = dst_access_mask;
        result.dependencyFlags = dependency_flags;
    }
};


class RenderPassCreateInfo
{
public:
    std::vector<AttachmentDescription> attachments{};
    std::vector<SubpassDescription> subpasses{};
    std::vector<SubpassDependency> dependencies{};
};


template<typename VK, typename D>
std::vector<VK> populate(std::vector<D>& items)
{
    std::vector<VK> results(items.size());

    for (int i = 0; i < results.size(); ++i)
    {
        items[i].populate(results[i]);
    }

    return std::move(results);
}


RenderPass create_render_pass(Device& device, RenderPassCreateInfo& create_info)
{
    RenderPass result;
    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.flags = 0;

    std::vector<VkAttachmentDescription> attachments = populate<VkAttachmentDescription>(create_info.attachments);
    std::vector<VkSubpassDependency> dependencies = populate< VkSubpassDependency>(create_info.dependencies);

    ci.attachmentCount = attachments.size();
    ci.pAttachments = attachments.data();

    ci.dependencyCount = dependencies.size();
    ci.pDependencies = dependencies.data();

    std::vector<VkSubpassDescription> subpasses(create_info.subpasses.size());
    std::vector<VkAttachmentReference> refs{};
    for (uint32_t i = 0; i < subpasses.size(); i++)
    {
        create_info.subpasses[i].populate(refs);
    }

    for (uint32_t i = 0; i < subpasses.size(); i++)
    {
        create_info.subpasses[i].populate(subpasses[i], refs);
    }

    ci.subpassCount = subpasses.size();
    ci.pSubpasses = subpasses.data();


    auto vk_result = vkCreateRenderPass(device.device, &ci, nullptr, &result.handle);

    // TODO: error handling

    return result;
}


void destroy_device(Device& device)
{
    vkb::destroy_device(device.device);
}


void destroy_framebuffer(Device& device, Framebuffer& framebuffer)
{
    vkDestroyFramebuffer(device.device, framebuffer, nullptr);
}


void destroy_renderpass(Device& device, RenderPass& renderpass)
{
    vkDestroyRenderPass(device.device, renderpass, nullptr);
}

class CommandPoolCreateInfo
{
public:
    VkCommandPoolCreateFlags flags{ 0 };
    uint32_t queue_family_index{ 0 };
};


CommandPool create_command_pool(const Device& device, const CommandPoolCreateInfo& create_info)
{
    CommandPool result;
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = create_info.flags;
    ci.pNext = nullptr;
    ci.queueFamilyIndex = create_info.queue_family_index;
    auto vk_result = vkCreateCommandPool(device.device, &ci, nullptr, &result.handle);
    // TODO: error handling
    return result;
}

void destroy_command_pool(const Device& device, CommandPool& command_pool)
{
    vkDestroyCommandPool(device.device, command_pool, nullptr);
}

class CommandBufferAllocateInfo
{
public:
    uint32_t command_buffer_count{ 1 };
    CommandPool command_pool{ VK_NULL_HANDLE };
    VkCommandBufferLevel level{ VK_COMMAND_BUFFER_LEVEL_PRIMARY };
};


std::vector<vku::CommandBuffer> allocate_command_buffers(const Device& device, const CommandBufferAllocateInfo&i)
{
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandBufferCount = i.command_buffer_count;
    alloc_info.commandPool = i.command_pool.handle;
    alloc_info.level = i.level;
    std::vector<VkCommandBuffer> command_buffers(alloc_info.commandBufferCount);
    auto vk_result = vkAllocateCommandBuffers(device.device, &alloc_info, command_buffers.data());

    // TODO: error handling;

    std::vector<vku::CommandBuffer> results;
    results.reserve(command_buffers.size());

    for (const auto& cb : command_buffers)
    {
        results.emplace_back(device, cb, i.command_pool);
    }

    return std::move(results);

    // return std::move(get_vku_wrappers<vku::CommandBuffer>(device, command_pool, command_buffers));
}

Semaphore create_semaphore(const Device& device)
{
    Semaphore result{};
    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.flags = 0;
    create_info.pNext = nullptr;
    auto vk_result = vkCreateSemaphore(device.device, &create_info, nullptr, &result.handle);
    // TODO: error handling.
    return result;
}


void destroy_semaphore(const Device& device, Semaphore &semaphore)
{
    vkDestroySemaphore(device.device, semaphore, nullptr);
}

class FenceCreateInfo
{
public:
    VkFenceCreateFlags flags{  };
};

Fence create_fence(const Device& device, const FenceCreateInfo& create_info)
{
    Fence result{};
    VkFenceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ci.flags = create_info.flags;
    ci.pNext = nullptr;
    auto vk_result = vkCreateFence(device.device, &ci, nullptr, &result.handle);
    // TODO: ERROR HANDLING
    return result;
}

void destroy_fence(const Device& device, const Fence& fence)
{
    vkDestroyFence(device.device, fence, nullptr);
}

void wait_for_fences(const Device &device, std::vector<Fence> &fences, bool wait_for_all, uint64_t timeout)
{
    std::vector<VkFence> vkfences = get_handles<VkFence>(fences);
    auto vk_result = vkWaitForFences(device.device, vkfences.size(), vkfences.data(), wait_for_all, timeout);

    // TODO: error handling
}

uint32_t acquire_next_image(const Device &device, const Swapchain &swapchain, uint64_t timeout, const Semaphore &semaphore)
{
    uint32_t image_index{ 0 };
    VkResult result = vkAcquireNextImageKHR(device.device, swapchain.swapchain, timeout, semaphore, VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        throw SwapchainOutOfDateError("swapchain must be recreated.");
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    return image_index;
}

void reset_fences(const Device &device, const std::vector<Fence> &fences)
{
    std::vector<VkFence> vkfences = get_handles<VkFence>(fences);
    auto vk_result = vkResetFences(device.device, vkfences.size(), vkfences.data());

    // TODO: error handling
}

void reset_command_buffer(const vku::CommandBuffer& command_buffer, VkCommandBufferResetFlags flags)
{
    vkResetCommandBuffer(command_buffer, flags);
}


class SubmitInfo
{
public:
    std::vector<Semaphore> wait_semaphores{};
    std::vector<VkPipelineStageFlagBits> wait_dst_stage_masks{};
    std::vector<vku::CommandBuffer> command_buffers{};
    std::vector<Semaphore> signal_semaphores{};

    uint32_t wait_semaphores_start{ 0 };
    uint32_t wait_dst_stage_masks_start{ 0 };
    uint32_t command_buffers_start{ 0 };
    uint32_t signal_semaphores_start{ 0 };
};


void queue_submit(const Queue queue, std::vector<SubmitInfo> &submit_infos, const Fence &fence)
{
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = nullptr;

    std::vector<VkSubmitInfo> vk_submit_infos(submit_infos.size());

    // get a list of wait_semaphores
    std::vector<VkSemaphore> wait_semaphores{};

    // get a list of wait_dst_stage_masks
    std::vector<VkPipelineStageFlagBits> wait_dst_stage_masks{};

    // get a list of command_buffers
    std::vector<VkCommandBuffer> command_buffers{};

    // get a list of signal_semaphores
    std::vector<VkSemaphore> signal_semaphores{};

    for (uint32_t i = 0; i < submit_infos.size(); ++i)
    {
        submit_infos[i].wait_semaphores_start = accumulate_handles(submit_infos[i].wait_semaphores, wait_semaphores);
        submit_infos[i].wait_dst_stage_masks_start = accumulate_handles(submit_infos[i].wait_dst_stage_masks, wait_dst_stage_masks);
        submit_infos[i].command_buffers_start = accumulate_handles(submit_infos[i].command_buffers, command_buffers);
        submit_infos[i].signal_semaphores_start = accumulate_handles(submit_infos[i].signal_semaphores, signal_semaphores);
    }

    for (uint32_t i = 0; i < vk_submit_infos.size(); ++i)
    {
        vk_submit_infos[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        vk_submit_infos[i].pNext = nullptr;

        vk_submit_infos[i].commandBufferCount =  submit_infos[i].command_buffers.size();
        vk_submit_infos[i].pCommandBuffers = command_buffers.data() + submit_infos[i].command_buffers_start;

        vk_submit_infos[i].signalSemaphoreCount = submit_infos[i].signal_semaphores.size();
        vk_submit_infos[i].pSignalSemaphores = signal_semaphores.data() + submit_infos[i].signal_semaphores_start;

        vk_submit_infos[i].waitSemaphoreCount = submit_infos[i].wait_semaphores.size();
        vk_submit_infos[i].pWaitSemaphores = wait_semaphores.data() + submit_infos[i].wait_semaphores_start;
        vk_submit_infos[i].pWaitDstStageMask = (VkPipelineStageFlags *)(wait_dst_stage_masks.data() + submit_infos[i].wait_dst_stage_masks_start);

    }

    // populate vk_submit_infos
    auto vk_result = vkQueueSubmit(queue, vk_submit_infos.size(), vk_submit_infos.data(), fence);
    // TODO: error handling
}


class PresentInfo
{
public:
    std::vector<Semaphore> wait_semaphores{};
    std::vector<Swapchain> swapchains{};
    std::vector<uint32_t> image_indices{};
};


void queue_present(const Queue &queue, const PresentInfo &present_info)
{
    VkPresentInfoKHR pi{};

    std::vector<VkSwapchainKHR> swapchains = get_handles<VkSwapchainKHR>(present_info.swapchains);
    std::vector<VkSemaphore> wait_semaphores = get_handles<VkSemaphore>(present_info.wait_semaphores);

    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.pNext = nullptr;
    pi.pImageIndices = present_info.image_indices.data();
    pi.pSwapchains = swapchains.data();
    pi.swapchainCount = swapchains.size();
    pi.pWaitSemaphores = wait_semaphores.data();
    pi.waitSemaphoreCount = wait_semaphores.size();

    auto vk_result = vkQueuePresentKHR(queue, &pi);

    // TODO: error handlings
}


void begin_command_buffer(const vku::CommandBuffer& command_buffer)
{
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    auto vk_result = vkBeginCommandBuffer(command_buffer, &begin_info);
    // TODO: error handling
}


class ClearValue
{
public:
    VkClearValue value{};

    static ClearValue colorf(float r, float g, float b, float a)
    {
        VkClearValue v{ { { r, g, b, a } } };
        return ClearValue{ v };
    }

    operator VkClearValue() const { return value; }
};


class RenderPassBeginInfo
{
public:
    RenderPass render_pass{};
    Framebuffer framebuffer{};
    VkRect2D render_area{};
    std::vector<ClearValue> clear_values{};

    void populate(VkRenderPassBeginInfo& begin_info, std::vector<VkClearValue> &vk_clear_values) const
    {
        begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin_info.pNext = nullptr;
        begin_info.framebuffer = framebuffer;
        begin_info.renderPass = render_pass;
        begin_info.renderArea = render_area;

        vk_clear_values  = get_handles<VkClearValue>(clear_values);

        begin_info.clearValueCount = vk_clear_values.size();
        begin_info.pClearValues = vk_clear_values.data();
    }
};


void cmd_set_viewport(const vku::CommandBuffer &command_buffer, uint32_t first_viewport, uint32_t viewport_count, std::vector<VkViewport> &viewports)
{
    vkCmdSetViewport(command_buffer, first_viewport, viewport_count, viewports.data());
}


void cmd_set_scissor(const vku::CommandBuffer& command_buffer, uint32_t first_scissor, uint32_t scissor_count, std::vector<VkRect2D>& scissors)
{
    vkCmdSetScissor(command_buffer, first_scissor, scissor_count, scissors.data());
}

void cmd_begin_render_pass(const vku::CommandBuffer& command_buffer, const RenderPassBeginInfo &info, VkSubpassContents contents)
{
    std::vector<VkClearValue> vk_clear_values;
    VkRenderPassBeginInfo begin_info;
    info.populate(begin_info, vk_clear_values);
    vkCmdBeginRenderPass(command_buffer, &begin_info, contents);
}

void cmd_end_render_pass(const vku::CommandBuffer& command_buffer)
{
    vkCmdEndRenderPass(command_buffer);
}


void end_command_buffer(const vku::CommandBuffer& command_buffer)
{
    vkEndCommandBuffer(command_buffer);
}

void cmd_bind_pipeline(const vku::CommandBuffer& command_buffer, VkPipelineBindPoint bind_point, const vku::Pipeline &pipeline)
{
    vkCmdBindPipeline(command_buffer, bind_point, pipeline);
}


void cmd_bind_vertex_buffers(const vku::CommandBuffer& command_buffer, uint32_t first_binding, std::vector<vku::Buffer> buffers, std::vector<VkDeviceSize> offsets)
{
    // TODO: boudns checking for everything.
    std::vector<VkBuffer> bufs = get_handles<VkBuffer>(buffers);
    vkCmdBindVertexBuffers(command_buffer, first_binding, bufs.size(), bufs.data(), offsets.data());
}


class PipelineLayoutBuilder
{
public:
    PipelineLayoutBuilder(Device& device) : pipeline_layout_builder(device.device) { }

    PipelineLayoutBuilder& add_descriptor_set(const vku::DescriptorSetLayout& layout)
    {
        pipeline_layout_builder.add_descriptor_set(layout);
        return *this;
    }

    vku::PipelineLayout build() const
    {
        // PipelineLayout pipeline_layout(pipeline_layout_builder.device, VK_NULL_HANDLE)

        auto result = pipeline_layout_builder.build();

        // TODO: error handling.

        return result.get_value();
    }

    vku::PipelineLayoutBuilder pipeline_layout_builder;
};


vku::ShaderModule create_shader_module(const Device &device, py::buffer data)
{
    py::buffer_info info = data.request();
    auto shader_module_result = vku::create_shader_module(device.device, reinterpret_cast<const uint32_t *>(info.ptr), info.size * info.itemsize);
    
    // TODO: error handling;

    return shader_module_result.get_value();
}


class GraphicsPipelineBuilder
{
public:
    GraphicsPipelineBuilder(const Device& device) : graphics_pipeline_builder(device.device) { }

    GraphicsPipelineBuilder& add_shader_stage(VkShaderStageFlagBits stage, vku::ShaderModule module)
    {
        graphics_pipeline_builder.add_shader_stage(stage, module, "main");
        return *this;
    }

    GraphicsPipelineBuilder& add_scissor(VkRect2D scissor)
    {
        graphics_pipeline_builder.add_scissor(scissor);
        return *this;
    }

    GraphicsPipelineBuilder& add_viewport(VkViewport viewport)
    {
        graphics_pipeline_builder.add_viewport(viewport);
        return *this;
    }

    GraphicsPipelineBuilder& add_dynamic_state(VkDynamicState s)
    {
        graphics_pipeline_builder.add_dynamic_state(s);
        return *this;
    }

    GraphicsPipelineBuilder& set_pipeline_layout(vku::PipelineLayout &p)
    {
        graphics_pipeline_builder.set_pipeline_layout(p);
        return *this;
    }

    GraphicsPipelineBuilder& set_render_pass(RenderPass& rp)
    {
        graphics_pipeline_builder.set_render_pass(rp.handle);
        return *this;
    }

    GraphicsPipelineBuilder& add_color_blend_attachment(VkPipelineColorBlendAttachmentState &c)
    {
        graphics_pipeline_builder.add_color_blend_attachment(c);
        return *this;
    }

    GraphicsPipelineBuilder& add_vertex_binding(VkVertexInputBindingDescription &desc)
    {
        graphics_pipeline_builder.add_vertex_binding(desc);
        return *this;
    }
    
    GraphicsPipelineBuilder& add_vertex_attributes(std::vector<VkVertexInputAttributeDescription>& descs)
    {
        graphics_pipeline_builder.add_vertex_attributes(descs);
        return *this;
    }

    GraphicsPipelineBuilder& set_viewport_count(uint32_t c)
    {
        graphics_pipeline_builder.set_viewport_count(c);
        return *this;
    }

    GraphicsPipelineBuilder& set_scissor_count(uint32_t c)
    {
        graphics_pipeline_builder.set_scissor_count(c);
        return *this;
    }

    GraphicsPipelineBuilder& set_cull_mode(VkCullModeFlagBits mode)
    {
        graphics_pipeline_builder.set_cull_mode(mode);
        return *this;
    }

    GraphicsPipelineBuilder& set_front_face(VkFrontFace f)
    {
        graphics_pipeline_builder.set_front_face(f);
        return *this;
    }

    vku::Pipeline build()
    {
        auto res = graphics_pipeline_builder.build();

        // TODO: error handling

        return res.get_value();
    }

    vku::GraphicsPipelineBuilder graphics_pipeline_builder;
};


class BufferFactory
{
public:
    BufferFactory(const Device& device, const PhysicalDevice& physical_device) : buffer_factory(device.device, physical_device.physical_device) { }

    vku::Buffer build(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_properties)
    {
        auto res = buffer_factory.build(size, usage, memory_properties);

        // TODO; error handling
        if (!res)
        {
            throw std::runtime_error("failed to build buffer");
        }

        return res.get_value();;
    }

    vku::BufferFactory buffer_factory;
};


std::set<std::string> get_supported_extensions() {
    uint32_t count;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr); //get number of extensions
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()); //populate buffer
    std::set<std::string> results;
    for (auto& extension : extensions) {
        results.insert(extension.extensionName);
    }
    return results;
}


std::tuple<uint32_t, uint32_t, uint32_t> get_vulkan_version()
{
    uint32_t instanceVersion = VK_API_VERSION_1_0;
    auto FN_vkEnumerateInstanceVersion = PFN_vkEnumerateInstanceVersion(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (vkEnumerateInstanceVersion) {
        vkEnumerateInstanceVersion(&instanceVersion);
    }

    // 3 macros to extract version info
    uint32_t major = VK_VERSION_MAJOR(instanceVersion);
    uint32_t minor = VK_VERSION_MINOR(instanceVersion);
    uint32_t patch = VK_VERSION_PATCH(instanceVersion);

    return std::tuple<uint32_t, uint32_t, uint32_t>(major, minor, patch);
}

VkPhysicalDeviceProperties get_physical_device_properties(const PhysicalDevice& physical_device)
{
    VkPhysicalDeviceProperties result{};
    vkGetPhysicalDeviceProperties(physical_device.physical_device, &result);
    return result;
}


void cmd_draw(const vku::CommandBuffer &command_buffer, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
    vkCmdDraw(command_buffer, vertex_count, instance_count, first_vertex, first_instance);
}


class SingleTimeCommandExecutor
{
public:
    SingleTimeCommandExecutor(const Device& device, const CommandPool& command_pool, const Queue &queue) : executor(device.device, command_pool.handle, queue.handle) { }

    vku::CommandBuffer enter()
    {
        auto result = executor.enter();

        // TODO: error handling

        return vku::CommandBuffer{ result.get_value() };
    }

    Fence exit()
    {
        auto result = executor.exit();
        
        // TODO: error handling

        return Fence{ result.get_value() };
    }

    void wait()
    {
        auto result = executor.wait();

        if (result.has_value())
        {
            // handle error.
        }
    }

    void destroy()
    {
        executor.destroy();
    }

    vku::SingleTimeCommandExecutor executor{};
};


void cmd_copy_buffer(const vku::CommandBuffer &command_buffer, const vku::Buffer &src_buffer, const vku::Buffer &dst_buffer, const std::vector<VkBufferCopy> &regions)
{
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, regions.size(), regions.data());
}


void cmd_bind_index_buffer(const vku::CommandBuffer& command_buffer, const vku::Buffer &buffer, VkDeviceSize offset, VkIndexType index_type)
{
    vkCmdBindIndexBuffer(command_buffer, buffer, offset, index_type);
}


void cmd_draw_indexed(const vku::CommandBuffer& command_buffer, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
    vkCmdDrawIndexed(command_buffer, index_count, instance_count, first_index, vertex_offset, first_instance);
}


template<typename T>
py::class_<T> register_vku_class(py::module m, const char *name)
{
    return py::class_<T>(m, name)
        .def("destroy", &T::destroy)
        .def("is_valid", &T::is_valid);
}

class DescriptorPoolBuilder
{
public:
    DescriptorPoolBuilder(const Device& d) : descriptor_pool_builder((VkDevice)d) { }

    DescriptorPoolBuilder& add_descriptor_sets(const vku::DescriptorSetLayout& set_layout, uint32_t count = 1)
    {
        descriptor_pool_builder.add_descriptor_sets(set_layout, count);
        return *this;
    }

    vku::DescriptorPool build()
    {
        auto result = descriptor_pool_builder.build();

        if (!result)
        {
            // TODO: error handling
        }

        return result.get_value();
    }
private:
    vku::DescriptorPoolBuilder descriptor_pool_builder;
};

/*

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

        inline Result<VkDescriptorPool> build() const
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

            return Result<VkDescriptorPool>(result);
        }
    private:
        VkDevice device{ VK_NULL_HANDLE };
        uint32_t max_sets{ 0 };
        DescriptorTypeCounts descriptor_type_counts{};
    };
*/


class DescriptorSetLayoutBuilder
{
public:
    DescriptorSetLayoutBuilder(const Device& device) : builder(device)
    {
    }

    DescriptorSetLayoutBuilder& add_binding(uint32_t binding, VkDescriptorType descriptor_type, uint32_t descriptor_count, VkPipelineStageFlags stage_flags)
    {
        builder.add_binding(binding, descriptor_type, descriptor_count, stage_flags);
        return *this;
    }

    vku::DescriptorSetLayout build() const
    {
        auto result = builder.build();

        // TODO: error handling

        return result.get_value();
    }
private:
    vku::DescriptorSetLayoutBuilder builder;
};


class DescriptorSetBuilder
{
public:
    DescriptorSetBuilder(const Device& d, const vku::DescriptorPool& dp, vku::DescriptorSetLayout l) : builder(d, dp, l) { }

    DescriptorSetBuilder& write_uniform_buffer(uint32_t binding, uint32_t array_element, vku::Buffer buffer, VkDeviceSize offset, VkDeviceSize range)
    {
        builder.write_uniform_buffer(binding, array_element, buffer, offset, range);
        return *this;
    }

    inline DescriptorSetBuilder& write_combined_image_sampler(uint32_t binding, uint32_t array_element, VkSampler sampler, VkImageView image_view, VkImageLayout image_layout)
    {
        builder.write_combined_image_sampler(binding, array_element, sampler, image_view, image_layout);
        return *this;
    }

    inline vku::DescriptorSet build() const
    {
        auto result = builder.build();

        // TODO: error handling

        return result.get_value();
    }
private:
    vku::DescriptorSetBuilder builder;
};


void cmd_bind_descriptor_sets(const vku::CommandBuffer &command_buffer, VkPipelineBindPoint bind_point, vku::PipelineLayout layout, uint32_t first_binding, const std::vector<vku::DescriptorSet> &descriptor_sets)
{
    std::vector<VkDescriptorSet> ds = get_handles<VkDescriptorSet>(descriptor_sets);
    vkCmdBindDescriptorSets(command_buffer, bind_point, layout, first_binding, ds.size(), ds.data(), 0, nullptr);
}


PYBIND11_MODULE(pyvku, m) {
    m.doc() = "vku test"; // optional module docstring

    m.attr("CLIENT_API") = py::int_(GLFW_CLIENT_API);
    m.attr("NO_API") = py::int_(GLFW_NO_API);
    m.attr("SUBPASS_EXTERNAL") = py::int_(VK_SUBPASS_EXTERNAL);
    m.attr("UINT64_MAX") = py::int_(UINT64_MAX);

    m.def("destroy_instance", destroy_instance)
        .def("create_window", create_window)
        .def("init", init)
        .def("terminate", _terminate)
        .def("window_hint", window_hint)
        .def("poll_events", poll_events)
        .def("destroy_surface", destroy_surface)
        .def("create_framebuffer", create_framebuffer)
        .def("create_render_pass", create_render_pass)
        .def("destroy_swapchain", destroy_swapchain)
        .def("destroy_device", destroy_device)
        .def("destroy_framebuffer", destroy_framebuffer)
        .def("destroy_renderpass", destroy_renderpass)
        .def("destroy_command_pool", destroy_command_pool)
        .def("create_command_pool", create_command_pool)
        .def("create_semaphore", create_semaphore)
        .def("destroy_semaphore", destroy_semaphore)
        .def("create_fence", create_fence)
        .def("destroy_fence", destroy_fence)
        .def("allocate_command_buffers", allocate_command_buffers)
        .def("wait_for_fences", wait_for_fences)
        .def("acquire_next_image", acquire_next_image)
        .def("reset_fences", reset_fences)
        .def("reset_command_buffer", reset_command_buffer)
        .def("queue_submit", queue_submit)
        .def("queue_present", queue_present)
        .def("begin_command_buffer", begin_command_buffer)
        .def("cmd_set_viewport", cmd_set_viewport)
        .def("cmd_set_scissor", cmd_set_scissor)
        .def("cmd_begin_render_pass", cmd_begin_render_pass)
        .def("cmd_end_render_pass", cmd_end_render_pass)
        .def("end_command_buffer", end_command_buffer)
        .def("create_shader_module", create_shader_module)
        .def("cmd_bind_pipeline", cmd_bind_pipeline)
        .def("cmd_begin_render_pass", cmd_begin_render_pass)
        .def("get_supported_extensions", get_supported_extensions)
        .def("get_vulkan_version", get_vulkan_version)
        .def("get_physical_device_properties", get_physical_device_properties)
        .def("cmd_bind_vertex_buffers", cmd_bind_vertex_buffers)
        .def("cmd_draw", cmd_draw)
        .def("cmd_copy_buffer", cmd_copy_buffer)
        .def("cmd_bind_index_buffer", cmd_bind_index_buffer)
        .def("cmd_draw_indexed", cmd_draw_indexed)
        .def("load_image", load_image)
        .def("cmd_bind_descriptor_sets", cmd_bind_descriptor_sets);

    py::class_<InstanceBuilder>(m, "InstanceBuilder")
        .def(py::init<>())
        .def("set_app_name", &InstanceBuilder::set_app_name)
        .def("set_engine_name", &InstanceBuilder::set_engine_name)
        .def("request_validation_layers", &InstanceBuilder::request_validation_layers)
        .def("use_default_debug_messenger", &InstanceBuilder::use_default_debug_messenger)
        .def("require_api_version", &InstanceBuilder::require_api_version)
        .def("build", &InstanceBuilder::build);

    py::class_<Instance>(m, "Instance");

    py::class_<Window>(m, "Window")
        .def("make_context_current", &Window::make_context_current)
        .def("create_surface", &Window::create_surface)
        .def("should_close", &Window::should_close)
        .def("swap_buffers", &Window::swap_buffers);

    py::class_<Surface>(m, "Surface");
    py::class_<PhysicalDeviceSelector>(m, "PhysicalDeviceSelector")
        .def(py::init<Instance>())
        .def("set_surface", &PhysicalDeviceSelector::set_surface)
        .def("prefer_gpu_device_type", &PhysicalDeviceSelector::prefer_gpu_device_type)
        .def("allow_any_gpu_device_type", &PhysicalDeviceSelector::allow_any_gpu_device_type)
        .def("select", &PhysicalDeviceSelector::select)
        .def("select_devices", &PhysicalDeviceSelector::select_devices);

    py::class_<PhysicalDevice>(m, "PhysicalDevice");

    py::class_<Device>(m, "Device")
        .def("get_queue", &Device::get_queue)
        .def("get_queue_index", &Device::get_queue_index)
        .def("wait_idle", &Device::wait_idle);

    py::class_<DeviceBuilder>(m, "DeviceBuilder")
        .def(py::init<PhysicalDevice>())
        .def("build", &DeviceBuilder::build);

    py::class_<Swapchain>(m, "Swapchain")
        .def("get_images", &Swapchain::get_images)
        .def("get_image_views", &Swapchain::get_image_views)
        .def_property("image_format", &Swapchain::get_image_format, &Swapchain::set_image_format)
        .def_property_readonly("extent", &Swapchain::get_extent)
        .def("destroy_image_views", &Swapchain::destroy_image_views);

    py::class_<SwapchainBuilder>(m, "SwapchainBuilder")
        .def(py::init<Device>())
        .def("set_old_swapchain", &SwapchainBuilder::set_old_swapchain)
        .def("build", &SwapchainBuilder::build);

    py::enum_<vkb::QueueType>(m, "QueueType")
        .value("compute", vkb::QueueType::compute)
        .value("graphics", vkb::QueueType::graphics)
        .value("present", vkb::QueueType::present)
        .value("transfer", vkb::QueueType::transfer)
        .export_values();

    py::class_<Queue>(m, "Queue");
    py::class_<Image>(m, "Image");
    py::class_<ImageView>(m, "ImageView");

    register_vku_class<vku::ShaderModule>(m, "ShaderModule");
    register_vku_class<vku::DescriptorPool>(m, "DescriptorPool");

    py::class_<FramebufferCreateInfo>(m, "FramebufferCreateInfo")
        .def(py::init<>())
        .def_readwrite("render_pass", &FramebufferCreateInfo::render_pass)
        .def_readwrite("attachments", &FramebufferCreateInfo::attachments)
        .def_readwrite("layers", &FramebufferCreateInfo::layers)
        .def_readwrite("width", &FramebufferCreateInfo::width)
        .def_readwrite("height", &FramebufferCreateInfo::height);
    
    py::class_<AttachmentDescription>(m, "AttachmentDescription")
        .def(py::init<>())
        .def_readwrite("format", &AttachmentDescription::format)
        .def_readwrite("samples", &AttachmentDescription::samples)
        .def_readwrite("load_op", &AttachmentDescription::load_op)
        .def_readwrite("store_op", &AttachmentDescription::store_op)
        .def_readwrite("stencil_load_op", &AttachmentDescription::stencil_load_op)
        .def_readwrite("stencil_store_op", &AttachmentDescription::stencil_store_op)
        .def_readwrite("initial_layout", &AttachmentDescription::initial_layout)
        .def_readwrite("final_layout", &AttachmentDescription::final_layout);

    py::class_<AttachmentReference>(m, "AttachmentReference")
        .def(py::init<>())
        .def_readwrite("attachment", &AttachmentReference::attachment)
        .def_readwrite("layout", &AttachmentReference::layout);

    py::enum_<VkFormat>(m, "Format")
        .value("UNDEFINED", VK_FORMAT_UNDEFINED)
        .value("R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT);

    py::enum_<VkImageLayout>(m, "ImageLayout")
        .value("UNDEFINED", VK_IMAGE_LAYOUT_UNDEFINED)
        .value("PRESENT_SRC_KHR", VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        .value("COLOR_ATTACHMENT_OPTIMAL", VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    py::enum_<VkAttachmentLoadOp>(m, "AttachmentLoadOp")
        .value("DONT_CARE", VK_ATTACHMENT_LOAD_OP_DONT_CARE)
        .value("CLEAR", VK_ATTACHMENT_LOAD_OP_CLEAR)
        .value("LOAD", VK_ATTACHMENT_LOAD_OP_LOAD);

    py::enum_<VkAttachmentStoreOp>(m, "AttachmentStoreOp")
        .value("DONT_CARE", VK_ATTACHMENT_STORE_OP_DONT_CARE)
        .value("STORE", VK_ATTACHMENT_STORE_OP_STORE);

    py::enum_<VkSampleCountFlagBits>(m, "SampleCount", py::arithmetic())
        .value("_1_BIT", VK_SAMPLE_COUNT_1_BIT);

    py::enum_<VkPipelineBindPoint>(m, "PipelineBindPoint")
        .value("GRAPHICS", VK_PIPELINE_BIND_POINT_GRAPHICS);

    py::class_< SubpassDescription>(m, "SubpassDescription")
        .def(py::init<>())
        .def_readwrite("pipeline_bind_point", &SubpassDescription::pipeline_bind_point)
        .def_readwrite("color_attachments", &SubpassDescription::color_attachments);

    py::class_<SubpassDependency>(m, "SubpassDependency")
        .def(py::init<>())
        .def_readwrite("src_subpass", &SubpassDependency::src_subpass)
        .def_readwrite("dst_subpass", &SubpassDependency::dst_subpass)
        .def_readwrite("src_stage_mask", &SubpassDependency::src_stage_mask)
        .def_readwrite("dst_stage_mask", &SubpassDependency::dst_stage_mask)
        .def_readwrite("src_access_mask", &SubpassDependency::src_access_mask)
        .def_readwrite("dst_access_mask", &SubpassDependency::dst_access_mask)
        .def_readwrite("dependency_flags", &SubpassDependency::dependency_flags);

    py::enum_< VkPipelineStageFlagBits>(m, "PipelineStage", py::arithmetic())
        .value("COLOR_ATTACHMENT_OUTPUT_BIT", VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    py::enum_< VkAccessFlagBits>(m, "Access", py::arithmetic())
        .value("COLOR_ATTACHMENT_READ_BIT", VK_ACCESS_COLOR_ATTACHMENT_READ_BIT)
        .value("COLOR_ATTACHMENT_WRITE_BIT", VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);


    py::class_< RenderPassCreateInfo>(m, "RenderPassCreateInfo")
        .def(py::init<>())
        .def_readwrite("subpasses", &RenderPassCreateInfo::subpasses)
        .def_readwrite("attachments", &RenderPassCreateInfo::attachments)
        .def_readwrite("dependencies", &RenderPassCreateInfo::dependencies);


    py::class_<RenderPass>(m, "RenderPass");

    py::class_<VkExtent2D>(m, "Extent2D")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t>())
        .def_readwrite("width", &VkExtent2D::width)
        .def_readwrite("height", &VkExtent2D::height);

    py::class_<VkOffset2D>(m, "Offset2D")
        .def(py::init<>())
        .def(py::init<int32_t, int32_t>())
        .def_readwrite("x", &VkOffset2D::x)
        .def_readwrite("y", &VkOffset2D::y);

    py::class_<Framebuffer>(m, "Framebuffer");
    py::class_<Fence>(m, "Fence");

    register_vku_class<vku::DescriptorSetLayout>(m, "DescriptorSetLayout");
    register_vku_class<vku::DescriptorSet>(m, "DescriptorSet");

    register_vku_class<vku::Pipeline>(m, "Pipeline")
        .def("get_layout", [&](vku::Pipeline& self) -> vku::PipelineLayout {
            return vku::PipelineLayout(self.get_device(), self.get_layout());
        });

    register_vku_class<vku::PipelineLayout>(m, "PipelineLayout");

    py::class_<FenceCreateInfo>(m, "FenceCreateInfo")
        .def(py::init<>())
        .def(py::init<VkFenceCreateFlags>())
        .def_readwrite("flags", &FenceCreateInfo::flags);
    py::class_<Semaphore>(m, "Semaphore");

    py::enum_<VkFenceCreateFlagBits>(m, "FenceCreate")
        .value("SIGNALED_BIT", VK_FENCE_CREATE_SIGNALED_BIT);

    py::class_<CommandPoolCreateInfo>(m, "CommandPoolCreateInfo")
        .def(py::init<>())
        .def_readwrite("flags", &CommandPoolCreateInfo::flags)
        .def_readwrite("queue_family_index", &CommandPoolCreateInfo::queue_family_index);

    py::enum_<VkCommandPoolCreateFlagBits>(m, "CommandPoolCreate")
        .value("RESET_COMMAND_BUFFER_BIT", VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    
    py::class_<CommandPool>(m, "CommandPool");

    py::class_< CommandBufferAllocateInfo>(m, "CommandBufferAllocateInfo")
        .def(py::init<>())
        .def_readwrite("level", &CommandBufferAllocateInfo::level)
        .def_readwrite("command_pool", &CommandBufferAllocateInfo::command_pool)
        .def_readwrite("command_buffer_count", &CommandBufferAllocateInfo::command_buffer_count);

    py::enum_<VkCommandBufferLevel>(m, "CommandBufferLevel")
        .value("PRIMARY", VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        .value("SECONDARY", VK_COMMAND_BUFFER_LEVEL_SECONDARY);

    register_vku_class<vku::CommandBuffer>(m, "CommandBuffer");

    py::register_exception<SwapchainOutOfDateError>(m, "SwapchainOutOfDateError");

    py::enum_< VkCommandBufferResetFlagBits>(m, "CommandBufferReset")
        .value("RELEASE_RESOURCES_BIT", VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

    py::class_<SubmitInfo>(m, "SubmitInfo")
        .def(py::init<>())
        .def_readwrite("command_buffers", &SubmitInfo::command_buffers)
        .def_readwrite("signal_semaphores", &SubmitInfo::signal_semaphores)
        .def_readwrite("wait_dst_stage_masks", &SubmitInfo::wait_dst_stage_masks)
        .def_readwrite("wait_semaphores", &SubmitInfo::wait_semaphores);

    py::class_<PresentInfo>(m, "PresentInfo")
        .def(py::init<>())
        .def_readwrite("image_indices", &PresentInfo::image_indices)
        .def_readwrite("swapchains", &PresentInfo::swapchains)
        .def_readwrite("wait_semaphores", &PresentInfo::wait_semaphores);

    py::class_<ClearValue>(m, "ClearValue")
        .def(py::init<>())
        .def_static("colorf", &ClearValue::colorf);

    py::class_< RenderPassBeginInfo>(m, "RenderPassBeginInfo")
        .def(py::init<>())
        .def_readwrite("render_pass", &RenderPassBeginInfo::render_pass)
        .def_readwrite("framebuffer", &RenderPassBeginInfo::framebuffer)
        .def_readwrite("render_area", &RenderPassBeginInfo::render_area)
        .def_readwrite("clear_values", &RenderPassBeginInfo::clear_values);

    py::class_<VkRect2D>(m, "Rect2D")
        .def(py::init<>())
        .def(py::init<VkOffset2D, VkExtent2D>())
        .def_readwrite("extent", &VkRect2D::extent)
        .def_readwrite("offset", &VkRect2D::offset);


    py::class_<VkViewport>(m, "Viewport")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, float>())
        .def_readwrite("x", &VkViewport::x)
        .def_readwrite("y", &VkViewport::y)
        .def_readwrite("width", &VkViewport::width)
        .def_readwrite("height", &VkViewport::height)
        .def_readwrite("min_depth", &VkViewport::minDepth)
        .def_readwrite("max_depth", &VkViewport::maxDepth);


    py::enum_<VkSubpassContents>(m, "SubpassContents")
        .value("INLINE", VK_SUBPASS_CONTENTS_INLINE)
        .value("SECONDARY_COMMAND_BUFFERS", VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

    py::class_<VkVertexInputBindingDescription>(m, "VertexInputBindingDescription")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, VkVertexInputRate>())
        .def_readwrite("binding", &VkVertexInputBindingDescription::binding)
        .def_readwrite("input_rate", &VkVertexInputBindingDescription::inputRate)
        .def_readwrite("stride", &VkVertexInputBindingDescription::stride);

    py::class_<VkVertexInputAttributeDescription>(m, "VertexInputAttributeDescription")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, VkFormat, uint32_t>())
        .def_readwrite("binding", &VkVertexInputAttributeDescription::binding)
        .def_readwrite("format", &VkVertexInputAttributeDescription::format)
        .def_readwrite("location", &VkVertexInputAttributeDescription::location)
        .def_readwrite("offset", &VkVertexInputAttributeDescription::offset);

    py::enum_< VkVertexInputRate>(m, "VertexInputRate")
        .value("VERTEX", VK_VERTEX_INPUT_RATE_VERTEX)
        .value("INSTANCE", VK_VERTEX_INPUT_RATE_INSTANCE);

    py::class_<PipelineLayoutBuilder>(m, "PipelineLayoutBuilder")
        .def(py::init<Device>())
        .def("add_descriptor_set", &PipelineLayoutBuilder::add_descriptor_set)
        .def("build", &PipelineLayoutBuilder::build);

    py::class_< GraphicsPipelineBuilder>(m, "GraphicsPipelineBuilder")
        .def(py::init<Device>())
        .def("add_color_blend_attachment", &GraphicsPipelineBuilder::add_color_blend_attachment)
        .def("add_dynamic_state", &GraphicsPipelineBuilder::add_dynamic_state)
        .def("add_scissor", &GraphicsPipelineBuilder::add_scissor)
        .def("add_shader_stage", &GraphicsPipelineBuilder::add_shader_stage)
        .def("add_vertex_attributes", &GraphicsPipelineBuilder::add_vertex_attributes)
        .def("add_vertex_binding", &GraphicsPipelineBuilder::add_vertex_binding)
        .def("add_viewport", &GraphicsPipelineBuilder::add_viewport)
        .def("set_pipeline_layout", &GraphicsPipelineBuilder::set_pipeline_layout)
        .def("set_render_pass", &GraphicsPipelineBuilder::set_render_pass)
        .def("set_viewport_count", &GraphicsPipelineBuilder::set_viewport_count)
        .def("set_scissor_count", &GraphicsPipelineBuilder::set_scissor_count)
        .def("build", &GraphicsPipelineBuilder::build)
        .def("set_front_face", &GraphicsPipelineBuilder::set_front_face)
        .def("set_cull_mode", &GraphicsPipelineBuilder::set_cull_mode);

    py::enum_<VkFrontFace>(m, "FrontFace")
        .value("CLOCKWISE", VK_FRONT_FACE_CLOCKWISE)
        .value("COUNTERCLOCKWISE", VK_FRONT_FACE_COUNTER_CLOCKWISE);

    py::enum_<VkCullModeFlagBits>(m, "CullMode")
        .value("BACK_BIT", VK_CULL_MODE_BACK_BIT)
        .value("FRONT_AND_BACK", VK_CULL_MODE_FRONT_AND_BACK)
        .value("FRONT_BIT", VK_CULL_MODE_FRONT_BIT)
        .value("NONE", VK_CULL_MODE_NONE);

    py::class_< BufferFactory>(m, "BufferFactory")
        .def(py::init<Device, PhysicalDevice>())
        .def("build", &BufferFactory::build);

    py::class_<vku::Buffer>(m, "Buffer")
        .def("destroy", &vku::Buffer::destroy)
        .def("is_valid", &vku::Buffer::is_valid)
        .def("map_memory", [&](vku::Buffer& buffer, VkDeviceSize size, VkDeviceSize offset) {
            auto vk_result = buffer.map_memory(size, offset);
            // TODO: error handling.
        }, py::arg("size") = VK_WHOLE_SIZE, py::arg("offset") = 0)
        .def("unmap_memory", &vku::Buffer::unmap_memory)
        .def("get_mapped", [&](vku::Buffer& buffer) -> py::memoryview {
            return py::memoryview::from_memory(buffer.get_mapped(), buffer.get_size());
         })
         .def("get_size", &vku::Buffer::get_size);

    py::enum_<VkBufferUsageFlagBits>(m, "BufferUsage", py::arithmetic())
        .value("VERTEX_BUFFER_BIT", VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        .value("INDEX_BUFFER_BIT", VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        .value("TRANSFER_DST_BIT", VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        .value("TRANSFER_SRC_BIT", VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        .value("UNIFORM_BUFFER_BIT", VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    py::enum_<VkMemoryPropertyFlagBits>(m, "MemoryProperty", py::arithmetic())
        .value("HOST_COHERENT_BIT", VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        .value("HOST_VISIBLE_BIT", VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        .value("DEVICE_LOCAL_BIT", VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    py::class_< VkPipelineColorBlendAttachmentState>(m, "PipelineColorBlendAttachmentState")
        .def(py::init<>())
        .def_readwrite("blend_enable", &VkPipelineColorBlendAttachmentState::blendEnable)
        .def_readwrite("color_write_mask", &VkPipelineColorBlendAttachmentState::colorWriteMask);


    py::enum_<VkColorComponentFlagBits>(m, "ColorComponent", py::arithmetic())
        .value("R_BIT", VK_COLOR_COMPONENT_R_BIT)
        .value("G_BIT", VK_COLOR_COMPONENT_G_BIT)
        .value("B_BIT", VK_COLOR_COMPONENT_B_BIT)
        .value("A_BIT", VK_COLOR_COMPONENT_A_BIT);

    py::enum_<VkShaderStageFlagBits>(m, "ShaderStage", py::arithmetic())
        .value("ALL", VK_SHADER_STAGE_ALL)
        .value("FRAGMENT_BIT", VK_SHADER_STAGE_FRAGMENT_BIT)
        .value("VERTEX_BIT", VK_SHADER_STAGE_VERTEX_BIT);

    py::enum_<VkDynamicState>(m, "DynamicState")
        .value("VIEWPORT", VK_DYNAMIC_STATE_VIEWPORT)
        .value("SCISSOR", VK_DYNAMIC_STATE_SCISSOR);


    py::class_< VkPhysicalDeviceProperties>(m, "PhysicalDeviceProperties")
        .def_property_readonly("api_version", [](const VkPhysicalDeviceProperties& d) { 
            return std::tuple<uint32_t, uint32_t, uint32_t>(
                VK_VERSION_MAJOR(d.apiVersion),
                VK_VERSION_MINOR(d.apiVersion),
                VK_VERSION_PATCH(d.apiVersion));
        })
        .def_readonly("device_id", &VkPhysicalDeviceProperties::deviceID)
        .def_property_readonly("device_name", [](const VkPhysicalDeviceProperties& d) { return std::string(d.deviceName);  })
        .def_readonly("device_type", &VkPhysicalDeviceProperties::deviceType)
        .def_readonly("driver_version", &VkPhysicalDeviceProperties::driverVersion)
        // .def_readwrite("", &VkPhysicalDeviceProperties::limits)
        .def_readonly("pipeline_cache_uuid", &VkPhysicalDeviceProperties::pipelineCacheUUID)
        // .def_readwrite("", &VkPhysicalDeviceProperties::sparseProperties)
        .def_readonly("vendor_id", &VkPhysicalDeviceProperties::vendorID);

        py::enum_< vkb::PreferredDeviceType>(m, "PreferredDeviceType")
            .value("integrated", vkb::PreferredDeviceType::integrated)
            .value("discrete", vkb::PreferredDeviceType::discrete);

        py::enum_< VkPhysicalDeviceType>(m, "PhysicalDeviceType")
            .value("CPU", VK_PHYSICAL_DEVICE_TYPE_CPU)
            .value("DISCRETE_GPU", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            .value("INTEGRATED_GPU", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            .value("TYPE_OTHER", VK_PHYSICAL_DEVICE_TYPE_OTHER);

        py::class_< SingleTimeCommandExecutor>(m, "SingleTimeCommandExecutor")
            .def(py::init<Device, CommandPool, Queue>())
            .def("enter", &SingleTimeCommandExecutor::enter)
            .def("exit", &SingleTimeCommandExecutor::exit)
            .def("wait", &SingleTimeCommandExecutor::wait)
            .def("destroy", &SingleTimeCommandExecutor::enter)
            .def("__enter__", [&](SingleTimeCommandExecutor& e) -> vku::CommandBuffer { return e.enter();  })
            .def("__exit__", [&](SingleTimeCommandExecutor& e, py::object exc_type, py::object exc_value, py::object traceback) { e.exit();  });

        py::class_<VkBufferCopy>(m, "BufferCopy")
            .def(py::init<VkDeviceSize, VkDeviceSize, VkDeviceSize>(), py::arg("src_offset") = 0, py::arg("dst_offset") = 0, py::arg("size") = 0)
            .def_readwrite("size", &VkBufferCopy::size)
            .def_readwrite("dst_offset", &VkBufferCopy::dstOffset)
            .def_readwrite("src_offset", &VkBufferCopy::srcOffset);

        py::enum_< VkIndexType>(m, "IndexType")
            .value("UINT16", VK_INDEX_TYPE_UINT16)
            .value("UINT32", VK_INDEX_TYPE_UINT32);

        py::class_<ImageData>(m, "ImageData")
            .def_readonly("channels", &ImageData::channels)
            .def_readonly("height", &ImageData::height)
            .def_readonly("width", &ImageData::width)
            .def("is_valid", &ImageData::is_valid)
            .def("get_data", &ImageData::get_data);

        py::class_< DescriptorPoolBuilder>(m, "DescriptorPoolBuilder")
            .def(py::init<Device>())
            .def("build", &DescriptorPoolBuilder::build)
            .def("add_descriptor_sets", &DescriptorPoolBuilder::add_descriptor_sets);

        py::class_< DescriptorSetLayoutBuilder>(m, "DescriptorSetLayoutBuilder")
            .def(py::init<Device>())
            .def("build", &DescriptorSetLayoutBuilder::build)
            .def("add_binding", &DescriptorSetLayoutBuilder::add_binding);

        py::class_< DescriptorSetBuilder>(m, "DescriptorSetBuilder")
            .def(py::init<Device, vku::DescriptorPool, vku::DescriptorSetLayout>())
            .def("build", &DescriptorSetBuilder::build)
            .def("write_uniform_buffer", &DescriptorSetBuilder::write_uniform_buffer);

        py::enum_<VkDescriptorType>(m, "DescriptorType")
            .value("UNIFORM_BUFFER", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}


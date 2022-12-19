#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <vku.h>

#include <stdexcept>

namespace py = pybind11;


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
    operator T() const { return handle;  }
    VkHandle& operator=(const T& value) { handle = value; return *this; }
};


class Queue : public VkHandle<VkQueue> { };
class Surface : public VkHandle<VkSurfaceKHR> { };
class Image : public VkHandle<VkImage> { };
class ImageView : public VkHandle<VkImageView> { };
class RenderPass : public VkHandle<VkRenderPass> { };
class Framebuffer : public VkHandle<VkFramebuffer> { };
class CommandPool : public VkHandle<VkCommandPool> { };
class CommandBuffer : public VkHandle<VkCommandBuffer> { };
class Semaphore : public VkHandle<VkSemaphore> { };
class Fence : public VkHandle<VkFence> { };


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
        results[i].handle = handles[i];
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

    PhysicalDevice select()
    {
        auto result = physical_device_selector.select();
        auto physical_device = result.value();
        return PhysicalDevice(physical_device);
    }
};


class Device
{
public:
    Device(const vkb::Device &d) : device(d) { }

    vkb::Device device;

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

    SwapchainBuilder& set_old_swapchain(Swapchain& s)
    {
        swapchain_builder.set_old_swapchain(s);
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


std::vector<CommandBuffer> allocate_command_buffers(const Device& device, const CommandBufferAllocateInfo&i)
{
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandBufferCount = i.command_buffer_count;
    alloc_info.commandPool = i.command_pool.handle;
    alloc_info.level = i.level;
    std::vector<VkCommandBuffer> command_buffers(alloc_info.commandBufferCount);
    auto vk_result = vkAllocateCommandBuffers(device.device, &alloc_info, command_buffers.data());

    // TODO: error handling;

    return std::move(get_wrappers<CommandBuffer>(command_buffers));
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

void reset_command_buffer(const CommandBuffer& command_buffer, VkCommandBufferResetFlags flags)
{
    vkResetCommandBuffer(command_buffer, flags);
}


class SubmitInfo
{
public:
    std::vector<Semaphore> wait_semaphores{};
    std::vector<VkPipelineStageFlagBits> wait_dst_stage_masks{};
    std::vector<CommandBuffer> command_buffers{};
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


void begin_command_buffer(const CommandBuffer& command_buffer)
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


void cmd_set_viewport(const CommandBuffer &command_buffer, uint32_t first_viewport, uint32_t viewport_count, std::vector<VkViewport> &viewports)
{
    vkCmdSetViewport(command_buffer, first_viewport, viewport_count, viewports.data());
}


void cmd_set_scissor(const CommandBuffer& command_buffer, uint32_t first_scissor, uint32_t scissor_count, std::vector<VkRect2D>& scissors)
{
    vkCmdSetScissor(command_buffer, first_scissor, scissor_count, scissors.data());
}

void cmd_begin_render_pass(const CommandBuffer& command_buffer, const RenderPassBeginInfo &info, VkSubpassContents contents)
{
    std::vector<VkClearValue> vk_clear_values;
    VkRenderPassBeginInfo begin_info;
    info.populate(begin_info, vk_clear_values);
    vkCmdBeginRenderPass(command_buffer, &begin_info, contents);
}

void cmd_end_render_pass(const CommandBuffer& command_buffer)
{
    vkCmdEndRenderPass(command_buffer);
}


void end_command_buffer(const CommandBuffer& command_buffer)
{
    vkEndCommandBuffer(command_buffer);
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
        .def("end_command_buffer", end_command_buffer);

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
        .def("select", &PhysicalDeviceSelector::select);

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
        .value("UNDEFINED", VK_FORMAT_UNDEFINED);

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

    py::class_< CommandBuffer>(m, "CommandBuffer");

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



    /*
    py::class_<VkClearColorValue>(m, "ColorClearValue")
        .def(py::init<>())
        .def_readwrite("float32", &VkClearColorValue::float32)
        .def_readwrite("int32", &VkClearColorValue::int32)
        .def_readwrite("uint32", &VkClearColorValue::uint32);
        */

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
}


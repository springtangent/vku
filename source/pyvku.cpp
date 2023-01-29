#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"


#include <VkBootstrap.h>
#include <vku.h>

#include <stdexcept>
#include <tuple>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// TODO: do we need this?
#include <glm/gtx/projection.hpp>

namespace py = pybind11;

class ImageData
{
public:
    int width{0};
    int height{ 0 };
    int channels{ 0 };
    stbi_uc* pixels{ nullptr };

    ImageData() = default;
    ImageData(const ImageData& d) = delete;
    ImageData(ImageData&& d) : width(d.width), height(d.height), channels(d.channels), pixels(d.pixels) { d.clear();  }

    ~ImageData()
    {
        if (pixels)
        {
            stbi_image_free(pixels);
        }

        clear();
    }

    ImageData &operator=(ImageData&& d)
    {
        pixels = d.pixels;
        width = d.width;
        height = d.height;
        channels = d.channels;
        d.clear();
        return *this;
    }

    bool is_valid()
    {
        return pixels != nullptr;
    }

    std::optional<py::memoryview> get_data()
    {
        if (!pixels)
        {
            return {};
        }
        return { py::memoryview::from_memory(pixels, size()) };
    }

    VkExtent2D extent() const
    {
        return VkExtent2D{ (uint32_t)width, (uint32_t)height};
    }
    
    VkDeviceSize size() const
    {
        return width * height * 4;
    }

private:
    void clear()
    {
        pixels = nullptr;
        width = height = channels = 0;
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


template<typename W, typename H, typename F>
std::vector<W> get_wrappers(const std::vector<H>& handles, F f)
{
    std::vector<W> results;
    results.reserve(handles.size());

    for (const auto& h : handles)
    {
        results.push_back(f(h));
    }

    return std::move(results);
}



class PhysicalDevice 
{
public:
    PhysicalDevice(vkb::PhysicalDevice d) : physical_device(d) { }

    operator VkPhysicalDevice() const { return (VkPhysicalDevice)physical_device; }

    vkb::PhysicalDevice physical_device{};

    std::vector<std::string> get_extensions() const
    {
        return physical_device.get_extensions();
    }

    std::vector<VkExtensionProperties> enumerate_extensions() const
    {
        uint32_t extension_count = 0;
        auto result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> results(extension_count);
        result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, results.data());
        return results;
    }
};

enum class InputEventType
{
    NONE = 0,
    KEY = 1,
    CHARACTER = 2,
    MOUSE_BUTTON = 3,
    CURSOR_POS = 4,
    CURSOR_ENTER = 5,
    CURSOR_SCROLL = 6,
    DROP = 7
};

struct KeyInputEvent {
    int key;
    int scancode;
    int action;
    int mods;
};

struct CharInputEvent {
    unsigned int codepoint;
};

struct MouseButtonEvent {
    int button;
    int action;
    int mods;
};

struct CursorPosEvent
{
    double xpos;
    double ypos;
};

struct CursorEnterEvent
{
    int entered;
};

struct CursorScrollEvent
{
    double xoffset;
    double yoffset;
};

struct InputEvent
{
    InputEventType type;

    union {
        KeyInputEvent key;
        CharInputEvent character;
        MouseButtonEvent mouse_button;
        CursorPosEvent cursor_pos;
        CursorEnterEvent cursor_enter;
        CursorScrollEvent cursor_scroll;
    };
};

inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
inline void char_callback(GLFWwindow* window, unsigned int codepoint);
inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
inline void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
inline void cursor_enter_callback(GLFWwindow* window, int entered);
inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

class Window
{
public:
    Window(GLFWwindow* w) : window(w), input_events()
    {
        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, key_callback);
        glfwSetCharCallback(window, char_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetCursorPosCallback(window, cursor_pos_callback);
        glfwSetCursorEnterCallback(window, cursor_enter_callback);
glfwSetScrollCallback(window, scroll_callback);

    }

    void make_context_current()
    {
        glfwMakeContextCurrent(window);
    }

    vku::Surface create_surface(vku::Instance& instance)
    {
        VkSurfaceKHR result{ VK_NULL_HANDLE };

        auto vk_result = glfwCreateWindowSurface(instance.handle, window, nullptr, &result);

        return vku::Surface(instance, result);
    }

    bool should_close()
    {
        return glfwWindowShouldClose(window);
    }

    void swap_buffers()
    {
        glfwSwapBuffers(window);
    }

    void set_input_mode(int mode, int value)
    {
        glfwSetInputMode(window, mode, value);
    }

    void push_event(const InputEvent& e)
    {
        input_events.push_back(e);
    }

    std::vector<InputEvent> get_events()
    {
        auto result = input_events;
        input_events.clear();
        return result;
    }

    GLFWwindow* window{ nullptr };
    std::vector<InputEvent> input_events{};
};



struct Imgui
{
    VkInstance instance{ VK_NULL_HANDLE };
    VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
    VkDevice device{ VK_NULL_HANDLE };
    VkQueue queue{ VK_NULL_HANDLE };
    uint32_t queue_family{ 0 };
    std::shared_ptr<Window> window;

    VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
    uint32_t min_image_count{ 3 };
    uint32_t image_count{ 3 };
    VkSampleCountFlagBits samples{ VK_SAMPLE_COUNT_1_BIT };
    bool show_demo_window = true;

    vku::SingleTimeCommandExecutor init_executor{};
    VkCommandPool command_pool{ VK_NULL_HANDLE };
    //  ImGui_ImplVulkanH_Window main_window_data;

    Imgui(std::shared_ptr<Window> _window, vku::Instance _instance, PhysicalDevice _physical_device, vku::Device _device, vku::Queue _queue, uint32_t _queue_family, vku::CommandPool _command_pool, VkSampleCountFlagBits _samples) :
        window(_window), instance(_instance), physical_device(_physical_device), device(_device), queue(_queue), command_pool(_command_pool), samples(_samples)
    {
    }

    bool init(vku::RenderPass _render_pass)
    {
        py::print("Imgui::init creating descriptor pool.");
        VkDescriptorPoolSize pool_sizes[] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = std::size(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;

        if (VK_SUCCESS != vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool))
        {
            return false;
        }

        py::print("Imgui::init creating context.");
        ImGui::CreateContext();

        py::print("Imgui::init initializing vulkan.");
        // window_data.
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physical_device;
        init_info.Device = device;
        init_info.QueueFamily = queue_family;
        init_info.Queue = queue;
        init_info.DescriptorPool = descriptor_pool;
        init_info.MinImageCount = min_image_count;
        init_info.ImageCount = image_count;
        init_info.MSAASamples = samples;

        ImGui_ImplVulkan_Init(&init_info, _render_pass);
        bool install_callbacks = true;
        if (!ImGui_ImplGlfw_InitForVulkan(window->window, install_callbacks))
        {
            return false;
        }

        py::print("Imgui::init creating init queue.");
        init_executor.init(device, command_pool, queue);
        auto executor_result = init_executor.enter();

        if (!executor_result)
        {
            // TODO: throw an error nd stuff.
            return false;
        }

        py::print("Imgui::init populating init pool.");
        auto command_buffer = executor_result.get_value();

        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

        // this submits the queue and starts it executing.
        py::print("Imgui::init finalizing init pool.");
        init_executor.exit();

        return true;
    }

    void set_image_count(uint32_t image_count)
    {
        ImGui_ImplVulkan_SetMinImageCount(image_count);
    }

    void wait_init()
    {
        // wait on the fence, destroy the fence and the command queue once it's done executing.
        init_executor.wait();
        init_executor.destroy();
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    void new_frame()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);
    }

    void render(vku::CommandBuffer command_buffer)
    {
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
    }

    void destroy()
    {
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
};



inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    py::gil_scoped_acquire acquire;
    // do nothing for now
    InputEvent input_event{ InputEventType::KEY };
    input_event.key = { key, scancode, action, mods };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}

inline void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    py::gil_scoped_acquire acquire;
    // do nothing for now
    InputEvent input_event{ InputEventType::CHARACTER };
    input_event.character = { codepoint };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}

inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    py::gil_scoped_acquire acquire;
    InputEvent input_event{ InputEventType::MOUSE_BUTTON };
    input_event.mouse_button = MouseButtonEvent{ button, action, mods };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}

inline void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    py::gil_scoped_acquire acquire;
    InputEvent input_event{ InputEventType::CURSOR_POS };
    input_event.cursor_pos = CursorPosEvent{ xpos, ypos };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}

inline void cursor_enter_callback(GLFWwindow* window, int entered)
{
    py::gil_scoped_acquire acquire;
    InputEvent input_event{ InputEventType::CURSOR_ENTER };
    input_event.cursor_enter = CursorEnterEvent{ entered };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}

inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    py::gil_scoped_acquire acquire;
    InputEvent input_event{ InputEventType::CURSOR_SCROLL };
    input_event.cursor_scroll = CursorScrollEvent{ xoffset, yoffset };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->push_event(input_event);
}


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


std::shared_ptr<Window> create_window(int width, int height, const char *title)
{
    auto result = glfwCreateWindow(width, height, title, nullptr, nullptr);

    if (!result)
    {
        // TODO: error stuff.
        const char* error = nullptr;
        glfwGetError(&error);
        if (error)
        {
            throw std::runtime_error(error);
        }
    }

    return std::make_shared<Window>(result);
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

    InstanceBuilder& enable_extension(const char *extension_name)
    {
        instance_builder.enable_extension(extension_name);
        return *this;
    }

    vku::Instance build() const
    {
        auto instance_result = instance_builder.build();

        if (!instance_result)
        {
            throw std::runtime_error(instance_result.error().message());
        }

        return vku::Instance(instance_result.value());
    }
};



class PhysicalDeviceSelector
{
public:
    PhysicalDeviceSelector(vku::Instance &instance) : physical_device_selector(instance.handle) { }

    vkb::PhysicalDeviceSelector physical_device_selector;

    PhysicalDeviceSelector& set_surface(const vku::Surface &surface)
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

    PhysicalDeviceSelector& set_required_features(VkPhysicalDeviceFeatures &features)
    {
        physical_device_selector.set_required_features(features);
        return *this;
    }

    PhysicalDeviceSelector& add_required_extension(const char*e)
    {
        physical_device_selector.add_required_extension(e);
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


class DeviceBuilder
{
public:
    DeviceBuilder(PhysicalDevice& physical_device) : device_builder(physical_device.physical_device) { }

    vkb::DeviceBuilder device_builder;

    vku::Device build()
    {
        auto res = device_builder.build();
        if (!res.has_value())
        {
            throw std::runtime_error(res.error().message());
        }
        return vku::Device(res.value());
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

    std::vector<vku::Image> get_images()
    {
        auto result = swapchain.get_images();
        // TODO: error handling
        auto images = result.value();

        // TODO: see what happens if you try to destroy this, it might cause a crash...
        return std::move(get_wrappers<vku::Image>(images, [&](VkImage image) { return vku::Image(VK_NULL_HANDLE, image, VK_NULL_HANDLE);  }));
    }

    std::vector<vku::ImageView> get_image_views()
    {
        auto result = swapchain.get_image_views();
        // TODO: error handling
        auto image_views = result.value();
        return std::move(get_wrappers<vku::ImageView>(image_views, [&](VkImageView image_view) {
            return vku::ImageView(swapchain.device, image_view);
        }));
    }

    void destroy_image_views(std::vector<vku::ImageView> image_views)
    {
        std::vector<VkImageView> ivs = get_handles<VkImageView>(image_views);
        swapchain.destroy_image_views(ivs);
    }

    void destroy()
    {
        vkb::destroy_swapchain(swapchain);
    }
};


class FramebufferCreateInfo
{
public:
    vku::RenderPass render_pass;
    std::vector<vku::ImageView> attachments{};
    uint32_t width{ 0 };
    uint32_t height{ 0 };
    uint32_t layers{ 0 };
};


vku::Framebuffer create_framebuffer(vku::Device &device, FramebufferCreateInfo &create_info)
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

    VkFramebuffer result{ VK_NULL_HANDLE };
    auto vk_result = vkCreateFramebuffer(device, &ci, nullptr, &result);

    // TODO: error handler.

    return vku::Framebuffer(device, result);
}


class SwapchainBuilder
{
public:
    SwapchainBuilder(vku::Device& d) : swapchain_builder(d.get_device()) { }

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
        if (!res.has_value())
        {
            throw std::runtime_error(res.error().message());
        }
        return Swapchain(res.value());
    }

};


class SubpassDescription
{
public:
    VkPipelineBindPoint pipeline_bind_point{ VK_PIPELINE_BIND_POINT_GRAPHICS };
    std::vector<VkAttachmentReference> color_attachments{};
    std::vector<VkAttachmentReference> depth_stencil_attachments{};

    uint32_t color_attachments_start{ 0 };

    void populate(VkSubpassDescription& result)
    {
        result.flags = 0;
        result.pipelineBindPoint = pipeline_bind_point;
        result.colorAttachmentCount = color_attachments.size();

        if (color_attachments.size())
        {
            result.pColorAttachments = color_attachments.data();
        }

        if (depth_stencil_attachments.size())
        {
            result.pDepthStencilAttachment = depth_stencil_attachments.data();
        }
    }
};

/*
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
*/

class RenderPassCreateInfo
{
public:
    std::vector<VkAttachmentDescription> attachments{};
    std::vector<SubpassDescription> subpasses{};
    std::vector<VkSubpassDependency> dependencies{};
};


/*
template<typename VK, typename D>
std::vector<VK> populate(std::vector<D>& items)
{
    std::vector<VK> results(items.size());

    for (int i = 0; i < results.size(); ++i)
    {
        items[i].populate(results[i]);
    }

    return std::move(results);
}*/


vku::RenderPass create_render_pass(vku::Device& device, RenderPassCreateInfo& create_info)
{
    VkRenderPass result{ VK_NULL_HANDLE };
    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.flags = 0;

    ci.attachmentCount = create_info.attachments.size();
    ci.pAttachments = create_info.attachments.data();

    ci.dependencyCount = create_info.dependencies.size();
    ci.pDependencies = create_info.dependencies.data();

    std::vector<VkSubpassDescription> subpasses(create_info.subpasses.size());

    for (uint32_t i = 0; i < subpasses.size(); i++)
    {
        create_info.subpasses[i].populate(subpasses[i]);
    }

    ci.subpassCount = subpasses.size();
    ci.pSubpasses = subpasses.data();


    auto vk_result = vkCreateRenderPass(device, &ci, nullptr, &result);

    // TODO: error handling

    return vku::RenderPass(device, result);
}


class CommandPoolCreateInfo
{
public:
    VkCommandPoolCreateFlags flags{ 0 };
    uint32_t queue_family_index{ 0 };
};


vku::CommandPool create_command_pool(const vku::Device& device, const CommandPoolCreateInfo& create_info)
{
    VkCommandPool result{ VK_NULL_HANDLE };
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = create_info.flags;
    ci.pNext = nullptr;
    ci.queueFamilyIndex = create_info.queue_family_index;
    auto vk_result = vkCreateCommandPool(device, &ci, nullptr, &result);
    // TODO: error handling
    return vku::CommandPool(device, result);
}


std::vector<vku::CommandBuffer> allocate_command_buffers(const vku::Device& device, const VkCommandBufferAllocateInfo &alloc_info)
{
    std::vector<VkCommandBuffer> command_buffers(alloc_info.commandBufferCount);
    auto vk_result = vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data());

    // TODO: error handling;

    std::vector<vku::CommandBuffer> results;
    results.reserve(command_buffers.size());

    for (const auto& cb : command_buffers)
    {
        results.emplace_back(device, cb, alloc_info.commandPool);
    }

    return std::move(results);

    // return std::move(get_vku_wrappers<vku::CommandBuffer>(device, command_pool, command_buffers));
}

vku::Semaphore create_semaphore(const vku::Device& device)
{
    VkSemaphore result{ VK_NULL_HANDLE };
    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.flags = 0;
    create_info.pNext = nullptr;
    auto vk_result = vkCreateSemaphore(device, &create_info, nullptr, &result);
    // TODO: error handling.
    return vku::Semaphore(device, result);
}


class FenceCreateInfo
{
public:
    VkFenceCreateFlags flags{  };
};

vku::Fence create_fence(const vku::Device& device, const FenceCreateInfo& create_info)
{
    VkFence result{ VK_NULL_HANDLE };
    VkFenceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ci.flags = create_info.flags;
    ci.pNext = nullptr;
    auto vk_result = vkCreateFence(device, &ci, nullptr, &result);
    // TODO: ERROR HANDLING
    return vku::Fence(device, result);
}

void wait_for_fences(const vku::Device &device, std::vector<vku::Fence> &fences, bool wait_for_all, uint64_t timeout)
{
    std::vector<VkFence> vkfences = get_handles<VkFence>(fences);
    auto vk_result = vkWaitForFences(device, vkfences.size(), vkfences.data(), wait_for_all, timeout);

    // TODO: error handling
}

uint32_t acquire_next_image(const vku::Device &device, const Swapchain &swapchain, uint64_t timeout, const vku::Semaphore &semaphore)
{
    uint32_t image_index{ 0 };
    VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, timeout, semaphore, VK_NULL_HANDLE, &image_index);

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

void reset_fences(const vku::Device &device, const std::vector<vku::Fence> &fences)
{
    std::vector<VkFence> vkfences = get_handles<VkFence>(fences);
    auto vk_result = vkResetFences(device, vkfences.size(), vkfences.data());

    // TODO: error handling
}

void reset_command_buffer(const vku::CommandBuffer& command_buffer, VkCommandBufferResetFlags flags)
{
    vkResetCommandBuffer(command_buffer, flags);
}


class SubmitInfo
{
public:
    std::vector<vku::Semaphore> wait_semaphores{};
    std::vector<VkPipelineStageFlagBits> wait_dst_stage_masks{};
    std::vector<vku::CommandBuffer> command_buffers{};
    std::vector<vku::Semaphore> signal_semaphores{};

    uint32_t wait_semaphores_start{ 0 };
    uint32_t wait_dst_stage_masks_start{ 0 };
    uint32_t command_buffers_start{ 0 };
    uint32_t signal_semaphores_start{ 0 };
};


void queue_submit(const vku::Queue &queue, std::vector<SubmitInfo> &submit_infos, const vku::Fence &fence)
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
    std::vector<vku::Semaphore> wait_semaphores{};
    std::vector<Swapchain> swapchains{};
    std::vector<uint32_t> image_indices{};
};


void queue_present(const vku::Queue &queue, const PresentInfo &present_info)
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


class RenderPassBeginInfo
{
public:
    vku::Framebuffer framebuffer{};
    vku::RenderPass render_pass{};
    VkRect2D render_area{};
    std::vector<VkClearValue> clear_values{};

    void populate(VkRenderPassBeginInfo& begin_info) const
    {
        begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin_info.pNext = nullptr;
        begin_info.framebuffer = framebuffer;
        begin_info.renderPass = render_pass;
        begin_info.renderArea = render_area;
        begin_info.clearValueCount = clear_values.size();
        begin_info.pClearValues = clear_values.data();
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
    VkRenderPassBeginInfo begin_info;
    info.populate(begin_info);
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
    PipelineLayoutBuilder(vku::Device& device) : pipeline_layout_builder(device.get_device()) { }

    PipelineLayoutBuilder& add_descriptor_set(const vku::DescriptorSetLayout& layout)
    {
        pipeline_layout_builder.add_descriptor_set(layout);
        return *this;
    }

    PipelineLayoutBuilder& add_push_constant_range(VkShaderStageFlags stage_flags, uint32_t offset, uint32_t size)
    {
        pipeline_layout_builder.add_push_constant_range(stage_flags, offset, size);
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


vku::ShaderModule create_shader_module(const vku::Device &device, py::buffer data)
{
    py::buffer_info info = data.request();
    auto shader_module_result = vku::create_shader_module(device, reinterpret_cast<const uint32_t *>(info.ptr), info.size * info.itemsize);
    
    // TODO: error handling;

    return shader_module_result.get_value();
}


class GraphicsPipelineBuilder
{
public:
    GraphicsPipelineBuilder(vku::Device& device) : graphics_pipeline_builder(device.get_device()) { }

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

    GraphicsPipelineBuilder& set_render_pass(vku::RenderPass& rp)
    {
        graphics_pipeline_builder.set_render_pass(rp);
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
    BufferFactory(const vku::Device& device, const PhysicalDevice& physical_device) : buffer_factory(device, physical_device.physical_device) { }

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
    SingleTimeCommandExecutor(const vku::Device& d, const vku::CommandPool& command_pool, const vku::Queue &queue) : executor(d, command_pool, queue), device(d) { }

    vku::CommandBuffer enter()
    {
        auto result = executor.enter();

        if (!result)
        {
            throw std::runtime_error(result.get_error().message());
        }

        return vku::CommandBuffer{ result.get_value() };
    }

    vku::Fence exit()
    {
        auto result = executor.exit();

        if (!result)
        {
            throw std::runtime_error(result.get_error().message());
        }

        return vku::Fence(device, result.get_value());
    }

    void wait()
    {
        auto result = executor.wait();

        if (result.has_value())
        {
            throw std::runtime_error(result.value().type.message());
        }
    }

    void destroy()
    {
        executor.destroy();
    }

    vku::SingleTimeCommandExecutor executor{};
    VkDevice device{ VK_NULL_HANDLE };
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
    DescriptorPoolBuilder(const vku::Device& d) : descriptor_pool_builder((VkDevice)d) { }

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


class DescriptorSetLayoutBuilder
{
public:
    DescriptorSetLayoutBuilder(const vku::Device& device) : builder(device)
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
    DescriptorSetBuilder(const vku::Device& d, const vku::DescriptorPool& dp, vku::DescriptorSetLayout l) : builder(d, dp, l) { }

    DescriptorSetBuilder& write_uniform_buffer(uint32_t binding, uint32_t array_element, vku::Buffer buffer, VkDeviceSize offset, VkDeviceSize range)
    {
        builder.write_uniform_buffer(binding, array_element, buffer, offset, range);
        return *this;
    }

    DescriptorSetBuilder& write_storage_buffer(uint32_t binding, uint32_t array_element, vku::Buffer buffer, VkDeviceSize offset, VkDeviceSize range)
    {
        builder.write_storage_buffer(binding, array_element, buffer, offset, range);
        return *this;
    }

    inline DescriptorSetBuilder& write_combined_image_sampler(uint32_t binding, uint32_t array_element, const vku::Image image, const vku::ImageView image_view, vku::Sampler sampler)
    {
        builder.write_combined_image_sampler(binding, array_element, image, image_view, sampler);
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

void cmd_pipeline_barrier(
    const vku::CommandBuffer &command_buffer,
    VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask,
    VkDependencyFlags dependency_flags,
   //  std::vector<VkMemoryBarrier> memory_barriers,
    // std::vector<VkBufferMemoryBarrier> &buffer_memory_barriers,
    std::vector<VkImageMemoryBarrier> &image_memory_barriers
)
{
    vkCmdPipelineBarrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        dependency_flags,
        0, nullptr, // memory_barriers.size(), memory_barriers.data(),
        0, nullptr, // buffer_memory_barriers.size(), buffer_memory_barriers.data(),
        image_memory_barriers.size(), image_memory_barriers.data()
    );
}


void cmd_copy_buffer_to_image(const vku::CommandBuffer &command_buffer, const vku::Buffer &src_buffer, const vku::Image &dst_image, VkImageLayout dst_image_layout, const std::vector<VkBufferImageCopy> &regions)
{
    vkCmdCopyBufferToImage(command_buffer, src_buffer, dst_image, dst_image_layout, regions.size(), regions.data());
}


void bind_vk_aabb_positions_khr(py::module& m) {
    py::class_<VkAabbPositionsKHR> aabb_positions_khr(m, "AabbPositions");
    aabb_positions_khr.def(py::init<>())
        .def(py::init<float, float, float, float, float, float>(),
            py::arg("min_x") = 0.0f, py::arg("min_y") = 0.0f, py::arg("min_z") = 0.0f,
            py::arg("max_x") = 0.0f, py::arg("max_y") = 0.0f, py::arg("max_z") = 0.0f)
        .def_readwrite("min_x", &VkAabbPositionsKHR::minX)
        .def_readwrite("min_y", &VkAabbPositionsKHR::minY)
        .def_readwrite("min_z", &VkAabbPositionsKHR::minZ)
        .def_readwrite("max_x", &VkAabbPositionsKHR::maxX)
        .def_readwrite("max_y", &VkAabbPositionsKHR::maxY)
        .def_readwrite("max_z", &VkAabbPositionsKHR::maxZ);
}


void cmd_push_constants(
    const vku::CommandBuffer& command_buffer,
    const vku::PipelineLayout &pipeline_layout,
    VkShaderStageFlags stage_flags,
    uint32_t offset,
    py::buffer value)
{
    py::buffer_info info = value.request();
    // TODO: probably want some range checking...
    vkCmdPushConstants(command_buffer, pipeline_layout, stage_flags, offset, info.size * info.itemsize, info.ptr);
}


VkFormat find_supported_format(PhysicalDevice physical_device, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    return vku::find_supported_format(physical_device, candidates, tiling, features);
}

inline void bind_glfw_input_events(py::module m)
{
    py::enum_<InputEventType>(m, "InputEventType")
        .value("NONE", InputEventType::NONE)
        .value("KEY", InputEventType::KEY)
        .value("CHARACTER", InputEventType::CHARACTER)
        .value("MOUSE_BUTTON", InputEventType::MOUSE_BUTTON)
        .value("CURSOR_POS", InputEventType::CURSOR_POS)
        .value("CURSOR_ENTER", InputEventType::CURSOR_ENTER)
        .value("CURSOR_SCROLL", InputEventType::CURSOR_SCROLL);

    py::class_<KeyInputEvent>(m, "KeyInputEvent")
        .def_readonly("key", &KeyInputEvent::key)
        .def_readonly("scancode", &KeyInputEvent::scancode)
        .def_readonly("action", &KeyInputEvent::action)
        .def_readonly("mods", &KeyInputEvent::mods);

    py::class_<CharInputEvent>(m, "CharInputEvent")
        .def_readonly("codepoint", &CharInputEvent::codepoint);

    py::class_<MouseButtonEvent>(m, "MouseButtonEvent")
        .def_readonly("button", &MouseButtonEvent::button)
        .def_readonly("action", &MouseButtonEvent::action)
        .def_readonly("mods", &MouseButtonEvent::mods);

    py::class_<CursorPosEvent>(m, "CursorPosEvent")
        .def_readonly("xpos", &CursorPosEvent::xpos)
        .def_readonly("ypos", &CursorPosEvent::ypos);

    py::class_<CursorEnterEvent>(m, "CursorEnterEvent")
        .def_readonly("entered", &CursorEnterEvent::entered);

    py::class_<CursorScrollEvent>(m, "CursorScrollEvent")
        .def_readonly("xoffset", &CursorScrollEvent::xoffset)
        .def_readonly("yoffset", &CursorScrollEvent::yoffset);

    py::class_<InputEvent>(m, "InputEvent")
        .def_readonly("key", &InputEvent::key)
        .def_readonly("character", &InputEvent::character)
        .def_readonly("cursor_enter", &InputEvent::cursor_enter)
        .def_readonly("cursor_pos", &InputEvent::cursor_pos)
        .def_readonly("cursor_scroll", &InputEvent::cursor_scroll)
        .def_readonly("mouse_button", &InputEvent::mouse_button)
        .def_readonly("type", &InputEvent::type);

}

inline void bind_glfw_key_tokens(py::module m)
{
    m.attr("KEY_UNKNOWN") = py::int_(GLFW_KEY_UNKNOWN);
    m.attr("KEY_SPACE") = py::int_(GLFW_KEY_SPACE);
    m.attr("KEY_APOSTROPHE") = py::int_(GLFW_KEY_APOSTROPHE);
    m.attr("KEY_COMMA") = py::int_(GLFW_KEY_COMMA);
    m.attr("KEY_MINUS") = py::int_(GLFW_KEY_MINUS);
    m.attr("KEY_PERIOD") = py::int_(GLFW_KEY_PERIOD);
    m.attr("KEY_SLASH") = py::int_(GLFW_KEY_SLASH);
    m.attr("KEY_0") = py::int_(GLFW_KEY_0);
    m.attr("KEY_1") = py::int_(GLFW_KEY_1);
    m.attr("KEY_2") = py::int_(GLFW_KEY_2);
    m.attr("KEY_3") = py::int_(GLFW_KEY_3);
    m.attr("KEY_4") = py::int_(GLFW_KEY_4);
    m.attr("KEY_5") = py::int_(GLFW_KEY_5);
    m.attr("KEY_6") = py::int_(GLFW_KEY_6);
    m.attr("KEY_7") = py::int_(GLFW_KEY_7);
    m.attr("KEY_8") = py::int_(GLFW_KEY_8);
    m.attr("KEY_9") = py::int_(GLFW_KEY_9);
    m.attr("KEY_SEMICOLON") = py::int_(GLFW_KEY_SEMICOLON);
    m.attr("KEY_EQUAL") = py::int_(GLFW_KEY_EQUAL);
    m.attr("KEY_A") = py::int_(GLFW_KEY_A);
    m.attr("KEY_B") = py::int_(GLFW_KEY_B);
    m.attr("KEY_C") = py::int_(GLFW_KEY_C);
    m.attr("KEY_D") = py::int_(GLFW_KEY_D);
    m.attr("KEY_E") = py::int_(GLFW_KEY_E);
    m.attr("KEY_F") = py::int_(GLFW_KEY_F);
    m.attr("KEY_G") = py::int_(GLFW_KEY_G);
    m.attr("KEY_H") = py::int_(GLFW_KEY_H);
    m.attr("KEY_I") = py::int_(GLFW_KEY_I);
    m.attr("KEY_J") = py::int_(GLFW_KEY_J);
    m.attr("KEY_K") = py::int_(GLFW_KEY_K);
    m.attr("KEY_L") = py::int_(GLFW_KEY_L);
    m.attr("KEY_M") = py::int_(GLFW_KEY_M);
    m.attr("KEY_N") = py::int_(GLFW_KEY_N);
    m.attr("KEY_O") = py::int_(GLFW_KEY_O);
    m.attr("KEY_P") = py::int_(GLFW_KEY_P);
    m.attr("KEY_Q") = py::int_(GLFW_KEY_Q);
    m.attr("KEY_R") = py::int_(GLFW_KEY_R);
    m.attr("KEY_S") = py::int_(GLFW_KEY_S);
    m.attr("KEY_T") = py::int_(GLFW_KEY_T);
    m.attr("KEY_U") = py::int_(GLFW_KEY_U);
    m.attr("KEY_V") = py::int_(GLFW_KEY_V);
    m.attr("KEY_W") = py::int_(GLFW_KEY_W);
    m.attr("KEY_X") = py::int_(GLFW_KEY_X);
    m.attr("KEY_Y") = py::int_(GLFW_KEY_Y);
    m.attr("KEY_Z") = py::int_(GLFW_KEY_Z);
    m.attr("KEY_ESCAPE") = py::int_(GLFW_KEY_ESCAPE);
    m.attr("KEY_ENTER") = py::int_(GLFW_KEY_ENTER);
    m.attr("KEY_TAB") = py::int_(GLFW_KEY_TAB);
    m.attr("KEY_BACKSPACE") = py::int_(GLFW_KEY_BACKSPACE);
    m.attr("KEY_INSERT") = py::int_(GLFW_KEY_INSERT);
    m.attr("KEY_DELETE") = py::int_(GLFW_KEY_DELETE);
    m.attr("KEY_RIGHT") = py::int_(GLFW_KEY_RIGHT);
    m.attr("KEY_LEFT") = py::int_(GLFW_KEY_LEFT);
    m.attr("KEY_DOWN") = py::int_(GLFW_KEY_DOWN);
    m.attr("KEY_UP") = py::int_(GLFW_KEY_UP);
    m.attr("KEY_PAGE_UP") = py::int_(GLFW_KEY_PAGE_UP);
    m.attr("KEY_PAGE_DOWN") = py::int_(GLFW_KEY_PAGE_DOWN);
    m.attr("KEY_HOME") = py::int_(GLFW_KEY_HOME);
    m.attr("KEY_END") = py::int_(GLFW_KEY_END);
    m.attr("KEY_CAPS_LOCK") = py::int_(GLFW_KEY_CAPS_LOCK);
    m.attr("KEY_SCROLL_LOCK") = py::int_(GLFW_KEY_SCROLL_LOCK);
    m.attr("KEY_NUM_LOCK") = py::int_(GLFW_KEY_NUM_LOCK);
    m.attr("KEY_PRINT_SCREEN") = py::int_(GLFW_KEY_PRINT_SCREEN);
    m.attr("KEY_PAUSE") = py::int_(GLFW_KEY_PAUSE);
    m.attr("KEY_F1") = py::int_(GLFW_KEY_F1);
    m.attr("KEY_F2") = py::int_(GLFW_KEY_F2);
    m.attr("KEY_F3") = py::int_(GLFW_KEY_F3);
    m.attr("KEY_F4") = py::int_(GLFW_KEY_F4);
    m.attr("KEY_F5") = py::int_(GLFW_KEY_F5);
    m.attr("KEY_F6") = py::int_(GLFW_KEY_F6);
    m.attr("KEY_F7") = py::int_(GLFW_KEY_F7);
    m.attr("KEY_F8") = py::int_(GLFW_KEY_F8);
    m.attr("KEY_F9") = py::int_(GLFW_KEY_F9);
    m.attr("KEY_F10") = py::int_(GLFW_KEY_F10);
    m.attr("KEY_F11") = py::int_(GLFW_KEY_F11);
    m.attr("KEY_F12") = py::int_(GLFW_KEY_F12);
    m.attr("KEY_F13") = py::int_(GLFW_KEY_F13);
    m.attr("KEY_F14") = py::int_(GLFW_KEY_F14);
    m.attr("KEY_F15") = py::int_(GLFW_KEY_F15);
    m.attr("KEY_F16") = py::int_(GLFW_KEY_F16);
    m.attr("KEY_F17") = py::int_(GLFW_KEY_F17);
    m.attr("KEY_F18") = py::int_(GLFW_KEY_F18);
    m.attr("KEY_F19") = py::int_(GLFW_KEY_F19);
    m.attr("KEY_F20") = py::int_(GLFW_KEY_F20);
    m.attr("KEY_F21") = py::int_(GLFW_KEY_F21);
    m.attr("KEY_F22") = py::int_(GLFW_KEY_F22);
    m.attr("KEY_F23") = py::int_(GLFW_KEY_F23);
    m.attr("KEY_F24") = py::int_(GLFW_KEY_F24);
    m.attr("KEY_F25") = py::int_(GLFW_KEY_F25);
    m.attr("KEY_KP_0") = py::int_(GLFW_KEY_KP_0);
    m.attr("KEY_KP_1") = py::int_(GLFW_KEY_KP_1);
    m.attr("KEY_KP_2") = py::int_(GLFW_KEY_KP_2);
    m.attr("KEY_KP_3") = py::int_(GLFW_KEY_KP_3);
    m.attr("KEY_KP_4") = py::int_(GLFW_KEY_KP_4);
    m.attr("KEY_KP_5") = py::int_(GLFW_KEY_KP_5);
    m.attr("KEY_KP_6") = py::int_(GLFW_KEY_KP_6);
    m.attr("KEY_KP_7") = py::int_(GLFW_KEY_KP_7);
    m.attr("KEY_KP_8") = py::int_(GLFW_KEY_KP_8);
    m.attr("KEY_KP_9") = py::int_(GLFW_KEY_KP_9);
    m.attr("KEY_KP_DECIMAL") = py::int_(GLFW_KEY_KP_DECIMAL);
    m.attr("KEY_KP_DIVIDE") = py::int_(GLFW_KEY_KP_DIVIDE);


    m.attr("KEY_KP_MULTIPLY") = py::int_(GLFW_KEY_KP_MULTIPLY);
    m.attr("KEY_KP_SUBTRACT") = py::int_(GLFW_KEY_KP_SUBTRACT);
    m.attr("KEY_KP_ADD") = py::int_(GLFW_KEY_KP_ADD);
    m.attr("KEY_KP_ENTER") = py::int_(GLFW_KEY_KP_ENTER);
    m.attr("KEY_KP_EQUAL") = py::int_(GLFW_KEY_KP_EQUAL);
    m.attr("KEY_LEFT_SHIFT") = py::int_(GLFW_KEY_LEFT_SHIFT);
    m.attr("KEY_LEFT_CONTROL") = py::int_(GLFW_KEY_LEFT_CONTROL);
    m.attr("KEY_LEFT_ALT") = py::int_(GLFW_KEY_LEFT_ALT);
    m.attr("KEY_LEFT_SUPER") = py::int_(GLFW_KEY_LEFT_SUPER);
    m.attr("KEY_RIGHT_SHIFT") = py::int_(GLFW_KEY_RIGHT_SHIFT);
    m.attr("KEY_RIGHT_CONTROL") = py::int_(GLFW_KEY_RIGHT_CONTROL);
    m.attr("KEY_RIGHT_ALT") = py::int_(GLFW_KEY_RIGHT_ALT);
    m.attr("KEY_RIGHT_SUPER") = py::int_(GLFW_KEY_RIGHT_SUPER);
    m.attr("KEY_MENU") = py::int_(GLFW_KEY_MENU);
    m.attr("KEY_LAST") = py::int_(GLFW_KEY_LAST);

    m.attr("CLIENT_API") = py::int_(GLFW_CLIENT_API);
    m.attr("NO_API") = py::int_(GLFW_NO_API);
    m.attr("SUBPASS_EXTERNAL") = py::int_(VK_SUBPASS_EXTERNAL);
    m.attr("UINT64_MAX") = py::int_(UINT64_MAX);

    m.attr("INPUT_MODE_CURSOR") = py::int_((int)GLFW_CURSOR);
    m.attr("INPUT_MODE_STICKY_KEYS") = py::int_((int)GLFW_STICKY_KEYS);
    m.attr("INPUT_MODE_STICKY_MOUSE_BUTTONS") = py::int_((int)GLFW_STICKY_MOUSE_BUTTONS);
    m.attr("INPUT_MODE_LOCK_KEY_MODS") = py::int_((int)GLFW_LOCK_KEY_MODS);
    m.attr("INPUT_MODE_RAW_MOUSE_BUTTON") = py::int_((int)GLFW_RAW_MOUSE_MOTION);

    m.attr("ACTION_PRESS") = py::int_((int)GLFW_PRESS);
    m.attr("ACTION_REPEAT") = py::int_((int)GLFW_REPEAT);
    m.attr("ACTION_RELEASE") = py::int_((int)GLFW_RELEASE);
}

PYBIND11_MODULE(pyvku, m) {
    m.doc() = "vku test"; // optional module docstring

    m.attr("GREY") = py::int_((int)STBI_grey);
    m.attr("GREY_ALPHA") = py::int_((int)STBI_grey_alpha);
    m.attr("RGB") = py::int_((int)STBI_rgb);
    m.attr("RGB_ALPHA") = py::int_((int)STBI_rgb_alpha);
   
    bind_vk_aabb_positions_khr(m);
    bind_glfw_key_tokens(m);
    bind_glfw_input_events(m);



    m.def("create_window", create_window)
        .def("init", init)
        .def("terminate", _terminate)
        .def("window_hint", window_hint)
        .def("poll_events", poll_events)
        .def("create_framebuffer", create_framebuffer)
        .def("create_render_pass", create_render_pass)
        .def("create_command_pool", create_command_pool)
        .def("create_semaphore", create_semaphore)
        .def("create_fence", create_fence)
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
        .def("cmd_bind_descriptor_sets", cmd_bind_descriptor_sets)
        .def("cmd_pipeline_barrier", cmd_pipeline_barrier, 
            py::arg("command_buffer"), py::arg("src_stage_mask"), py::arg("dst_stage_mask"), py::arg("dependency_flags"), py::arg("image_memory_barriers"))
        .def("cmd_copy_buffer_to_image", cmd_copy_buffer_to_image)
        .def("cmd_push_constants", cmd_push_constants)
        .def("find_supported_format", find_supported_format, py::arg("physical_device"), py::arg("candidates"), py::arg("tiling"), py::arg("features"));

    register_vku_class<vku::ShaderModule>(m, "ShaderModule");
    register_vku_class<vku::DescriptorPool>(m, "DescriptorPool");
    register_vku_class<vku::Image>(m, "Image")
        .def_readwrite("layout", &vku::Image::layout)
        .def_readwrite("source_stage", &vku::Image::source_stage)
        .def_readwrite("src_access_mask", &vku::Image::src_access_mask);
    register_vku_class<vku::Queue>(m, "Queue");
    register_vku_class<vku::ImageView>(m, "ImageView");

    py::enum_<VkImageTiling>(m, "ImageTiling")
        .value("OPTIMAL", VK_IMAGE_TILING_OPTIMAL)
        .value("LINEAR", VK_IMAGE_TILING_LINEAR);

    py::class_<vku::ImageFactory>(m, "ImageFactory")
        .def(py::init([](PhysicalDevice pd, vku::Device d) {
            return vku::ImageFactory(pd, d);
        }), py::arg("physical_device"), py::arg("device"))
        .def("build_image_2d", [](vku::ImageFactory factory, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageTiling tiling) {
            auto result = factory.build_image_2d(extent, format, usage, properties, tiling);

            if (!result)
            {
                throw std::runtime_error(result.get_error().message());
            }

            return result.get_value();
        }, py::arg("extent"), py::arg("format"), py::arg("usage"), py::arg("properties"), py::arg("tiling") = VK_IMAGE_TILING_OPTIMAL);

    py::class_<InstanceBuilder>(m, "InstanceBuilder")
        .def(py::init<>())
        .def("set_app_name", &InstanceBuilder::set_app_name)
        .def("set_engine_name", &InstanceBuilder::set_engine_name)
        .def("request_validation_layers", &InstanceBuilder::request_validation_layers)
        .def("use_default_debug_messenger", &InstanceBuilder::use_default_debug_messenger)
        .def("require_api_version", &InstanceBuilder::require_api_version)
        .def("enable_extension", &InstanceBuilder::enable_extension)
        .def("build", &InstanceBuilder::build);

    py::class_<vku::Instance>(m, "Instance")
        .def("destroy", &vku::Instance::destroy);

    py::class_<Window, std::shared_ptr<Window>>(m, "Window")
        .def("make_context_current", &Window::make_context_current)
        .def("create_surface", &Window::create_surface)
        .def("should_close", &Window::should_close)
        .def("swap_buffers", &Window::swap_buffers)
        .def("get_events", &Window::get_events);

    register_vku_class<vku::Surface>(m, "Surface");

    py::class_<PhysicalDeviceSelector>(m, "PhysicalDeviceSelector")
        .def(py::init<vku::Instance>())
        .def("set_surface", &PhysicalDeviceSelector::set_surface)
        .def("prefer_gpu_device_type", &PhysicalDeviceSelector::prefer_gpu_device_type)
        .def("allow_any_gpu_device_type", &PhysicalDeviceSelector::allow_any_gpu_device_type)
        .def("select", &PhysicalDeviceSelector::select)
        .def("select_devices", &PhysicalDeviceSelector::select_devices)
        .def("add_required_extension", &PhysicalDeviceSelector::add_required_extension)
        .def("set_required_features", &PhysicalDeviceSelector::set_required_features);

    py::class_<PhysicalDevice>(m, "PhysicalDevice")
        .def("get_extensions", &PhysicalDevice::get_extensions)
        .def("enumerate_extensions", &PhysicalDevice::enumerate_extensions);

    py::class_<vku::Device>(m, "Device")
        .def("get_queue", &vku::Device::get_queue)
        .def("get_queue_index", &vku::Device::get_queue_index)
        .def("wait_idle", &vku::Device::wait_idle)
        .def("destroy", &vku::Device::destroy);

    py::class_<DeviceBuilder>(m, "DeviceBuilder")
        .def(py::init<PhysicalDevice>())
        .def("build", &DeviceBuilder::build);

    py::class_<Swapchain>(m, "Swapchain")
        .def("get_images", &Swapchain::get_images)
        .def("get_image_views", &Swapchain::get_image_views)
        .def_property("image_format", &Swapchain::get_image_format, &Swapchain::set_image_format)
        .def_property_readonly("extent", &Swapchain::get_extent)
        .def("destroy_image_views", &Swapchain::destroy_image_views)
        .def("destroy", &Swapchain::destroy);

    py::class_<SwapchainBuilder>(m, "SwapchainBuilder")
        .def(py::init<vku::Device>())
        .def("set_old_swapchain", &SwapchainBuilder::set_old_swapchain)
        .def("build", &SwapchainBuilder::build);

    py::enum_<vkb::QueueType>(m, "QueueType")
        .value("compute", vkb::QueueType::compute)
        .value("graphics", vkb::QueueType::graphics)
        .value("present", vkb::QueueType::present)
        .value("transfer", vkb::QueueType::transfer);

    py::enum_<VkDependencyFlagBits>(m, "Dependency")
        .value("BY_REGION", VK_DEPENDENCY_BY_REGION_BIT)
        .value("DEVICE_GROUP", VK_DEPENDENCY_DEVICE_GROUP_BIT)
        .value("VIEW_LOCAL", VK_DEPENDENCY_VIEW_LOCAL_BIT)
        .value("VIEW_LOCAL_KHR", VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR)
        .value("DEVICE_GROUP_KHR", VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR)
        .export_values();

    py::class_<FramebufferCreateInfo>(m, "FramebufferCreateInfo")
        .def(py::init<>())
        .def_readwrite("render_pass", &FramebufferCreateInfo::render_pass)
        .def_readwrite("attachments", &FramebufferCreateInfo::attachments)
        .def_readwrite("layers", &FramebufferCreateInfo::layers)
        .def_readwrite("width", &FramebufferCreateInfo::width)
        .def_readwrite("height", &FramebufferCreateInfo::height);

    py::class_<VkImageSubresourceLayers>(m, "ImageSubresourceLayers")
        .def(py::init<VkImageAspectFlags, uint32_t, uint32_t, uint32_t>(), py::arg("aspect_mask"), py::arg("mip_level") = 0, py::arg("array_base_layer")=0, py::arg("layer_count")=0)
        .def_readwrite("aspect_mask", &VkImageSubresourceLayers::aspectMask)
        .def_readwrite("mip_level", &VkImageSubresourceLayers::mipLevel)
        .def_readwrite("base_array_layer", &VkImageSubresourceLayers::baseArrayLayer)
        .def_readwrite("layer_count", &VkImageSubresourceLayers::layerCount);

    py::class_<VkOffset3D>(m, "Offset3D")
        .def(py::init<int32_t, int32_t, int32_t>(),
            py::arg("x") = 0,
            py::arg("y") = 0,
            py::arg("z") = 0)
        .def_readwrite("x", &VkOffset3D::x)
        .def_readwrite("y", &VkOffset3D::y)
        .def_readwrite("z", &VkOffset3D::z);

    py::class_<VkExtent3D>(m, "Extent3D")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
            py::arg("width") = 0,
            py::arg("height") = 0,
            py::arg("depth") = 0)
        .def_readwrite("width", &VkExtent3D::width)
        .def_readwrite("height", &VkExtent3D::height)
        .def_readwrite("depth", &VkExtent3D::depth);

    py::class_<VkBufferImageCopy>(m, "BufferImageCopy")
        .def(py::init<VkDeviceSize, uint32_t, uint32_t, VkImageSubresourceLayers, VkOffset3D, VkExtent3D>(),
            py::arg("buffer_offset") = 0,
            py::arg("buffer_row_length") = 0,
            py::arg("buffer_image_height") = 0,
            py::arg("image_subresource") = VkImageSubresourceLayers{},
            py::arg("image_offset") = VkOffset3D{ 0,0,0 },
            py::arg("image_extent") = VkOffset3D{ 0,0,0 })
        .def_readwrite("buffer_offset", &VkBufferImageCopy::bufferOffset)
        .def_readwrite("buffer_row_length", &VkBufferImageCopy::bufferRowLength)
        .def_readwrite("buffer_image_height", &VkBufferImageCopy::bufferImageHeight)
        .def_readwrite("image_subresource", &VkBufferImageCopy::imageSubresource)
        .def_readwrite("image_offset", &VkBufferImageCopy::imageOffset)
        .def_readwrite("image_extent", &VkBufferImageCopy::imageExtent);

    py::enum_<VkFormat>(m, "Format")
        .value("UNDEFINED", VK_FORMAT_UNDEFINED)
        .value("R4G4_UNORM_PACK8", VK_FORMAT_R4G4_UNORM_PACK8)
        .value("R4G4B4A4_UNORM_PACK16", VK_FORMAT_R4G4B4A4_UNORM_PACK16)
        .value("B4G4R4A4_UNORM_PACK16", VK_FORMAT_B4G4R4A4_UNORM_PACK16)
        .value("R5G6B5_UNORM_PACK16", VK_FORMAT_R5G6B5_UNORM_PACK16)
        .value("B5G6R5_UNORM_PACK16", VK_FORMAT_B5G6R5_UNORM_PACK16)
        .value("R5G5B5A1_UNORM_PACK16", VK_FORMAT_R5G5B5A1_UNORM_PACK16)
        .value("B5G5R5A1_UNORM_PACK16", VK_FORMAT_B5G5R5A1_UNORM_PACK16)
        .value("A1R5G5B5_UNORM_PACK16", VK_FORMAT_A1R5G5B5_UNORM_PACK16)
        .value("R8_UNORM", VK_FORMAT_R8_UNORM)
        .value("R8_SNORM", VK_FORMAT_R8_SNORM)
        .value("R8_USCALED", VK_FORMAT_R8_USCALED)
        .value("R8_SSCALED", VK_FORMAT_R8_SSCALED)
        .value("R8_UINT", VK_FORMAT_R8_UINT)
        .value("R8_SINT", VK_FORMAT_R8_SINT)
        .value("R8_SRGB", VK_FORMAT_R8_SRGB)
        .value("R8G8_UNORM", VK_FORMAT_R8G8_UNORM)
        .value("R8G8_SNORM", VK_FORMAT_R8G8_SNORM)
        .value("R8G8_USCALED", VK_FORMAT_R8G8_USCALED)
        .value("R8G8_SSCALED", VK_FORMAT_R8G8_SSCALED)
        .value("R8G8_UINT", VK_FORMAT_R8G8_UINT)
        .value("R8G8B8A8_SRGB", VK_FORMAT_R8G8B8A8_SRGB)
        .value("R32_SFLOAT", VK_FORMAT_R32_SFLOAT)
        .value("R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT)
        .value("R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT)
        .value("R32G32B32A32_SFLOAT", VK_FORMAT_R32G32B32A32_SFLOAT)
        .value("R64_SFLOAT", VK_FORMAT_R64_SFLOAT)
        .value("R64G64_SFLOAT", VK_FORMAT_R64G64_SFLOAT)
        .value("R64G64B64_SFLOAT", VK_FORMAT_R64G64B64_SFLOAT)
        .value("R64G64B64A64_SFLOAT", VK_FORMAT_R64G64B64A64_SFLOAT)
        .value("B10G11R11_UFLOAT_PACK32", VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        .value("E5B9G9R9_UFLOAT_PACK32", VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)
        .value("D32_SFLOAT", VK_FORMAT_D32_SFLOAT)
        .value("D32_SFLOAT_S8_UINT", VK_FORMAT_D32_SFLOAT_S8_UINT)
        .value("D24_UNORM_S8_UINT", VK_FORMAT_D24_UNORM_S8_UINT);

    py::enum_<VkSampleCountFlagBits>(m, "SampleCount", py::arithmetic())
        .value("_1", VK_SAMPLE_COUNT_1_BIT);

    py::enum_<VkAttachmentLoadOp>(m, "AttachmentLoadOp")
        .value("DONT_CARE", VK_ATTACHMENT_LOAD_OP_DONT_CARE)
        .value("CLEAR", VK_ATTACHMENT_LOAD_OP_CLEAR)
        .value("LOAD", VK_ATTACHMENT_LOAD_OP_LOAD);

    py::enum_<VkAttachmentStoreOp>(m, "AttachmentStoreOp")
        .value("DONT_CARE", VK_ATTACHMENT_STORE_OP_DONT_CARE)
        .value("STORE", VK_ATTACHMENT_STORE_OP_STORE);

    py::enum_<VkImageLayout>(m, "ImageLayout")
        .value("UNDEFINED", VK_IMAGE_LAYOUT_UNDEFINED)
        .value("GENERAL", VK_IMAGE_LAYOUT_GENERAL)
        .value("COLOR_ATTACHMENT_OPTIMAL", VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        .value("DEPTH_STENCIL_ATTACHMENT_OPTIMAL", VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .value("DEPTH_STENCIL_READ_ONLY_OPTIMAL", VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .value("SHADER_READ_ONLY_OPTIMAL", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .value("TRANSFER_SRC_OPTIMAL", VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        .value("TRANSFER_DST_OPTIMAL", VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        .value("PREINITIALIZED", VK_IMAGE_LAYOUT_PREINITIALIZED)
        .value("PRESENT_SRC_KHR", VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    py::class_<VkAttachmentDescription>(m, "AttachmentDescription")
        .def(py::init<uint32_t, VkFormat, VkSampleCountFlagBits, VkAttachmentLoadOp, VkAttachmentStoreOp, VkAttachmentLoadOp, VkAttachmentStoreOp, VkImageLayout, VkImageLayout>(),
            py::arg("flags") = 0,
            py::arg("format") = VK_FORMAT_UNDEFINED,
            py::arg("samples") = VK_SAMPLE_COUNT_1_BIT,
            py::arg("load_op") = VK_ATTACHMENT_LOAD_OP_LOAD,
            py::arg("store_op") = VK_ATTACHMENT_STORE_OP_STORE,
            py::arg("stencil_load_op") = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            py::arg("stencil_store_op") = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            py::arg("initial_layout") = VK_IMAGE_LAYOUT_UNDEFINED,
            py::arg("final_layout") = VK_IMAGE_LAYOUT_UNDEFINED)
        .def_readwrite("flags", &VkAttachmentDescription::flags)
        .def_readwrite("format", &VkAttachmentDescription::format)
        .def_readwrite("samples", &VkAttachmentDescription::samples)
        .def_readwrite("load_op", &VkAttachmentDescription::loadOp)
        .def_readwrite("store_op", &VkAttachmentDescription::storeOp)
        .def_readwrite("stencil_load_op", &VkAttachmentDescription::stencilLoadOp)
        .def_readwrite("stencil_store_op", &VkAttachmentDescription::stencilStoreOp)
        .def_readwrite("initial_layout", &VkAttachmentDescription::initialLayout)
        .def_readwrite("final_layout", &VkAttachmentDescription::finalLayout);
 

    py::class_<VkAttachmentReference>(m, "AttachmentReference")
        .def(py::init<>())
        .def_readwrite("attachment", &VkAttachmentReference::attachment)
        .def_readwrite("layout", &VkAttachmentReference::layout);

    py::enum_<VkImageUsageFlagBits>(m, "ImageUsage", py::arithmetic())
        .value("TRANSFER_SRC", VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST", VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .value("SAMPLED", VK_IMAGE_USAGE_SAMPLED_BIT)
        .value("STORAGE", VK_IMAGE_USAGE_STORAGE_BIT)
        .value("COLOR_ATTACHMENT", VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT", VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        .value("TRANSIENT_ATTACHMENT", VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
        .value("INPUT_ATTACHMENT", VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    py::enum_<VkFormatFeatureFlagBits>(m, "FormatFeature")
        .value("SAMPLED_IMAGE", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
        .value("STORAGE_IMAGE", VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
        .value("STORAGE_IMAGE_ATOMIC", VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)
        .value("UNIFORM_TEXEL_BUFFER", VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)
        .value("STORAGE_TEXEL_BUFFER", VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)
        .value("STORAGE_TEXEL_BUFFER_ATOMIC", VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT)
        .value("VERTEX_BUFFER", VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)
        .value("COLOR_ATTACHMENT", VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
        .value("COLOR_ATTACHMENT_BLEND", VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT", VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
        .value("BLIT_SRC", VK_FORMAT_FEATURE_BLIT_SRC_BIT)
        .value("BLIT_DST", VK_FORMAT_FEATURE_BLIT_DST_BIT)
        .value("SAMPLED_IMAGE_FILTER_LINEAR", VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
        .value("SAMPLED_IMAGE_FILTER_CUBIC_IMG", VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG)
        .value("TRANSFER_SRC", VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST", VK_FORMAT_FEATURE_TRANSFER_DST_BIT)
        .value("MIDPOINT_CHROMA_SAMPLES", VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT)
        .value("DISJOINT", VK_FORMAT_FEATURE_DISJOINT_BIT)
        .value("COSITED_CHROMA_SAMPLES", VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT)
        .value("SAMPLED_IMAGE_FILTER_MINMAX", VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT)
        .value("SAMPLED_IMAGE_FILTER_CUBIC_EXT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT)
        .value("FRAGMENT_DENSITY_MAP", VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT)
        .value("TRANSFER_SRC_KHR", VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR)
        .value("TRANSFER_DST_KHR", VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR)
        .value("MIDPOINT_CHROMA_SAMPLES_KHR", VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT_KHR)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_KHR", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_KHR", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR)
        .value("SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_KHR", VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR);

    py::enum_<VkPipelineBindPoint>(m, "PipelineBindPoint")
        .value("GRAPHICS", VK_PIPELINE_BIND_POINT_GRAPHICS);

    py::class_<VkClearValue>(m, "ClearValue")
        .def(py::init([](std::array<float, 4> values) -> VkClearValue  {
            VkClearValue result;
            result.color.float32[0] = values[0];
            result.color.float32[1] = values[1];
            result.color.float32[2] = values[2];
            result.color.float32[3] = values[3];
            return result;
        }))
        /*
        .def(py::init([](std::array<uint32_t, 4> values) -> VkClearValue {
            VkClearValue result = { {values[0], values[1], values[2], values[3]} };
            return result;
        }))
        */
        .def(py::init([](float depth, uint32_t stencil) -> VkClearValue {
            VkClearValue result;
            result.depthStencil = VkClearDepthStencilValue{depth, stencil};
            return result;
        }))
        .def_readwrite("color", &VkClearValue::color)
        .def_readwrite("depth_stencil", &VkClearValue::depthStencil);

    py::class_< SubpassDescription>(m, "SubpassDescription")
        .def(py::init<>())
        .def_readwrite("pipeline_bind_point", &SubpassDescription::pipeline_bind_point)
        .def_readwrite("color_attachments", &SubpassDescription::color_attachments)
        .def_readwrite("depth_stencil_attachments", &SubpassDescription::depth_stencil_attachments);

    py::class_<VkSubpassDependency>(m, "SubpassDependency")
        .def(py::init<>())
        .def_readwrite("src_subpass", &VkSubpassDependency::srcSubpass)
        .def_readwrite("dst_subpass", &VkSubpassDependency::dstSubpass)
        .def_readwrite("src_stage_mask", &VkSubpassDependency::srcStageMask)
        .def_readwrite("dst_stage_mask", &VkSubpassDependency::dstStageMask)
        .def_readwrite("src_access_mask", &VkSubpassDependency::srcAccessMask)
        .def_readwrite("dst_access_mask", &VkSubpassDependency::dstAccessMask)
        .def_readwrite("dependency_flags", &VkSubpassDependency::dependencyFlags);

    py::enum_< VkPipelineStageFlagBits>(m, "PipelineStage", py::arithmetic())
        .value("TOP_OF_PIPE", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
        .value("DRAW_INDIRECT", VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
        .value("VERTEX_INPUT", VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
        .value("VERTEX_SHADER", VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
        .value("TESSELLATION_CONTROL_SHADER", VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT)
        .value("TESSELLATION_EVALUATION_SHADER", VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT)
        .value("GEOMETRY_SHADER", VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT)
        .value("FRAGMENT_SHADER", VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
        .value("EARLY_FRAGMENT_TESTS", VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
        .value("LATE_FRAGMENT_TESTS", VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
        .value("COLOR_ATTACHMENT_OUTPUT", VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
        .value("COMPUTE_SHADER", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        .value("TRANSFER", VK_PIPELINE_STAGE_TRANSFER_BIT)
        .value("BOTTOM_OF_PIPE", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        .value("HOST", VK_PIPELINE_STAGE_HOST_BIT)
        .value("ALL_GRAPHICS", VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        .value("ALL_COMMANDS", VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    py::enum_< VkAccessFlagBits>(m, "Access", py::arithmetic())
        .value("INDIRECT_COMMAND_READ", VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
        .value("INDEX_READ", VK_ACCESS_INDEX_READ_BIT)
        .value("VERTEX_ATTRIBUTE_READ", VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
        .value("UNIFORM_READ", VK_ACCESS_UNIFORM_READ_BIT)
        .value("INPUT_ATTACHMENT_READ", VK_ACCESS_INPUT_ATTACHMENT_READ_BIT)
        .value("SHADER_READ", VK_ACCESS_SHADER_READ_BIT)
        .value("SHADER_WRITE", VK_ACCESS_SHADER_WRITE_BIT)
        .value("COLOR_ATTACHMENT_READ", VK_ACCESS_COLOR_ATTACHMENT_READ_BIT)
        .value("COLOR_ATTACHMENT_WRITE", VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT_READ", VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT_WRITE", VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        .value("TRANSFER_READ", VK_ACCESS_TRANSFER_READ_BIT)
        .value("TRANSFER_WRITE", VK_ACCESS_TRANSFER_WRITE_BIT)
        .value("HOST_READ", VK_ACCESS_HOST_READ_BIT)
        .value("HOST_WRITE", VK_ACCESS_HOST_WRITE_BIT)
        .value("MEMORY_READ", VK_ACCESS_MEMORY_READ_BIT)
        .value("MEMORY_WRITE", VK_ACCESS_MEMORY_WRITE_BIT);


    py::class_<VkImageSubresourceRange>(m, "ImageSubresourceRange")
        .def(py::init<VkImageAspectFlags, uint32_t, uint32_t, uint32_t, uint32_t>(), py::arg("aspect_mask") = 0, py::arg("base_mip_level") = 0, py::arg("level_count") = 1, py::arg("base_array_layer") = 0, py::arg("layer_count") = 1)
        .def_readwrite("aspect_mask", &VkImageSubresourceRange::aspectMask)
        .def_readwrite("base_mip_level", &VkImageSubresourceRange::baseMipLevel)
        .def_readwrite("level_count", &VkImageSubresourceRange::levelCount)
        .def_readwrite("base_array_layer", &VkImageSubresourceRange::baseArrayLayer)
        .def_readwrite("layer_count", &VkImageSubresourceRange::layerCount);

    py::class_<VkImageMemoryBarrier>(m, "ImageMemoryBarrier")
        .def(py::init([](VkAccessFlags src_access_flags, VkAccessFlags dst_access_flags, VkImageLayout old_layout, VkImageLayout new_layout,
            uint32_t src_queue_family_index, uint32_t dst_queue_family_index, const vku::Image &image, VkImageSubresourceRange &subresource_rang) {
        VkImageMemoryBarrier result{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                src_access_flags,
                dst_access_flags,
                old_layout,
                new_layout,
                src_queue_family_index,
                dst_queue_family_index,
                image,
                subresource_rang
            };
            return result;
        }), py::arg("src_access_mask"), py::arg("dst_access_mask"), py::arg("old_layout") = VK_IMAGE_LAYOUT_UNDEFINED, py::arg("new_layout")=VK_IMAGE_LAYOUT_UNDEFINED, py::arg("src_queue_family_index")=VK_QUEUE_FAMILY_IGNORED,
            py::arg("dst_queue_family_index") = VK_QUEUE_FAMILY_IGNORED, py::arg("image") = vku::Image{}, py::arg("subresource_range") = VkImageSubresourceRange{})
        .def_readwrite("s_type", &VkImageMemoryBarrier::sType)
        .def_readwrite("p_next", &VkImageMemoryBarrier::pNext)
        .def_readwrite("src_access_mask", &VkImageMemoryBarrier::srcAccessMask)
        .def_readwrite("dst_access_mask", &VkImageMemoryBarrier::dstAccessMask)
        .def_readwrite("old_layout", &VkImageMemoryBarrier::oldLayout)
        .def_readwrite("new_layout", &VkImageMemoryBarrier::newLayout)
        .def_readwrite("src_queue_family_index", &VkImageMemoryBarrier::srcQueueFamilyIndex)
        .def_readwrite("dst_queue_family_index", &VkImageMemoryBarrier::dstQueueFamilyIndex)
        .def_readwrite("subresource_range", &VkImageMemoryBarrier::subresourceRange);


    py::enum_<VkImageAspectFlagBits>(m, "ImageAspect")
        .value("COLOR", VK_IMAGE_ASPECT_COLOR_BIT)
        .value("DEPTH", VK_IMAGE_ASPECT_DEPTH_BIT)
        .value("STENCIL", VK_IMAGE_ASPECT_STENCIL_BIT)
        .value("METADATA", VK_IMAGE_ASPECT_METADATA_BIT);

    py::class_< RenderPassCreateInfo>(m, "RenderPassCreateInfo")
        .def(py::init<>())
        .def_readwrite("subpasses", &RenderPassCreateInfo::subpasses)
        .def_readwrite("attachments", &RenderPassCreateInfo::attachments)
        .def_readwrite("dependencies", &RenderPassCreateInfo::dependencies);

    register_vku_class<vku::RenderPass>(m, "RenderPass");

    py::class_<VkExtent2D>(m, "Extent2D")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t>())
        .def_readwrite("width", &VkExtent2D::width)
        .def_readwrite("height", &VkExtent2D::height);

    py::class_<VkOffset2D>(m, "Offset2D")
        .def(py::init<int32_t, int32_t>(), py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &VkOffset2D::x)
        .def_readwrite("y", &VkOffset2D::y);

    register_vku_class<vku::DescriptorSetLayout>(m, "DescriptorSetLayout");
    register_vku_class<vku::DescriptorSet>(m, "DescriptorSet");
    register_vku_class<vku::Framebuffer>(m, "Framebuffer");
    register_vku_class<vku::Fence>(m, "Fence");
    register_vku_class<vku::Semaphore>(m, "Semaphore");

    register_vku_class<vku::Pipeline>(m, "Pipeline")
        .def("get_layout", [&](vku::Pipeline& self) -> vku::PipelineLayout {
            return vku::PipelineLayout(self.get_parent(), self.get_layout());
        });

    register_vku_class<vku::PipelineLayout>(m, "PipelineLayout");

    py::class_<FenceCreateInfo>(m, "FenceCreateInfo")
        .def(py::init<>())
        .def(py::init<VkFenceCreateFlags>())
        .def_readwrite("flags", &FenceCreateInfo::flags);

    py::enum_<VkFenceCreateFlagBits>(m, "FenceCreate")
        .value("SIGNALED", VK_FENCE_CREATE_SIGNALED_BIT);

    py::class_<CommandPoolCreateInfo>(m, "CommandPoolCreateInfo")
        .def(py::init<>())
        .def_readwrite("flags", &CommandPoolCreateInfo::flags)
        .def_readwrite("queue_family_index", &CommandPoolCreateInfo::queue_family_index);

    py::enum_<VkCommandPoolCreateFlagBits>(m, "CommandPoolCreate")
        .value("RESET_COMMAND_BUFFER", VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    
    register_vku_class<vku::CommandPool>(m, "CommandPool");

    py::class_<VkCommandBufferAllocateInfo>(m, "CommandBufferAllocateInfo")
        .def(py::init([](vku::CommandPool cp, VkCommandBufferLevel bl, uint32_t cbc) {
            VkCommandBufferAllocateInfo result{};

            result.commandBufferCount = cbc;
            result.commandPool = cp;
            result.level = bl;
            result.pNext = nullptr;
            result.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

            return result;
        }), py::arg("command_pool"), py::arg("level"), py::arg("command_buffer_count"))
        .def_readwrite("level", &VkCommandBufferAllocateInfo::level)
        // .def_readwrite("command_pool", &VkCommandBufferAllocateInfo::commandPool)
        .def_readwrite("command_buffer_count", &VkCommandBufferAllocateInfo::commandBufferCount);

    py::enum_<VkCommandBufferLevel>(m, "CommandBufferLevel")
        .value("PRIMARY", VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        .value("SECONDARY", VK_COMMAND_BUFFER_LEVEL_SECONDARY);

    register_vku_class<vku::CommandBuffer>(m, "CommandBuffer");

    py::register_exception<SwapchainOutOfDateError>(m, "SwapchainOutOfDateError");


    py::enum_< VkCommandBufferResetFlagBits>(m, "CommandBufferReset")
        .value("RELEASE_RESOURCES", VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

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
        .def(py::init<uint32_t, uint32_t, VkVertexInputRate>(), py::arg("binding"), py::arg("stride"), py::arg("input_rate"))
        .def_readwrite("binding", &VkVertexInputBindingDescription::binding)
        .def_readwrite("input_rate", &VkVertexInputBindingDescription::inputRate)
        .def_readwrite("stride", &VkVertexInputBindingDescription::stride);

    py::class_<VkVertexInputAttributeDescription>(m, "VertexInputAttributeDescription")
        .def(py::init<uint32_t, uint32_t, VkFormat, uint32_t>(), py::arg("location")=0, py::arg("binding")=0, py::arg("format")=VK_FORMAT_R32G32B32_SFLOAT, py::arg("offset")=0)
        .def_readwrite("binding", &VkVertexInputAttributeDescription::binding)
        .def_readwrite("format", &VkVertexInputAttributeDescription::format)
        .def_readwrite("location", &VkVertexInputAttributeDescription::location)
        .def_readwrite("offset", &VkVertexInputAttributeDescription::offset);

    py::enum_< VkVertexInputRate>(m, "VertexInputRate")
        .value("VERTEX", VK_VERTEX_INPUT_RATE_VERTEX)
        .value("INSTANCE", VK_VERTEX_INPUT_RATE_INSTANCE);

    py::class_<PipelineLayoutBuilder>(m, "PipelineLayoutBuilder")
        .def(py::init<vku::Device>())
        .def("add_descriptor_set", &PipelineLayoutBuilder::add_descriptor_set)
        .def("add_push_constant_range", &PipelineLayoutBuilder::add_push_constant_range)
        .def("build", &PipelineLayoutBuilder::build);




    py::class_<VkPipelineDepthStencilStateCreateInfo> pipeline_depth_stencil_state_create_info(m, "PipelineDepthStencilStateCreateInfo");
    pipeline_depth_stencil_state_create_info.def(py::init([](bool depth_test_enable, bool depth_write_enable, VkCompareOp depth_compare_op, bool depth_bounds_test_enable, bool stencil_test_enable, VkStencilOpState front, VkStencilOpState back, float min_depth_bounds, float max_depth_bounds) {
        VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo = {};
        pipelineDepthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        pipelineDepthStencilStateCreateInfo.pNext = nullptr;
        pipelineDepthStencilStateCreateInfo.depthTestEnable = depth_test_enable;
        pipelineDepthStencilStateCreateInfo.depthWriteEnable = depth_write_enable;
        pipelineDepthStencilStateCreateInfo.depthCompareOp = depth_compare_op;
        pipelineDepthStencilStateCreateInfo.depthBoundsTestEnable = depth_bounds_test_enable;
        pipelineDepthStencilStateCreateInfo.stencilTestEnable = stencil_test_enable;
        pipelineDepthStencilStateCreateInfo.front = front;
        pipelineDepthStencilStateCreateInfo.back = back;
        pipelineDepthStencilStateCreateInfo.minDepthBounds = min_depth_bounds;
        pipelineDepthStencilStateCreateInfo.maxDepthBounds = max_depth_bounds;
        return pipelineDepthStencilStateCreateInfo;
        }), py::arg("depth_test_enable"), py::arg("depth_write_enable"), py::arg("depth_compare_op"), py::arg("depth_bounds_test_enable"), py::arg("stencil_test_enable"), py::arg("front"), py::arg("back"), py::arg("min_depth_bounds"), py::arg("max_depth_bounds"))
        .def_readwrite("s_type", &VkPipelineDepthStencilStateCreateInfo::sType)
        .def_readwrite("p_next", &VkPipelineDepthStencilStateCreateInfo::pNext)
        .def_readwrite("flags", &VkPipelineDepthStencilStateCreateInfo::flags)
        .def_readwrite("depth_test_enable", &VkPipelineDepthStencilStateCreateInfo::depthTestEnable)
        .def_readwrite("depth_write_enable", &VkPipelineDepthStencilStateCreateInfo::depthWriteEnable)
        .def_readwrite("depth_compare_op", &VkPipelineDepthStencilStateCreateInfo::depthCompareOp)
        .def_readwrite("depth_bounds_test_enable", &VkPipelineDepthStencilStateCreateInfo::depthBoundsTestEnable)
        .def_readwrite("stencil_test_enable", &VkPipelineDepthStencilStateCreateInfo::stencilTestEnable)
        .def_readwrite("front", &VkPipelineDepthStencilStateCreateInfo::front)
        .def_readwrite("back", &VkPipelineDepthStencilStateCreateInfo::back)
        .def_readwrite("min_depth_bounds", &VkPipelineDepthStencilStateCreateInfo::minDepthBounds)
        .def_readwrite("max_depth_bounds", &VkPipelineDepthStencilStateCreateInfo::maxDepthBounds);


    py::class_< GraphicsPipelineBuilder>(m, "GraphicsPipelineBuilder")
        .def(py::init<vku::Device>())
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
        .value("BACK", VK_CULL_MODE_BACK_BIT)
        .value("FRONT_AND_BACK", VK_CULL_MODE_FRONT_AND_BACK)
        .value("FRONT", VK_CULL_MODE_FRONT_BIT)
        .value("NONE", VK_CULL_MODE_NONE);

    py::class_< BufferFactory>(m, "BufferFactory")
        .def(py::init<vku::Device, PhysicalDevice>())
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
            if (buffer.get_mapped() == nullptr)
            {
                throw std::runtime_error("called .get_mapped() on a buffer which hasn't been mapped yet.");
            }
            return py::memoryview::from_memory(buffer.get_mapped(), buffer.get_size());
         })
         .def("get_size", &vku::Buffer::get_size);

    py::enum_<VkBufferUsageFlagBits>(m, "BufferUsage", py::arithmetic())
        .value("TRANSFER_SRC", VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST", VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        .value("UNIFORM_TEXEL_BUFFER", VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)
        .value("STORAGE_TEXEL_BUFFER", VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)
        .value("UNIFORM_BUFFER", VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        .value("STORAGE_BUFFER", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        .value("INDEX_BUFFER", VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        .value("VERTEX_BUFFER", VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        .value("INDIRECT_BUFFER", VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    py::enum_<VkMemoryPropertyFlagBits>(m, "MemoryProperty", py::arithmetic())
        .value("DEVICE_LOCAL", VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        .value("HOST_VISIBLE", VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        .value("HOST_COHERENT", VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        .value("HOST_CACHED", VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        .value("LAZILY_ALLOCATED", VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);

    py::class_< VkPipelineColorBlendAttachmentState>(m, "PipelineColorBlendAttachmentState")
        .def(py::init<>())
        .def_readwrite("blend_enable", &VkPipelineColorBlendAttachmentState::blendEnable)
        .def_readwrite("color_write_mask", &VkPipelineColorBlendAttachmentState::colorWriteMask);


    py::enum_<VkColorComponentFlagBits>(m, "ColorComponent", py::arithmetic())
        .value("R", VK_COLOR_COMPONENT_R_BIT)
        .value("G", VK_COLOR_COMPONENT_G_BIT)
        .value("B", VK_COLOR_COMPONENT_B_BIT)
        .value("A", VK_COLOR_COMPONENT_A_BIT);

    py::enum_<VkShaderStageFlagBits>(m, "ShaderStage", py::arithmetic())
        .value("ALL", VK_SHADER_STAGE_ALL)
        .value("FRAGMENT", VK_SHADER_STAGE_FRAGMENT_BIT)
        .value("VERTEX", VK_SHADER_STAGE_VERTEX_BIT);

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
            .def(py::init<vku::Device, vku::CommandPool, vku::Queue>())
            .def("enter", &SingleTimeCommandExecutor::enter)
            .def("exit", &SingleTimeCommandExecutor::exit)
            .def("wait", &SingleTimeCommandExecutor::wait)
            .def("destroy", &SingleTimeCommandExecutor::destroy)
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
            .def("get_data", &ImageData::get_data)
            .def_property_readonly("extent", &ImageData::extent)
            .def_property_readonly("size", &ImageData::size);

        py::class_< DescriptorPoolBuilder>(m, "DescriptorPoolBuilder")
            .def(py::init<vku::Device>())
            .def("build", &DescriptorPoolBuilder::build)
            .def("add_descriptor_sets", &DescriptorPoolBuilder::add_descriptor_sets, py::arg("descriptor_set"), py::arg("count")=1);

        py::class_< DescriptorSetLayoutBuilder>(m, "DescriptorSetLayoutBuilder")
            .def(py::init<vku::Device>())
            .def("build", &DescriptorSetLayoutBuilder::build)
            .def("add_binding", &DescriptorSetLayoutBuilder::add_binding);

        py::class_< DescriptorSetBuilder>(m, "DescriptorSetBuilder")
            .def(py::init<vku::Device, vku::DescriptorPool, vku::DescriptorSetLayout>())
            .def("build", &DescriptorSetBuilder::build)
            .def("write_uniform_buffer", &DescriptorSetBuilder::write_uniform_buffer)
            .def("write_storage_buffer", &DescriptorSetBuilder::write_storage_buffer)
            .def("write_combined_image_sampler", &DescriptorSetBuilder::write_combined_image_sampler);

        py::enum_<VkDescriptorType>(m, "DescriptorType")
            .value("SAMPLER", VK_DESCRIPTOR_TYPE_SAMPLER)
            .value("COMBINED_IMAGE_SAMPLER", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            .value("SAMPLED_IMAGE", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
            .value("STORAGE_IMAGE", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            .value("UNIFORM_TEXEL_BUFFER", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
            .value("STORAGE_TEXEL_BUFFER", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
            .value("UNIFORM_BUFFER", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            .value("STORAGE_BUFFER", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .value("UNIFORM_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
            .value("STORAGE_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
            .value("INPUT_ATTACHMENT", VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

        py::class_<vku::ImageViewFactory>(m, "ImageViewFactory")
            .def(py::init([](const vku::Device& device) -> vku::ImageViewFactory { return vku::ImageViewFactory(device);  }))
            .def("build2D", [](const vku::ImageViewFactory &self, const vku::Image& image, VkFormat format, VkImageAspectFlags aspect) -> vku::ImageView {
                auto res = self.build2D(image, format, aspect);
                // TODO: error handling.
                return res.get_value();
             }, py::arg("image"), py::arg("format"), py::arg("aspect") = VK_IMAGE_ASPECT_COLOR_BIT);

        register_vku_class<vku::Sampler>(m, "Sampler");

        py::class_<vku::SamplerFactory>(m, "SamplerFactory")
            .def(py::init([](const vku:: Device& device, const PhysicalDevice& physical_device) -> vku::SamplerFactory {
                return vku::SamplerFactory(device, physical_device);
             }))
            .def("build", [](const vku::SamplerFactory& self) -> vku::Sampler {
                auto res = self.build();
                // TODO: error handling.
                return res.get_value();
             });

        py::class_< VkExtensionProperties>(m, "ExtensionProperties")
            .def_readonly("extension_name", &VkExtensionProperties::extensionName)
            .def_readonly("spec_version", &VkExtensionProperties::specVersion);

        py::class_<VkPhysicalDeviceFeatures>(m, "PhysicalDeviceFeatures")
            .def(py::init<>())
            .def_readwrite("robust_buffer_access", &VkPhysicalDeviceFeatures::robustBufferAccess)
            .def_readwrite("full_draw_index_uint32", &VkPhysicalDeviceFeatures::fullDrawIndexUint32)
            .def_readwrite("image_cube_array", &VkPhysicalDeviceFeatures::imageCubeArray)
            .def_readwrite("independent_blend", &VkPhysicalDeviceFeatures::independentBlend)
            .def_readwrite("geometry_shader", &VkPhysicalDeviceFeatures::geometryShader)
            .def_readwrite("tessellation_shader", &VkPhysicalDeviceFeatures::tessellationShader)
            .def_readwrite("sample_rate_shading", &VkPhysicalDeviceFeatures::sampleRateShading)
            .def_readwrite("dual_src_blend", &VkPhysicalDeviceFeatures::dualSrcBlend)
            .def_readwrite("logic_op", &VkPhysicalDeviceFeatures::logicOp)
            .def_readwrite("multi_draw_indirect", &VkPhysicalDeviceFeatures::multiDrawIndirect)
            .def_readwrite("draw_indirect_first_instance", &VkPhysicalDeviceFeatures::drawIndirectFirstInstance)
            .def_readwrite("depth_clamp", &VkPhysicalDeviceFeatures::depthClamp)
            .def_readwrite("depth_bias_clamp", &VkPhysicalDeviceFeatures::depthBiasClamp)
            .def_readwrite("fill_mode_non_solid", &VkPhysicalDeviceFeatures::fillModeNonSolid)
            .def_readwrite("depth_bounds", &VkPhysicalDeviceFeatures::depthBounds)
            .def_readwrite("wide_lines", &VkPhysicalDeviceFeatures::wideLines)
            .def_readwrite("large_points", &VkPhysicalDeviceFeatures::largePoints)
            .def_readwrite("alpha_to_one", &VkPhysicalDeviceFeatures::alphaToOne)
            .def_readwrite("multi_viewport", &VkPhysicalDeviceFeatures::multiViewport)
            .def_readwrite("sampler_anisotropy", &VkPhysicalDeviceFeatures::samplerAnisotropy)
            .def_readwrite("texture_compression_etc2", &VkPhysicalDeviceFeatures::textureCompressionETC2);

        py::class_<vku::BufferAddressGetter>(m, "BufferAddressGetter");
        //    .def(py::init([]() {}))


        py::class_<VkPushConstantRange>(m, "PushConstantRange")
            .def(py::init<VkShaderStageFlags, uint32_t, uint32_t>(),
                py::arg("stage_flags"), py::arg("offset"), py::arg("size"))
            .def_readwrite("stage_flags", &VkPushConstantRange::stageFlags)
            .def_readwrite("offset", &VkPushConstantRange::offset)
            .def_readwrite("size", &VkPushConstantRange::size);

        py::class_< Imgui, std::shared_ptr<Imgui>>(m, "Imgui")
            .def(py::init<std::shared_ptr<Window>, vku::Instance, PhysicalDevice, vku::Device, vku::Queue, uint32_t, vku::CommandPool, VkSampleCountFlagBits>(),
                py::arg("window"), py::arg("instance"), py::arg("physical_device"), py::arg("device"), py::arg("queue"), py::arg("queue_family"), py::arg("command_pool"), py::arg("sample_count") = VK_SAMPLE_COUNT_1_BIT)
            .def("init", &Imgui::init)
            .def("set_image_count", &Imgui::set_image_count)
            .def("wait_init", &Imgui::wait_init)
            .def("new_frame", &Imgui::new_frame)
            .def("render", &Imgui::render)
            .def("destroy", &Imgui::destroy);
}


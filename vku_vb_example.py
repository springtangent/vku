import sys
sys.path.append("C:\\Users\\bafre\\vku\\build\\Debug")
import pyvku as vku
import struct

MAX_FRAMES_IN_FLIGHT = 2
VERTICES = [
    (( 0.0, -0.5, 0.0), (1.0, 0.0, 0.0)),
    (( 0.5,  0.5, 0.0), (0.0, 1.0, 0.0)),
    ((-0.5,  0.5, 0.0), (0.0, 0.0, 1.0))
]

vec3 = struct.Struct("fff")

VERTEX_BYTES = b''.join(vec3.pack(*position) + vec3.pack(*color) for position, color in VERTICES)


def get_binding_description(binding=0):
    result = vku.VertexInputBindingDescription()

    result.binding = binding
    result.stride = vec3.size * 2
    result.input_rate = vku.VertexInputRate.VERTEX

    return result


def get_attribute_descriptions(binding=0):
    POSITION_LOCATION = 0
    COLOR_LOCATION = 1

    position_desc = vku.VertexInputAttributeDescription()
    position_desc.binding = binding
    position_desc.location = POSITION_LOCATION
    position_desc.format = vku.Format.R32G32B32_SFLOAT
    position_desc.offset = 0

    color_desc = vku.VertexInputAttributeDescription()
    color_desc.binding = binding
    color_desc.location = COLOR_LOCATION
    color_desc.format = vku.Format.R32G32B32_SFLOAT
    color_desc.offset = vec3.size

    return [position_desc, color_desc]


class Scene:
    pass


class VulkanContext:
    def __init__(self):
        super().__init__()
        self.current_frame = 0
        self.physical_device = None
        self.instance = None
        self.device = None
        self.window = None
        self.surface = None
        self.swapchain = None

    def init(self):
        if not vku.init():
            raise RuntimeError("vku failed to initialize")

        self.create_window()
        self.create_instance()
        self.create_surface()
        self.select_physical_device()
        self.create_device()
        self.create_swapchain()
        self.create_render_pass()
        self.create_framebuffers()
        self.create_sync_objects()
        self.create_command_pool()
        self.allocate_command_buffers()

    def destroy(self):
        self.device.wait_idle()

        self.destroy_sync_objects()
        self.destroy_command_pool()
        self.destroy_framebuffers()

        vku.destroy_swapchain(self.device, self.swapchain)
        vku.destroy_renderpass(self.device, self.render_pass)
        vku.destroy_device(self.device)
        vku.destroy_surface(self.instance, self.surface)
        vku.destroy_instance(self.instance)

    def create_window(self):
        vku.window_hint(vku.CLIENT_API, vku.NO_API)
        self.window = vku.create_window(640, 480, "Hello World")
        if not self.window:
            vku.terminate()
            raise RuntimeError("failed to create window")
        self.window.make_context_current()

    def create_instance(self):
        # create the vulkan instance
        instance_builder = vku.InstanceBuilder()
        instance_builder.request_validation_layers()
        instance_builder.use_default_debug_messenger()
        instance_builder.set_app_name("VKU Example Application")
        instance_builder.set_engine_name("No Engine")
        instance_builder.require_api_version(1, 3, 0)

        # TODO: handle exception if it occurs.
        self.instance = instance_builder.build()

    def create_surface(self):
        self.surface = self.window.create_surface(self.instance)

    def select_physical_device(self):
        physical_device_selector = vku.PhysicalDeviceSelector(self.instance)
        physical_device_selector.set_surface(self.surface)
        physical_device_selector.prefer_gpu_device_type(vku.PreferredDeviceType.discrete)

        # TODO: error handling.
        physical_devices = physical_device_selector.select_devices()

        for pd in physical_devices:
            pdprops = vku.get_physical_device_properties(pd)
            print(pdprops.api_version)
            print(pdprops.device_name)
            print(pdprops.vendor_id)
            print(pdprops.device_type)

        self.physical_device = physical_device_selector.select()

    def create_device(self):
        device_builder = vku.DeviceBuilder(self.physical_device)

        # TODO: error handling.
        self.device = device_builder.build()

        self.graphics_queue = self.device.get_queue(vku.QueueType.graphics)
        self.present_queue = self.device.get_queue(vku.QueueType.present)

    def create_swapchain(self):
        swapchain_builder = vku.SwapchainBuilder(self.device)
        # swapchain_builder.set_old_swapchain(None)
        self.swapchain = swapchain_builder.build()

    def recreate_swapchain(self):
        self.device.wait_idle()

        self.destroy_command_pool()
        self.destroy_framebuffers()

        swapchain_builder = vku.SwapchainBuilder(self.device)
        swapchain_builder.set_old_swapchain(self.swapchain)
        swapchain = swapchain_builder.build()
        vku.destroy_swapchain(self.device, self.swapchain)
        self.swapchain = swapchain

        self.create_framebuffers()
        self.create_command_pool()
        self.allocate_command_buffers()

    def create_render_pass(self):
        # create render pass
        color_attachment = vku.AttachmentDescription()
        color_attachment.format = self.swapchain.image_format
        color_attachment.samples = vku.SampleCount._1_BIT
        color_attachment.load_op = vku.AttachmentLoadOp.CLEAR
        color_attachment.store_op = vku.AttachmentStoreOp.STORE
        color_attachment.stencil_load_op = vku.AttachmentLoadOp.DONT_CARE
        color_attachment.stencil_store_op = vku.AttachmentStoreOp.DONT_CARE
        color_attachment.initial_layout = vku.ImageLayout.UNDEFINED
        color_attachment.final_layout = vku.ImageLayout.PRESENT_SRC_KHR

        color_attachment_ref = vku.AttachmentReference()
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = vku.ImageLayout.COLOR_ATTACHMENT_OPTIMAL

        subpass = vku.SubpassDescription()
        subpass.pipeline_bind_point = vku.PipelineBindPoint.GRAPHICS
        subpass.color_attachments = [ color_attachment_ref ]

        dependency = vku.SubpassDependency();
        dependency.src_subpass = vku.SUBPASS_EXTERNAL
        dependency.dst_subpass = 0
        dependency.src_stage_mask = vku.PipelineStage.COLOR_ATTACHMENT_OUTPUT_BIT
        dependency.src_access_mask = 0
        dependency.dst_stage_mask = vku.PipelineStage.COLOR_ATTACHMENT_OUTPUT_BIT
        dependency.dst_access_mask = vku.Access.COLOR_ATTACHMENT_READ_BIT | vku.Access.COLOR_ATTACHMENT_WRITE_BIT

        render_pass_info = vku.RenderPassCreateInfo()
        render_pass_info.attachments = [ color_attachment ]
        render_pass_info.subpasses = [ subpass ]
        render_pass_info.dependencies = [ dependency ]

        # TODO: error handling.
        self.render_pass = vku.create_render_pass(self.device, render_pass_info)

    def create_framebuffers(self):
        # create framebuffers
        self.swapchain_images = self.swapchain.get_images()
        self.swapchain_image_views = self.swapchain.get_image_views()
        self.framebuffers = []

        for image, image_view in zip(self.swapchain_images, self.swapchain_image_views):
            framebuffer_info = vku.FramebufferCreateInfo()
            framebuffer_info.render_pass = self.render_pass
            framebuffer_info.attachments = [ image_view ]
            framebuffer_info.width = self.swapchain.extent.width
            framebuffer_info.height = self.swapchain.extent.height
            framebuffer_info.layers = 1

            self.framebuffers.append(vku.create_framebuffer(self.device, framebuffer_info))

    def destroy_framebuffers(self):
        for fb in self.framebuffers:
            vku.destroy_framebuffer(self.device, fb)
        self.framebuffers.clear()
        self.swapchain.destroy_image_views(self.swapchain_image_views)
        self.swapchain_image_views.clear()

    def create_sync_objects(self):
        # create sync objects
        fence_info = vku.FenceCreateInfo(vku.FenceCreate.SIGNALED_BIT)

        self.available_semaphores = [vku.create_semaphore(self.device) for i in range(MAX_FRAMES_IN_FLIGHT)]
        self.finished_semaphores = [vku.create_semaphore(self.device) for i in range(MAX_FRAMES_IN_FLIGHT)]
        self.in_flight_fences = [vku.create_fence(self.device, fence_info) for i in range(MAX_FRAMES_IN_FLIGHT)]

    def destroy_sync_objects(self):
        for s in self.available_semaphores:
            vku.destroy_semaphore(self.device, s)

        for s in self.finished_semaphores:
            vku.destroy_semaphore(self.device, s)

        for f in self.in_flight_fences:
            vku.destroy_fence(self.device, f)

    def create_command_pool(self):
        # create command pool
        pool_info = vku.CommandPoolCreateInfo()
        pool_info.flags = vku.CommandPoolCreate.RESET_COMMAND_BUFFER_BIT

        # TODO: error handling
        pool_info.queue_family_index = self.device.get_queue_index(vku.QueueType.graphics);

        # TODO: error handling
        self.command_pool = vku.create_command_pool(self.device, pool_info)

    def destroy_command_pool(self):
        vku.destroy_command_pool(self.device, self.command_pool) 

    def update_frame_data(self, frame_index):
        pass

    def begin_record_command_buffer(self):
        command_buffer = self.command_buffers[self.current_frame]

        vku.begin_command_buffer(command_buffer)

        viewport = vku.Viewport(0.0, 0.0,  self.swapchain.extent.width, self.swapchain.extent.height, 0.0, 1.0)

        scissor = vku.Rect2D(vku.Offset2D(0,0), self.swapchain.extent)

        # it seems like these could be methods of the command buffer.
        vku.cmd_set_viewport(command_buffer, 0, 1, [viewport])
        vku.cmd_set_scissor(command_buffer, 0, 1, [scissor])

        return command_buffer

    def get_frame_buffer(self):
        return self.framebuffers[self.image_index]

    def begin_render_pass(self):
        command_buffer = self.command_buffers[self.current_frame]

        # TODO: should this begin the first render pass, or what?
        # TODO: should there be a context handler for recording command buffers, and recording render passes?
        render_pass_info = vku.RenderPassBeginInfo();
        render_pass_info.render_pass = self.render_pass
        render_pass_info.framebuffer = self.framebuffers[self.image_index]
        render_pass_info.render_area = vku.Rect2D(vku.Offset2D(0, 0), self.swapchain.extent)
        render_pass_info.clear_values = [ vku.ClearValue.colorf(0.0, 0.0, 0.0, 1.0) ]
        vku.cmd_begin_render_pass(command_buffer, render_pass_info, vku.SubpassContents.INLINE)

    def end_render_pass(self):
        vku.cmd_end_render_pass(self.command_buffers[self.current_frame])

    def end_record_command_buffer(self):
        vku.end_command_buffer(self.command_buffers[self.current_frame])

    def allocate_command_buffers(self):
        # create command buffers
        alloc_info = vku.CommandBufferAllocateInfo()
        alloc_info.command_pool = self.command_pool
        alloc_info.level = vku.CommandBufferLevel.PRIMARY
        alloc_info.command_buffer_count = MAX_FRAMES_IN_FLIGHT

        # TODO: error handling
        self.command_buffers = vku.allocate_command_buffers(self.device, alloc_info)

    def begin_draw_frame(self):
        vku.wait_for_fences(self.device, [self.in_flight_fences[self.current_frame]], True, vku.UINT64_MAX)

        self.image_index = vku.acquire_next_image(self.device, self.swapchain, vku.UINT64_MAX, self.available_semaphores[self.current_frame])

        vku.reset_fences(self.device, [self.in_flight_fences[self.current_frame]])
        vku.reset_command_buffer(self.command_buffers[self.current_frame], 0)

        return self.current_frame

    def draw_frame(self):
        try:
            self.begin_draw_frame()
        except vku.SwapchainOutOfDateError as sc_error:
            self.recreate_swapchain()
            return

        # TODO: this needs to be done outside of this class,
        #       in a callback or something.
        self.update_frame_data(self.current_frame)

        # TODO: this needs to be done outside of this class,
        #       in a callback or something.
        self.record_command_buffer()

        self.end_draw_frame()

    def end_draw_frame(self):
        submit_info = vku.SubmitInfo()
        submit_info.wait_semaphores = [ self.available_semaphores[self.current_frame] ]
        submit_info.wait_dst_stage_masks = [ vku.PipelineStage.COLOR_ATTACHMENT_OUTPUT_BIT ]
        submit_info.command_buffers = [ self.command_buffers[self.current_frame] ]
        submit_info.signal_semaphores = [ self.finished_semaphores[self.current_frame] ]

        vku.queue_submit(self.graphics_queue, [submit_info], self.in_flight_fences[self.current_frame])

        present_info = vku.PresentInfo()
        present_info.wait_semaphores = submit_info.signal_semaphores
        present_info.swapchains = [ self.swapchain ]
        present_info.image_indices = [ self.image_index ]

        try:
            vku.queue_present(self.present_queue, present_info)
        except vku.SwapchainOutOfDateError:
            self.framebuffer_resized = False
            self.recreate_swapchain()

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;


class SharedContext:
    def __init__(self, vulkan_context):
        super().__init__()
        self.vulkan_context = vulkan_context

    def init(self):
        self.buffer_factory = vku.BufferFactory(self.vulkan_context.device, self.vulkan_context.physical_device)
        pipeline_layout_builder = vku.PipelineLayoutBuilder(self.vulkan_context.device)
        self.pipeline_layout = pipeline_layout_builder.build()

        self.vertex_buffer = self.buffer_factory.build(len(VERTEX_BYTES), vku.BufferUsage.VERTEX_BUFFER_BIT, vku.MemoryProperty.HOST_VISIBLE_BIT | vku.MemoryProperty.HOST_COHERENT_BIT)
        self.vertex_buffer.map_memory()
        mem = self.vertex_buffer.get_mapped()
        mem[:] = VERTEX_BYTES
        self.vertex_buffer.unmap_memory()
        self.create_graphics_pipelines()

    def destroy(self):
        self.vertex_buffer.destroy()
        self.pipeline_layout.destroy()
        self.graphics_pipeline.destroy()

    def create_graphics_pipelines(self):
        pipeline_builder = vku.GraphicsPipelineBuilder(self.vulkan_context.device)

        with open("resources/shaders/triangle_buffer.vert.spv", "rb") as vert_file:
            vert_bytes = vert_file.read()

        with open("resources/shaders/triangle_buffer.frag.spv", "rb") as frag_file:
            frag_bytes = frag_file.read()

        print(len(vert_bytes))
        print(len(frag_bytes))

        vert_module = vku.create_shader_module(self.vulkan_context.device, vert_bytes)
        frag_module = vku.create_shader_module(self.vulkan_context.device, frag_bytes)

        colorBlendAttachment = vku.PipelineColorBlendAttachmentState()
        colorBlendAttachment.color_write_mask = vku.ColorComponent.R_BIT | vku.ColorComponent.G_BIT | vku.ColorComponent.B_BIT | vku.ColorComponent.A_BIT;
        colorBlendAttachment.blend_enable = False

        swapchain = self.vulkan_context.swapchain

        pipeline_builder.add_shader_stage(vku.ShaderStage.VERTEX_BIT, vert_module)
        pipeline_builder.add_shader_stage(vku.ShaderStage.FRAGMENT_BIT, frag_module)
        pipeline_builder.add_viewport(vku.Viewport(0.0, 0.0, swapchain.extent.width, swapchain.extent.height, 0.0, 1.0))
        pipeline_builder.add_scissor(vku.Rect2D(vku.Offset2D(0, 0), vku.Extent2D(swapchain.extent.width, swapchain.extent.height)))
        pipeline_builder.add_dynamic_state(vku.DynamicState.VIEWPORT)
        pipeline_builder.add_dynamic_state(vku.DynamicState.SCISSOR)
        pipeline_builder.set_pipeline_layout(self.pipeline_layout)
        pipeline_builder.set_render_pass(self.vulkan_context.render_pass)
        pipeline_builder.add_color_blend_attachment(colorBlendAttachment)
        pipeline_builder.add_vertex_binding(get_binding_description())
        pipeline_builder.add_vertex_attributes(get_attribute_descriptions())

        self.graphics_pipeline = pipeline_builder.build()

        vert_module.destroy()
        frag_module.destroy()


class FrameContext:
    def __init__(self, vulkan_context, shared_context):
        super().__init__()
        self.vulkan_context = vulkan_context
        self.shared_context = shared_context

    def init(self):
        pass

    def destroy(self):
        pass

    def record_command_buffer(self, command_buffer):
        pass

 


class GraphicsContext:
    def __init__(self):
        super().__init__()
        self.vulkan_context = VulkanContext()
        self.shared_context = SharedContext(self.vulkan_context)
        self.frame_contexts = [FrameContext(self.vulkan_context, self.shared_context) for i in range(MAX_FRAMES_IN_FLIGHT)]

    def init(self):
        self.vulkan_context.init()
        self.shared_context.init()
        for fc in self.frame_contexts:
            fc.init()

    def destroy(self):
        self.vulkan_context.device.wait_idle()

        for fc in self.frame_contexts:
            fc.destroy()
        self.shared_context.destroy()
        self.vulkan_context.destroy()
        vku.terminate()

    def record_command_buffer(self, command_buffer, scene):
        vku.cmd_bind_pipeline(command_buffer, vku.PipelineBindPoint.GRAPHICS, self.shared_context.graphics_pipeline)

        vertex_buffers = [ self.shared_context.vertex_buffer ]
        offsets = [ 0 ]
        vku.cmd_bind_vertex_buffers(command_buffer, 0, 1, vertex_buffers, offsets)

        vku.cmd_draw(command_buffer, len(VERTICES), 1, 0, 0)

    def draw_frame(self, scene):
        try:
            current_frame = self.vulkan_context.begin_draw_frame()
        except vku.SwapchainOutOfDateError as sc_error:
            self.vulkan_context.recreate_swapchain()
            return

        command_buffer = self.vulkan_context.begin_record_command_buffer()
        frame_context = self.frame_contexts[current_frame]

        self.vulkan_context.begin_render_pass()

        self.record_command_buffer(command_buffer, scene)

        self.vulkan_context.end_render_pass()

        self.vulkan_context.end_record_command_buffer()

        self.vulkan_context.end_draw_frame()

    def swap_buffers(self):
        return self.vulkan_context.window.swap_buffers()

    def poll_events(self):
        vku.poll_events()

    def should_close(self):
        return self.vulkan_context.window.should_close()



class Application:
    def __init__(self):
        super().__init__()
        self.graphics_context = GraphicsContext()
        self.scene = Scene()

    def main(self):
        print("supported extensions:", vku.get_supported_extensions())
        print("vulkan version:", vku.get_vulkan_version())
        self.graphics_context.init()
        pdprops = vku.get_physical_device_properties(self.graphics_context.vulkan_context.physical_device)
        print(pdprops.api_version)
        print(pdprops.device_name)
        print(pdprops.device_type)
        print(pdprops.vendor_id)

        # Loop until the user closes the window
        while not self.graphics_context.should_close():
            # Render here, e.g. using pyOpenGL
            self.graphics_context.draw_frame(self.scene)

            # Swap front and back buffers
            self.graphics_context.swap_buffers()

            # Poll for and process events
            self.graphics_context.poll_events()

            # TODO: update the scene


        self.graphics_context.destroy()


def main():
    application = Application()
    application.main()



if __name__ == "__main__":
    main()
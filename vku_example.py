import sys
sys.path.append("C:\\Users\\bafre\\vku\\build\\Debug")
import pyvku as vku

MAX_FRAMES_IN_FLIGHT = 2


class Application:
    def __init__(self):
        super().__init__()
        self.current_frame = 0
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
        instance_builder.require_api_version(1, 0, 0)

        # TODO: handle exception.
        self.instance = instance_builder.build()
        print(self.instance)

    def create_surface(self):
        self.surface = self.window.create_surface(self.instance)
        print(self.surface)

    def select_physical_device(self):
        physical_device_selector = vku.PhysicalDeviceSelector(self.instance)
        physical_device_selector.set_surface(self.surface)

        # TODO: error handling.
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

    def record_command_buffer_render_pass(self):
        pass

    def record_command_buffer(self, image_index):
        command_buffer = self.command_buffers[self.current_frame]
        vku.begin_command_buffer(command_buffer)

        render_pass_info = vku.RenderPassBeginInfo();
        render_pass_info.render_pass = self.render_pass
        render_pass_info.framebuffer = self.framebuffers[image_index]
        render_pass_info.render_area = vku.Rect2D(vku.Offset2D(0, 0), self.swapchain.extent)
        render_pass_info.clear_values = [ vku.ClearValue.colorf(0.0, 0.0, 0.0, 1.0) ]

        viewport = vku.Viewport(0.0, 0.0,  self.swapchain.extent.width, self.swapchain.extent.height, 0.0, 1.0)

        scissor = vku.Rect2D(vku.Offset2D(0,0), self.swapchain.extent)

        # it seems like these could be methods of the command buffer.
        vku.cmd_set_viewport(command_buffer, 0, 1, [viewport])
        vku.cmd_set_scissor(command_buffer, 0, 1, [scissor])
        vku.cmd_begin_render_pass(command_buffer, render_pass_info, vku.SubpassContents.INLINE)

        self.record_command_buffer_render_pass()

        vku.cmd_end_render_pass(command_buffer)
        vku.end_command_buffer(command_buffer)

    def allocate_command_buffers(self):
        # create command buffers
        alloc_info = vku.CommandBufferAllocateInfo()
        alloc_info.command_pool = self.command_pool
        alloc_info.level = vku.CommandBufferLevel.PRIMARY
        alloc_info.command_buffer_count = MAX_FRAMES_IN_FLIGHT

        # TODO: error handling
        self.command_buffers = vku.allocate_command_buffers(self.device, alloc_info)

    def draw_frame(self):
        vku.wait_for_fences(self.device, [self.in_flight_fences[self.current_frame]], True, vku.UINT64_MAX)

        try:
            image_index = vku.acquire_next_image(self.device, self.swapchain, vku.UINT64_MAX, self.available_semaphores[self.current_frame])
        except vku.SwapchainOutOfDateError as sc_error:
            self.recreate_swapchain()
            return
        except RuntimeError as re:
            raise

        self.update_frame_data(self.current_frame)
        vku.reset_fences(self.device, [self.in_flight_fences[self.current_frame]])
        vku.reset_command_buffer(self.command_buffers[self.current_frame], 0)

        self.record_command_buffer(image_index)

        submit_info = vku.SubmitInfo()
        submit_info.wait_semaphores = [ self.available_semaphores[self.current_frame] ]
        submit_info.wait_dst_stage_masks = [ vku.PipelineStage.COLOR_ATTACHMENT_OUTPUT_BIT ]
        submit_info.command_buffers = [ self.command_buffers[self.current_frame] ]
        submit_info.signal_semaphores = [ self.finished_semaphores[self.current_frame] ]

        vku.reset_fences(self.device, [ self.in_flight_fences[self.current_frame] ])
        vku.reset_command_buffer(self.command_buffers[self.current_frame], 0)
        self.record_command_buffer(image_index);

        vku.queue_submit(self.graphics_queue, [submit_info], self.in_flight_fences[self.current_frame])


        present_info = vku.PresentInfo()
        present_info.wait_semaphores = submit_info.signal_semaphores
        present_info.swapchains = [ self.swapchain ]
        present_info.image_indices = [ image_index ]

        try:
            vku.queue_present(self.present_queue, present_info)
        except vku.SwapchainOutOfDateError:
            self.framebuffer_resized = False
            self.recreate_swapchain()
        except RuntimeError:
            raise

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;


def main():
    app = Application()
    app.init()

    # Loop until the user closes the window
    while not app.window.should_close():
        # Render here, e.g. using pyOpenGL
        app.draw_frame()

        # Swap front and back buffers
        app.window.swap_buffers()

        # Poll for and process events
        vku.poll_events()


    app.destroy()

    vku.terminate()


if __name__ == "__main__":
    main()
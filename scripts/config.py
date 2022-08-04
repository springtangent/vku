import os
import os.path


VULKAN_GLSLC_EXECUTABLE = "C:/VulkanSDK/1.2.198.1/Bin/glslc.exe" # os.environ.get("Vulkan_GLSLC_EXECUTABLE")

RESOURCE_PATH = "resources"
BUILD_ASSET_PATH = os.path.join("build", "resources")

SHADER_SOURCE_EXTENSIONS = ('.vert', '.frag')
SHADER_OBJECT_EXTENSIONS = ('.spv',)
TEXTURE_EXTENSIONS = ('.jpg','.png','.tga')


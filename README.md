# vku
Vulkan Utility Libraries

## build examples

This builds a CMake project containing the code.

"python -m scripts.build"

## build shaders

compile shaders and copy compiled shaders to the build.
"python -m scripts.buildresources"

This has only been tested on windows. you will have to edit scripts/config.py to reflect the location of your glsc executable.

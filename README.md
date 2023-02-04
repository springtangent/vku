# vku
Vulkan Utility Libraries

## checkout/update submodules

This downloads required submodules.

"git submodule update --init --recursive"

## create projects

This creates projects from the CMake

"cmake . build"

## build examples

This builds the visual studio projects created from cmake (windows only)

"python -m scripts.build"

## build shaders for examples

compile shaders and copy compiled shaders to the build. TODO: this requires manually updating "scripts/config.py" with the path of your GLSC executable.
"python -m scripts.buildresources"

This has only been tested on windows. you will have to edit scripts/config.py to reflect the location of your glsc executable.

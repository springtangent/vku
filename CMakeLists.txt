cmake_minimum_required(VERSION 3.20)

project(vku_examples VERSION 1.0
                     DESCRIPTION "vku examples"
                     LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan)

if (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Cannot find Vulkan library")
endif()

message(STATUS "Found Vulkan; Vulkan_GLSLC_EXECUTABLE=${Vulkan_GLSLC_EXECUTABLE}")

# disable building docs, tests and examples for glfw.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(VKU_EXAMPLES_EXTERN_DIR "extern")

add_subdirectory("${VKU_EXAMPLES_EXTERN_DIR}/glfw")
add_subdirectory("${VKU_EXAMPLES_EXTERN_DIR}/vk-bootstrap")
add_subdirectory("${VKU_EXAMPLES_EXTERN_DIR}/pybind11")
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/glm")
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/stb")

include_directories(include ${Vulkan_INCLUDE_DIRS})
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/glm")

add_executable(vku_example examples/example.cpp)
target_link_libraries(vku_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

add_executable(vku_buffer_example examples/vertex_buffer_example.cpp)
target_link_libraries(vku_buffer_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

add_executable(vku_device_buffer_example examples/device_vertex_buffer_example.cpp)
target_link_libraries(vku_device_buffer_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

add_executable(vku_uniform_buffer_example examples/uniform_buffer_example.cpp)
target_link_libraries(vku_uniform_buffer_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

# find_package(pybind11 REQUIRED)
# pybind11_add_module(vku_examples source/pyvku.cpp)

add_library(pyvku MODULE source/pyvku.cpp)
target_link_libraries(pyvku PRIVATE pybind11::module pybind11::lto pybind11::windows_extras ${Vulkan_LIBRARIES} vk-bootstrap::vk-bootstrap glfw)
pybind11_extension(pyvku)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(pyvku)
endif()
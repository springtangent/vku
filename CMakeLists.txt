cmake_minimum_required(VERSION 3.20)

project(vku_examples VERSION 1.0
                     DESCRIPTION "vku examples"
                     LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

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
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/glm")

include_directories(include ${Vulkan_INCLUDE_DIRS})
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/glm")

add_executable(vku_example examples/example.cpp)
target_link_libraries(vku_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

add_executable(vku_buffer_example examples/vertex_buffer_example.cpp)
target_link_libraries(vku_buffer_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

add_executable(vku_device_buffer_example examples/device_vertex_buffer_example.cpp)
target_link_libraries(vku_device_buffer_example PRIVATE ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)

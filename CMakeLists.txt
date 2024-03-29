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
add_subdirectory("${VKU_EXAMPLES_EXTERN_DIR}/fmt")
add_subdirectory("${VKU_EXAMPLES_EXTERN_DIR}/SPIRV-Cross")

set(IMGUI_DIR "${VKU_EXAMPLES_EXTERN_DIR}/imgui")
include_directories("${IMGUI_DIR}")

set(IMGUIZMO_DIR "${VKU_EXAMPLES_EXTERN_DIR}/ImGuizmo")
include_directories("${IMGUIZMO_DIR}")
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/glm")
include_directories("${VKU_EXAMPLES_EXTERN_DIR}/stb")

set(IMNODES_DIR "${VKU_EXAMPLES_EXTERN_DIR}/imnodes")
include_directories("${IMNODES_DIR}")

set(IMGUI_BACKENDS_DIR "${IMGUI_DIR}/backends")
set(IMGUI_CPP_FILES 
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp"
    "${IMGUI_BACKENDS_DIR}/imgui_impl_vulkan.cpp"
)

set(IMGUIZMO_CPP_FILES 
    "${IMGUIZMO_DIR}/ImGuizmo.cpp"
)

set(IMNODES_CPP_FILES
    "${IMNODES_DIR}/imnodes.cpp"
)

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

add_library(pyvku MODULE source/pyvku.cpp ${IMGUI_CPP_FILES} ${IMGUIZMO_CPP_FILES} ${IMNODES_CPP_FILES})
target_link_libraries(pyvku PRIVATE pybind11::module pybind11::lto pybind11::windows_extras ${Vulkan_LIBRARIES} vk-bootstrap::vk-bootstrap glfw fmt::fmt)
pybind11_extension(pyvku)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(pyvku)
endif()
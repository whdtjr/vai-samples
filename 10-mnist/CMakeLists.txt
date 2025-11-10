cmake_minimum_required(VERSION 3.16)

project(10-mnist)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb REQUIRED)

add_executable(${PROJECT_NAME}
    main.cpp
    test.cpp
    vulkanApp.cpp
    spirvHelpers.cpp
    stb_image.cpp
    jsonParser.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    /home/js/workspace/vcpkg/installed/x64-linux/lib/libspirv-reflect-static.a
    ${Vulkan_LIBRARIES}
    glslang::glslang
    glslang::glslang-default-resource-limits
    nlohmann_json::nlohmann_json
)
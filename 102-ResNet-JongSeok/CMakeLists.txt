cmake_minimum_required(VERSION 3.16) # cmake 최소

project(102-ResNet-JongSeok) # project 이름

set(CMAKE_CXX_STANDARD 20) # C++ 20 표준 이용
set(CMAKE_CXX_STANDARD_REQUIRED ON) # C++ 20이 꼭 지원
set(CMAKE_CXX_EXTENSIONS OFF) # 확장 기능 사용 안함

add_executable(${PROJECT_NAME} # 컴파일 후 링킹하는 것들
    main.cpp
    vulkanApp.cpp
    neuralNodes.cpp
    spirvHelpers.cpp
    stb_image.cpp
    npzLoader.cpp
)

if (NOT TARGET spirv-reflect-static) # 중복 방지
    add_library(spirv-reflect-static
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/SPIRV-Reflect/spirv_reflect.c
    ) # SPIRV-Reflect 정적 라이브러리 추가
    target_include_directories(spirv-reflect-static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/SPIRV-Reflect
    ) # SPIRV-Reflect 헤더 파일 경로 추가
endif()

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
) # CMakelists.txt 경로, 상위 directory 경로

find_package(Vulkan REQUIRED) # Vulkan 패키지 찾기
find_package(glfw3 CONFIG REQUIRED) # glfw3 패키지 찾기

# 헤더 파일 경로 및 링크할 라이브러리 설정
target_include_directories(${PROJECT_NAME} PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/SPIRV-Reflect
)
# 링크할 라이브러리 설정
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    z
)

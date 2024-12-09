cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

project(Project)

set(CMAKE_CXX_STANDARD 17)

set(MKL_INCLUDE_DIR /home/zbwu/soft/intel/oneapi/mkl/latest/include)
set(MKL_LIB_DIR /home/zbwu/soft/intel/oneapi/mkl/latest/lib/intel64)

add_definitions(-DUSE_MKL)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -fopenmp -Wall")

set(TARGET_NAME a.out)

file(GLOB_RECURSE SOURCES "*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/blas-test")
list(FILTER SOURCES EXCLUDE REGEX "/CMakeFiles/")
message(STATUS "Filtered sources: ${SOURCES}")

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${MKL_INCLUDE_DIR})

target_link_directories(${TARGET_NAME} PRIVATE ${MKL_LIB_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE mkl_rt m)

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${CMAKE_SOURCE_DIR}/${TARGET_NAME}
    COMMENT "Moving the generated '${TARGET_NAME}' to the parent directory"
)

set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

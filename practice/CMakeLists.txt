cmake_minimum_required(VERSION 3.24)

project(leetgpu_cuda_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

set(PRACTICE_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

function(add_practice_target target_name exercise_dir source_file output_name)
  add_executable(${target_name} "${exercise_dir}/${source_file}")
  target_include_directories(${target_name} PRIVATE "${PRACTICE_COMMON_DIR}")
  set_target_properties(
    ${target_name}
    PROPERTIES
      OUTPUT_NAME "${output_name}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${exercise_dir}"
      CUDA_SEPARABLE_COMPILATION OFF
  )
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endfunction()

add_practice_target(practice_00_sanity 00-sanity-check cuda_sanity.cu cuda_sanity)
add_practice_target(practice_01_vector_add 01-vector-add exercise.cu vector_add)
add_practice_target(practice_02_matrix_copy 02-matrix-copy exercise.cu matrix_copy)
add_practice_target(practice_03_matrix_transpose 03-matrix-transpose exercise.cu matrix_transpose)
add_practice_target(practice_04_reduction 04-reduction exercise.cu reduction)
add_practice_target(practice_05_softmax 05-softmax exercise.cu softmax)
add_practice_target(practice_06_rmsnorm 06-rmsnorm exercise.cu rmsnorm)
add_practice_target(practice_07_gelu 07-gelu exercise.cu gelu)
add_practice_target(practice_08_rope 08-rope exercise.cu rope)
add_practice_target(practice_09_optimized_reduction 09-optimized-reduction exercise.cu optimized_reduction)
add_practice_target(practice_10_tiled_transpose 10-tiled-transpose exercise.cu tiled_transpose)
add_practice_target(practice_11_saxpy 11-saxpy exercise.cu saxpy)
add_practice_target(practice_12_layernorm 12-layernorm exercise.cu layernorm)
add_practice_target(practice_13_vector_dot 13-vector-dot exercise.cu vector_dot)
add_practice_target(practice_14_histogram 14-histogram exercise.cu histogram)
add_practice_target(practice_15_embedding_lookup 15-embedding-lookup exercise.cu embedding_lookup)

# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
)

add_executable(embeddings_model_sample embeddings_model_sample.cpp)
target_link_libraries(embeddings_model_sample PRIVATE openvino::genai)

set_target_properties(embeddings_model_sample PROPERTIES
    COMPILE_PDB_NAME embeddings_model_sample
    # Ensure the executable is placed in a directory that can find the library
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples"
)

install(TARGETS embeddings_model_sample
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL
)

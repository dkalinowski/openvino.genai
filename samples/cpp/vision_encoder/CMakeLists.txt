# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(vision_encoder_sample vision_encoder_sample.cpp)

target_link_libraries(vision_encoder_sample PRIVATE openvino::genai ${OpenCV_LIBRARIES})
target_include_directories(vision_encoder_sample PRIVATE ${OpenCV_INCLUDE_DIRS})

set_target_properties(vision_encoder_sample PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

install(TARGETS vision_encoder_sample
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)

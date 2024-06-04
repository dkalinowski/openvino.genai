# How to create environment to debug and develop continious batching project with OpenVINO:

1. Build OpenVINO with python bindings:
```
cd /path/to/openvino
mkdir build
cd build
cmake -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE={ov_build_type} ..
make -j24
```
2. Set PYTHONPATH, LD_LIBRARY_PATH and OpenVINO_DIR environment variables:
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/openvino/bin/intel64/{ov_build_type}
export PYTHONPATH=${PYTHONPATH}:/path/to/openvino/bin/intel64/Release/python:/path/to/openvino/tools/ovc
export OpenVINO_DIR=/path/to/openvino/{ov_build_type}
```
3. Build OpenVINO tokenizers:
```
cd /path/to/openvino.genai/thirdparty/openvino_tokenizers
mkdir build
cd build
cmake -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE={ov_build_type} ..
make -j24
```
4. Create virtual environment to generate models and run python tests:
> NOTE: Comment installation of `openvino` and `openvino_tokenizers` to your env in `/path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/python/tests/requirements.txt
```
cd /path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching
python3 -m venv .env
source .env/bin/activate
pip3 install -r python/tests/requirements.txt
```
5. Install `openvino_tokenizers` to your virtual environment:
```
cd /path/to/openvino.genai/thirdparty/openvino_tokenizers
export OpenVINO_DIR=/path/to/openvino/build
pip install --no-deps .
```
6. Create build directory in `continious batching` project:
```
mkdir /path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/build
```
7. Generate cmake project:
```
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DOpenVINO_DIR=/path/to/openvino/build -DENABLE_APPS=ON -DENABLE_PYTHON=ON ..
```
8. Build the project
```
make -j24
```
9. Extend `PYTHONPATH` by `continious batching`:
```
export PYTHONPATH=${PYTHONPATH}:/path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/build/python
```
10. Run python tests:
```
cd python/tests
pytest .
```
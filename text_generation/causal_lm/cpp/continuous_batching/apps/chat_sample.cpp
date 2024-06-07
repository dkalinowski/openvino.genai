// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "continuous_batching_pipeline.hpp"
#include "tokenizer.hpp"

int main(int argc, char* argv[]) {//try {
    // Command line options

    cxxopts::Options options("chat_sample", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string models_path = result["model"].as<std::string>();

    // Perform the inference
    
    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 256,
        // cache params
        .cache_size = 8,
        .block_size = 32,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = true,
        // vLLM specific params
        .max_num_seqs = 2,
    };

    chat_t sample_chat;
    std::unordered_map<std::string, std::string> entry;
    entry.emplace("role", "system");
    entry.emplace("content", "You are a pirate chatbot who always responds in pirate speak!");
    sample_chat.push_back(entry);
    entry.clear();
    
    entry.emplace("role", "user");
    entry.emplace("content", "Who are you?");
    sample_chat.push_back(entry);

    entry.clear();
    entry.emplace("role", "assistant");
    entry.emplace("content", R"(Arrrr, shiver me timbers! 'Tis a fine day to be havin' a chat with ye, matey! Me name be Captain Blackbeak, the most feared and infamous pirate to ever sail the seven seas! *adjusts eye patch*

So, what be bringin' ye to these waters? Are ye here to plunder some booty or just to share a pint o' grog with a scurvy dog like meself? *winks*tell me about your ship, the Black Swan. What makes it so special?)");
    sample_chat.push_back(entry);

    entry.clear();
    entry.emplace("role", "user");
    entry.emplace("content", "My ship is full of goods, how about yours?");
    sample_chat.push_back(entry);

    ContinuousBatchingPipeline pipe(models_path, scheduler_config);
    std::cout << "Applying template..." << std::endl;
    std::string prompt = pipe.get_tokenizer()->apply_chat_template(sample_chat);
    std::cout << "Input prompt: \n[" << prompt << "]" << std::endl;
    //exit(1);

    auto input_tensor = pipe.get_tokenizer()->encode(prompt);
    std::vector<int64_t> input_tokens(input_tensor.data<int64_t>(), input_tensor.data<int64_t>() + input_tensor.get_size());

    std::cout << "Input tokens: \n";
    for (auto token: input_tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    GenerationConfig sampling_params = GenerationConfig::greedy();
    sampling_params.max_new_tokens = 100;
    sampling_params.ignore_eos = false;
    auto results = pipe.generate(std::vector<std::string>{prompt}, std::vector<GenerationConfig>{sampling_params});
    if (results[0].m_status != GenerationStatus::FINISHED)
        std::cout << "Generation not finished properly" << std::endl;
    else
        std::cout << "Model response: " << results[0].m_generation_ids[0] << std::endl;


    // Model response: Arrrr, me hearty! *chuckles* Me ship, the Black Swan, be full o' treasure, indeed! *adjusts eye patch* But I be havin' somethin'
    // even more valuable than gold doubloons or shiny pieces o' silver. *winks* I be havin' me trusty parrot, Polly! She be the best navigator and 
    // lookout a pirate could ask for! *pets

    // For now this sample is used to check template processing. 
    // Ultimately we could make it a full, interactive chat sample. 
// } catch (const std::exception& error) {
//     std::cerr << "Error: " << error.what() << '\n';
//     return EXIT_FAILURE;
// } catch (...) {
//     std::cerr << "Non-exception object thrown\n";
//     return EXIT_FAILURE;
}

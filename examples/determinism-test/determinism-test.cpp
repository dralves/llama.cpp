#include "arg.h"      // from llama.cpp: for command-line argument parsing
#include "common.h"   // from llama.cpp: for common_params, common_init_from_params, etc.
#include "llama.h"    // from llama.cpp: for model and context APIs
#include "sampling.h" // from llama.cpp: for token sampling
#include <iostream>
#include <fstream>    // for std::ifstream
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>     // for std::numeric_limits<int>::max()
#include <iomanip>    // for std::setprecision, etc.
#include <chrono>      // optionally for build timestamp (if you want to embed date/time)
#include <sstream>    // for capturing tokens/sec logs if needed
#include <mutex>      // if you want thread-safe logging

// EXAMPLE: llama_log_callback signature:
//   void my_custom_logger(void * user_data, llama_log_level level, const char * message);

// Add forward declaration of our custom logger:
static std::ofstream  g_combined_log; // file stream for combined logging

static void my_custom_logger(ggml_log_level level, const char * message, void * /*user_data*/) {
    // Print to console
    std::cout << message;

    // Also to file
    if (g_combined_log.is_open()) {
        g_combined_log << message;
    }
}

int main(int argc, char **argv) {
    // Parse command line flags (the same approach used in server.cpp).
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        std::cerr << "Error: Failed to parse arguments.\n";
        return 1;
    }

    // Some global initialization, as done in server.cpp
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // ------------------------------------------------------------------
    // 1) Open a combined log file for the custom logger
    g_combined_log.open("determinism_results.txt");
    if (!g_combined_log.is_open()) {
        std::cerr << "Error: cannot open combined_log.txt for logging.\n";
        return 1;
    }
    llama_log_set(my_custom_logger, /*user_data=*/nullptr);

    // Load the model from the parsed flags
    std::cout << "Loading model: " << params.model << std::endl;

    // This might appear in your logs or output file

    common_init_result init_result = common_init_from_params(params);
    if (!init_result.model || !init_result.context) {
        std::cerr << "Error: Unable to load the model.\n";
        return 1;
    }

    llama_model   * model = init_result.model.get();
    llama_context * ctx   = init_result.context.get();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // We'll interpret --prompt as the path to a file that has multiple lines to read.
    if (params.prompt_file.empty()) {
        std::cerr << "Error: Must pass --file=<filename> for multi-line input.\n";
        return 1;
    }

    std::ifstream infile(params.prompt_file);
    if (!infile.is_open()) {
        std::cerr << "Error: cannot open " << params.prompt_file << "\n";
        return 1;
    }

    // Create the sampler, exactly as the server does
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        std::cerr << "Error: Unable to init sampler.\n";
        return 1;
    }

    // REMOVE references to outFile. Instead, log with my_custom_logger.
    // For example, we print the param summary:
    {
        std::stringstream ss;
        ss << "== Determinism Test Parameters ==\n"
           << "model       : " << params.model << "\n"
           << "n_batch     : " << params.n_batch << "\n"
           << "n_predict   : " << params.n_predict << "\n"
           << "seed        : " << params.sampling.seed << "\n"
           << "temperature : " << params.sampling.temp << "\n"
           << "----------------------------------\n\n";

        my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
    }

    // Keep a running "sessionTokens" vector that contains all tokens generated so far.
    std::vector<llama_token> sessionTokens;
    int n_ctx = llama_n_ctx(ctx);

    // Decide how many new tokens are allowed to be generated per line:
    int max_new_tokens = (params.n_predict < 0)
        ? std::numeric_limits<int>::max()
        : params.n_predict;

    std::string line;
    // Read the file line by line
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }

        // Instead of outFile, log prompt using my_custom_logger:
        {
            std::stringstream ss;
            ss << "Prompt: " << line << "\n";
            my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
        }

        // Tokenize this new line
        std::vector<llama_token> newTokens = common_tokenize(vocab, line, /* add_special= */ true);
        if (newTokens.empty()) {
            continue;
        }

        // Merge these new tokens into our existing session.
        sessionTokens.insert(sessionTokens.end(), newTokens.begin(), newTokens.end());

        // Evaluate the newly added tokens (single batch)
        {
            llama_batch batch = llama_batch_init(params.n_batch, /*n_pos=*/0, /*n_seq_id=*/1);
            size_t offset = sessionTokens.size() - newTokens.size();

            // Add each new token to the batch
            for (size_t i = 0; i < newTokens.size(); i++) {
                bool get_logits = (i == newTokens.size() - 1); // only get logits for the last token
                common_batch_add(batch, sessionTokens[offset + i], offset + i, {0}, get_logits);
            }

            // decode the entire batch
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << "Error: decoding new line failed.\n";
                llama_batch_free(batch);
                break; // or return 1; up to you
            }
            llama_batch_free(batch);

            // Accept these new tokens in the sampler
            for (auto t : newTokens) {
                common_sampler_accept(smpl, t, /*is_selected=*/false);
            }
        }

        // Now generate new tokens in response to this line, using the updated context.
        std::cout << "\n[Generating after line: \"" << line << "\"]\n";

        std::string generated_text;

        // For logging logits in one line, store (token, logit) as we generate
        std::vector<std::pair<llama_token, float>> generated_logits;

        // Generate tokens in response
        for (int i = 0; i < max_new_tokens; i++) {
            int n_past = llama_get_kv_cache_token_count(ctx);
            if (n_past + 1 >= n_ctx) {
                std::cout << "\n[Terminated: context limit.]\n";
                break;
            }

            // Sample the next token (like in server.cpp)
            llama_token id = common_sampler_sample(smpl, ctx, /*idx_last_token=*/-1);

            // If the model says we've reached EOS
            if (llama_vocab_is_eog(vocab, id)) {
                std::cout << "\n[Terminated: EOS token.]\n";
                break;
            }

            // Convert token => text string; optionally mark as special if needed
            std::string text = common_token_to_piece(ctx, id, /*special=*/false);
            generated_text += text;
            std::cout << text << std::flush;

            // Accept this token in the sampler and also store it in session
            common_sampler_accept(smpl, id, true);
            sessionTokens.push_back(id);

            // Evaluate the newly sampled token
            {
                llama_batch batch_next = llama_batch_init(1, 0, 1);
                common_batch_add(batch_next, id, n_past, {0}, true);

                if (llama_decode(ctx, batch_next) != 0) {
                    std::cerr << "\nError: decode failed while generating.\n";
                    llama_batch_free(batch_next);
                    break;
                }
                llama_batch_free(batch_next);
            }

            // For single-line logits logging:
            const float* logits = llama_get_logits(ctx);
            if (logits) {
                generated_logits.emplace_back(id, logits[id]);
            }
        }

        // Write the final response.
        {
            std::stringstream ss;
            ss << "Response: " << generated_text << "\n";
            my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
        }

        // Output all logits in one single line
        {
            std::stringstream ss;
            ss << "Logits: ";
            for (auto & [tok, logit] : generated_logits) {
                ss << tok << ":" << std::fixed << std::setprecision(6)
                   << logit << " ";
            }
            ss << "\n\n";
            my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
        }

        std::cout << "\n[End of line generation]\n";
    }  // end while reading lines

    infile.close();
    common_sampler_free(smpl);


    // Close the combined log
    g_combined_log.close();

    // ------------------------------------------------------------------
    // Clean up llama-related resources
    llama_backend_free();


    return 0;
}

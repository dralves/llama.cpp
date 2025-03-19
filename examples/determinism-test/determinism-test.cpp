#include <chrono>   // optionally for build timestamp (if you want to embed date/time)
#include <fstream>  // for std::ifstream
#include <iomanip>  // for std::setprecision, etc.
#include <iostream>
#include <limits>   // for std::numeric_limits<int>::max()
#include <mutex>    // if you want thread-safe logging
#include <sstream>  // for capturing tokens/sec logs if needed
#include <stdexcept>
#include <string>
#include <vector>

#include "arg.h"     // from llama.cpp: for command-line argument parsing
#include "common.h"  // from llama.cpp: for common_params, common_init_from_params, etc.
#include "llama.h"   // from llama.cpp: for model and context APIs

#ifdef __cplusplus
extern "C" {
#endif

#include "sha256/sha256.h"

#ifdef __cplusplus
}
#endif

// EXAMPLE: llama_log_callback signature:
//   void my_custom_logger(void * user_data, llama_log_level level, const char * message);

// Add forward declaration of our custom logger:
static std::ofstream g_combined_log;  // file stream for combined logging

static void my_custom_logger(ggml_log_level level, const char * message, void * /*user_data*/) {
    // Print to console
    std::cout << message;

    // Also to file
    if (g_combined_log.is_open()) {
        g_combined_log << message;
    }
}

// NEW: Function to check if a file exists
static bool file_exists(const std::string & filename) {
    std::ifstream ifs(filename.c_str());
    return ifs.is_open();
}

// NEW: If the requested filename exists, we do NOT rename the old file
// but we create a new name by appending .1, .2, etc. until we find a non-existent name.
// Returns the final file name we should open for writing.
static std::string handle_existing_file(const std::string & filename) {
    // If user's base filename does NOT exist, we just return it.
    if (!file_exists(filename)) {
        return filename;
    }

    // Otherwise, we pick a new name, e.g. filename.1
    int counter = 1;
    while (true) {
        std::stringstream ss;
        ss << filename << "." << counter;
        std::string candidate = ss.str();

        // If candidate doesn't exist, we use it as our new file name
        if (!file_exists(candidate)) {
            std::cerr << "File \"" << filename << "\" already exists.\n"
                      << "Using new output file: " << candidate << "\n";
            return candidate;
        }
        counter++;
    }
}

// NEW: A helper to convert a 32-byte digest to hex
#ifndef SHA256_DIGEST_SIZE
#    define SHA256_DIGEST_SIZE 32
#endif

static std::string to_hex(const unsigned char * digest, size_t len = SHA256_DIGEST_SIZE) {
    static const char * hex_digits = "0123456789abcdef";
    std::string         out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = digest[i];
        out.push_back(hex_digits[(c & 0xF0) >> 4]);
        out.push_back(hex_digits[(c & 0x0F)]);
    }
    return out;
}

int main(int argc, char ** argv) {
    // 1) We'll parse "-o" and now also parse "--repeat", removing them before calling common_params_parse.
    std::string outFileName = "determinism_results.txt";
    int         repeatCount = 1; // default is 1 iteration
    {
        int writeIndex = 1; // track how we shift argv
        for (int readIndex = 1; readIndex < argc; readIndex++) {
            std::string argStr = argv[readIndex];
            if (argStr == "-o") {
                // Check for next arg
                if (readIndex + 1 < argc) {
                    outFileName = argv[++readIndex];
                } else {
                    std::cerr << "Error: -o requires a filename\n";
                    return 1;
                }
                // skip storing "-o" or the filename in argv
            } else if (argStr == "--repeat") {
                // Check for next arg
                if (readIndex + 1 < argc) {
                    const char* rStr = argv[++readIndex];
                    try {
                        repeatCount = std::stoi(rStr);
                    } catch (...) {
                        std::cerr << "Error: --repeat must be followed by an integer\n";
                        return 1;
                    }
                } else {
                    std::cerr << "Error: --repeat requires an integer\n";
                    return 1;
                }
                // skip storing "--repeat" or the integer in argv
            } else {
                // keep the original arg in argv
                argv[writeIndex++] = argv[readIndex];
            }
        }
        argc = (argc > 1) ? writeIndex : 1;
    }

    // ------------------------------------------------------------------
    // THEN call llama.cpp's argument parser, ignoring any removed -o
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        std::cerr << "Error: Failed to parse arguments.\n";
        return 1;
    }

    // Some global initialization, as done in server.cpp
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // If the file exists, we do NOT rename it. Instead we pick a new file name
    // (the next available .1, .2, etc.).
    outFileName = handle_existing_file(outFileName);

    // 2) Now open "outFileName" as g_combined_log
    g_combined_log.open(outFileName);
    if (!g_combined_log.is_open()) {
        std::cerr << "Error: cannot open " << outFileName << " for logging.\n";
        return 1;
    }

    std::cerr << "Writing logs to: " << outFileName << "\n";

    llama_log_set(my_custom_logger, /*user_data=*/nullptr);

    // This might appear in your logs or output file

    common_init_result init_result = common_init_from_params(params);
    if (!init_result.model || !init_result.context) {
        std::cerr << "Error: Unable to load the model.\n";
        return 1;
    }

    llama_model *       model = init_result.model.get();
    llama_context *     ctx   = init_result.context.get();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // We'll interpret --prompt as the path to a file that has multiple lines to read.
    if (params.prompt_file.empty()) {
        std::cerr << "Error: Must pass --file=<filename> for multi-line input.\n";
        return 1;
    }

    // We'll run the entire logic of reading prompts "repeatCount" times
    for (int rep = 0; rep < repeatCount; rep++) {
        std::cerr << "== Iteration " << (rep + 1) << " of " << repeatCount << " ==\n";

        std::ifstream infile(params.prompt_file);
        if (!infile.is_open()) {
            std::cerr << "Error: cannot open " << params.prompt_file << "\n";
            return 1;
        }

        // REPLACE custom_sampler usage with the same approach from simple.cpp:
        // e.g. a sampler chain with a default (greedy) sampler. You can adapt for temperature, etc.
        auto sparams         = llama_sampler_chain_default_params();
        sparams.no_perf      = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        if (smpl == nullptr) {
            std::cerr << "Error: could not create sampler chain.\n";
            return 1;
        }
        // For a simple greedy approach:
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // REMOVE references to outFile. Instead, log with my_custom_logger.
        // For example, we print the param summary:
        // NEW: We'll store all prompt hashes, response hashes, and logits hashes
        std::vector<std::string> prompt_hashes;
        std::vector<std::string> response_hashes;
        std::vector<std::string> logits_hashes;

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
        int                      n_ctx = llama_n_ctx(ctx);

        // Decide how many new tokens are allowed to be generated per line:
        int max_new_tokens = (params.n_predict < 0) ? std::numeric_limits<int>::max() : params.n_predict;

        // Start measuring how long we run, and how many tokens we generate
        // from here on out:
        static int64_t t_main_start = ggml_time_us();
        int            n_decode     = 0;  // track how many tokens we generate from all lines

        std::string line;
        // Read the file line by line
        while (std::getline(infile, line)) {
            if (line.empty()) {
                continue;
            }

            // Instead of outFile, log prompt using my_custom_logger:
            {
                std::stringstream ss;
                ss << "Prompt: " << line << "\n\n";
                my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
            }

            const int n_src = -llama_tokenize(vocab, line.c_str(), line.size(), NULL, 0, true, true);
            if (n_src < 0) {
                std::cerr << "Error: failed to tokenize line.\n";
                continue;
            }
            std::vector<llama_token> line_tokens(n_src);
            if (llama_tokenize(vocab, line.c_str(), line.size(), line_tokens.data(), line_tokens.size(), true, true) < 0) {
                std::cerr << "Error: llama_tokenize returned negative.\n";
                continue;
            }

            // 2) Evaluate the prompt tokens in a single batch
            llama_batch batch_line = llama_batch_get_one(line_tokens.data(), line_tokens.size());
            if (llama_decode(ctx, batch_line) != 0) {
                std::cerr << "Error: decode of line prompt failed.\n";
                break;
            }

            std::string                                generated_text;
            std::vector<std::pair<llama_token, float>> generated_logits;

            for (int i = 0; i < max_new_tokens; i++) {
                // sample next token
                llama_token id = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, id)) {
                    std::cout << "\n[Terminated: EOS token.]\n";
                    break;
                }

                char buf[128];
                int  n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    std::cerr << "Error: convert token to piece.\n";
                    break;
                }
                std::string text(buf, n);
                generated_text += text;
                std::cout << text << std::flush;
                n_decode++;

                // evaluate newly sampled token
                llama_batch batch_next = llama_batch_get_one(&id, 1);
                if (llama_decode(ctx, batch_next)) {
                    std::cerr << "Error: decode failed while generating.\n";
                    break;
                }

                // store logit
                const float * logits = llama_get_logits(ctx);
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
                    ss << tok << ":" << std::fixed << std::setprecision(6) << logit << " ";
                }
                ss << "\n\n";
                my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);

                {
                    unsigned char digest[SHA256_DIGEST_SIZE];
                    sha256_hash(digest, reinterpret_cast<const unsigned char *>(line.data()), line.size());
                    std::string       prompt_hash = to_hex(digest);
                    std::stringstream s_hash;
                    s_hash << "Prompt Hash: " << prompt_hash << "\n";
                    my_custom_logger(GGML_LOG_LEVEL_INFO, s_hash.str().c_str(), nullptr);

                    // store for final hash-of-hashes
                    prompt_hashes.push_back(prompt_hash);
                }

                // NEW: hash the response
                {
                    unsigned char digest[SHA256_DIGEST_SIZE];
                    sha256_hash(digest, reinterpret_cast<const unsigned char *>(generated_text.data()),
                                generated_text.size());

                    std::string       response_hash = to_hex(digest);
                    std::stringstream s_hash;
                    s_hash << "Response Hash: " << response_hash << "\n";
                    my_custom_logger(GGML_LOG_LEVEL_INFO, s_hash.str().c_str(), nullptr);

                    response_hashes.push_back(response_hash);
                }

                // Now we also want to hash the entire "Logits: ..." line
                std::string logits_str = ss.str();  // copy the final string

                {
                    unsigned char digest[SHA256_DIGEST_SIZE];
                    sha256_hash(digest, reinterpret_cast<const unsigned char *>(logits_str.data()), logits_str.size());

                    std::string logits_hex = to_hex(digest);
                    std::stringstream hss;
                    hss << "Logits Hash: " << logits_hex << "\n";
                    my_custom_logger(GGML_LOG_LEVEL_INFO, hss.str().c_str(), nullptr);

                    logits_hashes.push_back(logits_hex);
                }
            }

        }  // end while reading lines

        infile.close();

        // The same technique as in simple.cpp
        // End-of-run timing:
        const auto t_main_end = ggml_time_us();
        double     elapsed_s  = double(t_main_end - t_main_start) / 1000000.0;
        double     tps        = (elapsed_s > 0.0) ? double(n_decode) / elapsed_s : 0.0;

        fprintf(stderr, "\n%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode, elapsed_s, tps);

        // Compute the "hash-of-hashes" for prompts, responses, and logits.
        // That is, for each category we combine all hash strings into one big string, then run SHA-256 on it.
        {
            // 1) prompt final hash-of-hashes
            std::string all_prompt_hash;
            all_prompt_hash.reserve(prompt_hashes.size() * (SHA256_DIGEST_SIZE * 2));
            for (auto & h : prompt_hashes) {
                all_prompt_hash += h; // just append all hex hash strings
            }
            {
                unsigned char digest[SHA256_DIGEST_SIZE];
                sha256_hash(digest,
                            reinterpret_cast<const unsigned char *>(all_prompt_hash.data()),
                            all_prompt_hash.size());

                std::stringstream ss;
                ss << "Final Prompt Hash-of-Hashes: " << to_hex(digest) << "\n";
                my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
            }

            // 2) response final hash-of-hashes
            std::string all_response_hash;
            all_response_hash.reserve(response_hashes.size() * (SHA256_DIGEST_SIZE * 2));
            for (auto & h : response_hashes) {
                all_response_hash += h;
            }
            {
                unsigned char digest[SHA256_DIGEST_SIZE];
                sha256_hash(digest,
                            reinterpret_cast<const unsigned char *>(all_response_hash.data()),
                            all_response_hash.size());

                std::stringstream ss;
                ss << "Final Response Hash-of-Hashes: " << to_hex(digest) << "\n";
                my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
            }

            // 3) logits final hash-of-hashes
            std::string all_logits_hash;
            all_logits_hash.reserve(logits_hashes.size() * (SHA256_DIGEST_SIZE * 2));
            for (auto & h : logits_hashes) {
                all_logits_hash += h;
            }
            {
                unsigned char digest[SHA256_DIGEST_SIZE];
                sha256_hash(digest,
                            reinterpret_cast<const unsigned char *>(all_logits_hash.data()),
                            all_logits_hash.size());

                std::stringstream ss;
                ss << "Final Logits Hash-of-Hashes: " << to_hex(digest) << "\n";
                my_custom_logger(GGML_LOG_LEVEL_INFO, ss.str().c_str(), nullptr);
            }
        }

        // Print performance stats.
        llama_perf_sampler_print(smpl);  // from #include "llama.h"
        llama_perf_context_print(ctx);

        // Close the combined log
        g_combined_log.close();

        // ------------------------------------------------------------------
        // Clean up llama-related resources
        llama_sampler_free(smpl);
        llama_backend_free();
    }

    return 0;
}

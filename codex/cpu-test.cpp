#include "server-context.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>

static void require(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

int main(int argc, char ** argv) {
    require(argc == 4, "usage: round-trace-cpu MODEL OUTPUT META(0|1)");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const auto begin = std::chrono::steady_clock::now();
    llama_backend_init();
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = 0;
    params.n_ctx = 512;
    params.n_batch = 128;
    params.n_ubatch = 32;
    params.n_parallel = 1;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 8;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = 3;
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.cpuparams.n_threads = params.speculative.draft.cpuparams_batch.n_threads = 8;
    const bool meta = std::atoi(argv[3]) != 0;
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu};
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
        params.speculative.draft.devices = {cpu, cpu};
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
        params.speculative.draft.cpuparams.n_threads = params.speculative.draft.cpuparams_batch.n_threads = 4;
    }
    std::ofstream out(argv[2], std::ios::binary);
    require(bool(out), "output open failed");
    {
        server_context server;
        require(server.load_model(params), "server model load failed");
        auto reader = server.get_response_reader();
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = reader.get_new_id();
        task.cli = true;
        task.cli_prompt = "The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:";
        task.params.n_predict = 48;
        task.params.return_tokens = true;
        task.params.cache_prompt = false;
        task.params.sampling.seed = 1729;
        task.params.sampling.temp = 1;
        task.params.sampling.top_k = 20;
        task.params.sampling.top_p = 0.95f;
        task.params.sampling.min_p = 0.05f;
        task.params.sampling.backend_sampling = true;
        std::thread inference([&] { server.start_loop(); });
        reader.post_task(std::move(task));
        auto results = reader.wait_for_all([] { return false; });
        server.terminate();
        inference.join();
        require(!results.error && results.results.size() == 1, "completion failed");
        auto * final = dynamic_cast<server_task_result_cmpl_final *>(results.results[0].get());
        require(final && !final->tokens.empty(), "missing completion token IDs");
        const uint64_t count = final->tokens.size();
        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        out.write(reinterpret_cast<const char *>(final->tokens.data()), count * sizeof(llama_token));
        auto * ctx = server.get_llama_context();
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * logits = llama_get_logits_ith(ctx, 0);
        require(logits, "missing server logits");
        out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
        std::printf("M: server tokens=%llu, final target logit bytes=%zu\n", (unsigned long long) count, nv * sizeof(float));
    }
    // A teacher-forced MTP fixture covers raw draft logits, independent of completion acceptance.
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    auto * model = llama_model_load_from_file(argv[1], mp);
    require(model, "MTP model load failed");
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 8;
    auto * ctx = llama_init_from_model(model, cp);
    require(ctx, "MTP context failed");
    const int ne = llama_model_n_embd(model);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> hidden(ne, 0.125f);
    auto * batch = llama_batch_ext_init(ctx);
    for (int step = 0; step < 6; ++step) {
        llama_batch_ext_clear(batch);
        const int row = llama_batch_ext_add_token(batch, 0, 100 + step);
        const llama_pos pos = step;
        require(llama_batch_ext_set_pos(batch, row, &pos), "MTP position failed");
        require(llama_batch_ext_set_embd_token(batch, row, {hidden.data(), 1, size_t(ne)}), "MTP hidden failed");
        require(llama_batch_ext_set_output_logits(batch, row, true), "MTP output failed");
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "MTP decode failed");
        const float * logits = llama_get_logits_ith(ctx, 0);
        require(logits, "missing MTP logits");
        out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    out.close();
    require(bool(out), "output write failed");
    std::printf("M: CPU wall seconds=%.6f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}

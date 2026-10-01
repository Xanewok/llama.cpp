#pragma once

#include "server-task.h"
#include "sampling.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <deque>

// DO NOT MERGE: local request diagnostics. Each line is flushed before generation continues.
struct server_capture {
    std::ofstream out;
    size_t round = 0;
    size_t emitted = 0;
    bool probs = false;
    std::deque<json> pending;
    json expected;
    double tolerance = 1e-6;
    bool mismatch = false;
    bool allow_drift = false;

    static const char * directory() {
        const char * value = std::getenv("LLAMA_CAPTURE_DIR");
        return value && *value ? value : nullptr;
    }

    static int limit() {
        const char * value = std::getenv("LLAMA_CAPTURE_LIMIT");
        if (!value) {
            return -1;
        }
        const int n = std::stoi(value);
        if (n <= 0) {
            throw std::runtime_error("LLAMA_CAPTURE_LIMIT must be positive");
        }
        return n;
    }

    void write(const json & event) {
        out << event.dump() << '\n';
        out.flush();
        if (!out) {
            throw std::runtime_error("capture write failed");
        }
    }

    void begin(const server_task & task, common_sampler * sampler, int n_ctx) {
        std::filesystem::create_directories(directory());
        const auto name = "request-" + std::to_string(ggml_time_us()) + "-" + std::to_string(task.id) + ".jsonl";
        out.open(std::filesystem::path(directory()) / name, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot open capture");
        }
        probs = std::getenv("LLAMA_CAPTURE_PROBS") && std::string(std::getenv("LLAMA_CAPTURE_PROBS")) == "1";
        expected = task.capture_expected;
        tolerance = task.capture_tolerance;
        allow_drift = task.capture_allow_drift;
        const auto & p = task.params.sampling;
        json ends = json::array();
        for (const auto & end : p.reasoning_budget_end) {
            ends.push_back(end);
        }
        write({{"event", "request"}, {"version", 1}, {"task_id", task.id},
               {"request", task.capture_request}, {"parsed", task.capture_parsed},
               {"effective", task.params.to_json()}, {"prompt_ids", task.tokens.get_text_tokens()},
               {"target_seed", common_sampler_get_seed(sampler)}, {"n_ctx", n_ctx},
               {"budget", {{"tokens", p.reasoning_budget_tokens}, {"start", p.reasoning_budget_start},
                           {"end", ends}, {"forced", p.reasoning_budget_forced}, {"control", p.reasoning_control}}},
               {"replay_force", task.capture_force}, {"branch_seed", task.capture_branch_seed},
               {"probs", probs}, {"limit", limit()}});
    }

    void observe(const std::vector<llama_token_data> & row, int budget_state) {
        if (!probs) {
            return;
        }
        auto p = row;
        for (const auto & e : p) {
            if (!std::isfinite(e.p) || e.p < 0) {
                pending.push_back({{"invalid_distribution", true}});
                mismatch |= !expected.empty();
                return;
            }
        }
        std::sort(p.begin(), p.end(), [](const auto & a, const auto & b) {
            return a.p != b.p ? a.p > b.p : a.id < b.id;
        });
        json top = json::array();
        double mass = 0;
        for (size_t i = 0; i < p.size() && i < 20; ++i) {
            if (p[i].p > 0) {
                top.push_back({{"id", p[i].id}, {"p", p[i].p}});
                mass += p[i].p;
            }
        }
        const json distribution = {{"top", top}, {"omitted_mass", std::max(0.0, 1.0 - mass)}, {"budget_state", budget_state}};
        const size_t index = emitted + pending.size();
        const std::string key = std::to_string(index);
        if (expected.is_object() && expected.contains(key)) {
            const auto & want = expected.at(key);
            bool equal = want.at("top").size() == top.size() && want.at("budget_state") == budget_state;
            for (size_t i = 0; equal && i < top.size(); ++i) {
                equal = want.at("top")[i].at("id") == top[i].at("id") &&
                        std::abs(want.at("top")[i].at("p").get<double>() - top[i].at("p").get<double>()) <= tolerance;
            }
            equal = equal && std::abs(want.at("omitted_mass").get<double>() - distribution.at("omitted_mass").get<double>()) <= tolerance;
            mismatch |= want.at("budget_state") != budget_state || (!equal && !allow_drift);
            write({{"event", "validation"}, {"index", index}, {"equal", equal}, {"target", distribution}, {"expected", want}, {"allow_numeric_drift", allow_drift}});
        }
        pending.push_back(distribution);
    }

    void start_round(const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & q, bool replay) {
        pending.clear();
        json proposals = json::array();
        for (const auto & row : q) {
            json p = json::array();
            for (const auto & e : row) {
                p.push_back({{"id", e.id}, {"p", e.p}});
            }
            proposals.push_back(p);
        }
        write({{"event", "round"}, {"round", round++}, {"at", emitted},
               {"draft", draft}, {"draft_q", proposals}, {"checkpoint_replay", replay}});
    }

    void token(llama_token id, const std::string & piece, bool forced) {
        static const char hex[] = "0123456789abcdef";
        std::string piece_hex;
        for (unsigned char c : piece) {
            piece_hex += hex[c >> 4];
            piece_hex += hex[c & 15];
        }
        json event = {{"event", "token"}, {"index", emitted++}, {"round", round - 1},
                      {"id", id}, {"piece_hex", piece_hex}, {"forced", forced}};
        if (probs) {
            if (pending.empty()) {
                event["distribution_missing"] = true;
            } else {
                event["target"] = pending.front();
                pending.pop_front();
            }
        }
        write(event);
    }
};

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <httplib.h>
#include <json.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

static std::atomic<bool> status{ true };
void signal(int signum) { status = false; }

class Nexus {

private:

    Ort::Env environment;
    Ort::Session model;
    sentencepiece::SentencePieceProcessor tokenizer;

    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probabilities(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); i++) {
            probabilities[i] = std::exp(logits[i] - max_logit);
            sum += probabilities[i];
        }
        for (float& probability : probabilities) { probability /= sum; }
        return probabilities;
    }

    int sampling(const std::vector<float>& probabilities) {
        static std::mt19937 generator{ std::random_device{}() };
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        return distribution(generator);
    }

    std::vector<int64_t> inference(
        std::vector<int64_t> context,
        int size_context,
        float temperature = 1.0,
        int top = 1
    ) {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (int pos = 0; pos < size_context / 2; pos++) {
            if (context.size() > size_context) {
                context.erase(context.begin(), context.end() - size_context);
            }

            std::vector<int64_t> input_shape = { 1, (int64_t)context.size() };
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, context.data(), context.size(), input_shape.data(), input_shape.size()
            );

            const char* input_names[] = { "input" };
            const char* output_names[] = { "logits" };

            auto outputs = model.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
            size_t vocab_size = type_info.GetShape().back();
            float* raw_logits = outputs[0].GetTensorMutableData<float>();

            std::vector<float> logits(vocab_size);
            for (size_t i = 0; i < vocab_size; i++) {
                logits[i] = raw_logits[(context.size() - 1) * vocab_size + i] / temperature;
            }

            std::vector<float> sorted = logits;
            std::nth_element(sorted.begin(), sorted.begin() + top, sorted.end(), std::greater<float>());
            float threshold = sorted[top - 1];
            for (float& l : logits) {
                if (l < threshold) { l = -INFINITY; }
            }

            auto probs = softmax(logits);
            int next = sampling(probs);
            context.push_back(next);

            if (next == 2) {
                auto it = std::find(context.rbegin(), context.rend(), 1);
                if (it != context.rend()) {
                    size_t idx = context.size() - 1 - (it - context.rbegin());
                    context = std::vector<int64_t>(context.begin() + idx, context.end());
                }
                break;
            }
        }

        return context;
    }

    std::vector<int64_t> preprocess(const std::string& prompt) {
        std::vector<int64_t> indices;
        indices.push_back(tokenizer.PieceToId("<|BEGIN|>"));
        std::vector<int> encoded;
        tokenizer.Encode(prompt, &encoded);
        for (int id : encoded) {
            indices.push_back(static_cast<int64_t>(id));
        }
        indices.push_back(tokenizer.PieceToId("<|END|>"));
        indices.push_back(tokenizer.PieceToId("<|SEPARATOR|>"));
        indices.push_back(tokenizer.PieceToId("<|BEGIN|>"));
        return indices;
    }

    std::string postprocess(const std::vector<int> encoded) {
        std::string output_text;
        if (!tokenizer.Decode(encoded, &output_text).ok()) {
            throw std::runtime_error("Postprocess failed");
        }
        return output_text;
    }

public:

    Nexus(
        const std::string& model_path,
        const std::string& tokenizer_path
    ) :
        environment(ORT_LOGGING_LEVEL_WARNING, "nexus"),
        model(environment, model_path.c_str(), Ort::SessionOptions{ nullptr })
    {
        tokenizer.Load(tokenizer_path);
    }

    std::string process(const std::string& prompt, int max_context = 1024, float temperature = 1.0f, int top_k = 5) {

        std::vector<int64_t> context = preprocess(prompt);
        std::vector<int64_t> result = inference(context, max_context, temperature, top_k);
        std::string answer = postprocess(std::vector<int>(result.begin(), result.end()));

        return answer;
    }
};

std::unique_ptr<Nexus> g_nexus;

nlohmann::json build_validation_error(const std::vector<std::tuple<std::vector<nlohmann::json>, std::string, std::string>>& errors) {
    nlohmann::json detail = nlohmann::json::array();
    for (const auto& [loc, msg, err_type] : errors) {
        detail.push_back({
            {"loc", loc},
            {"msg", msg},
            {"type", err_type}
            });
    }
    return { {"detail", detail} };
}

void setup_routes(httplib::Server& server) {

    try {
        g_nexus = std::make_unique<Nexus>("storage/nexus.onnx", "storage/tokenizer.model");
        std::cerr << "[model] loaded successfully\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[model] load failed: " << e.what() << "\n";
    }

    server.Get("/api/health", [](const httplib::Request& /*req*/, httplib::Response& res) {
        res.set_content("{}", "application/json");
        res.status = 200;
    });
    std::cerr << "[routes] GET /api/health\n";

    server.Post("/api/compose", [](const httplib::Request& req, httplib::Response& res) {
        if (!g_nexus) {
            res.status = 500;
            res.set_content(R"({"error":"Model not loaded"})", "application/json");
            return;
        }

        try {
            if (req.body.empty()) {
                auto error = build_validation_error({ {
                    {{"body"}}, "Request body is empty", "value_error.missing"
                } });
                res.status = 422;
                res.set_content(error.dump(), "application/json");
                return;
            }

            auto j = nlohmann::json::parse(req.body);

            std::vector<std::tuple<std::vector<nlohmann::json>, std::string, std::string>> errors;

            if (!j.contains("query") || !j["query"].is_string()) {
                errors.emplace_back(std::vector<nlohmann::json>{"query"}, "Field required", "value_error.missing");
            }

            if (!j.contains("metadata")) {
                errors.emplace_back(std::vector<nlohmann::json>{"metadata"}, "Field required", "value_error.missing");
            }
            else {
                auto& meta = j["metadata"];
                if (!meta.is_null() && !meta.is_object()) {
                    errors.emplace_back(std::vector<nlohmann::json>{"metadata"}, "Input should be an object or null", "type_error");
                }
            }

            if (!errors.empty()) {
                auto error_response = build_validation_error(errors);
                res.status = 422;
                res.set_content(error_response.dump(), "application/json");
                return;
            }

            std::string prompt = j["query"];
            std::string output = g_nexus->process(prompt, 1024, 1.0f, 1);

            nlohmann::json response = { {"response", output} };
            res.set_content(response.dump(), "application/json");
            res.status = 200;

        }
        catch (const nlohmann::json::parse_error&) {
            auto error = build_validation_error({ {
                {{"body"}}, "Invalid JSON", "value_error.jsondecode"
            } });
            res.status = 422;
            res.set_content(error.dump(), "application/json");
        }
        catch (const std::exception& e) {
            res.status = 500;
            res.set_content(R"({"error":")" + std::string(e.what()) + R"("})", "application/json");
        }
    });
    std::cerr << "[routes] POST /api/compose\n";

    server.Post("/api/recommend", [](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                auto error = build_validation_error({ {
                    {{"body"}}, "Request body is empty", "value_error.missing"
                } });
                res.status = 422;
                res.set_content(error.dump(), "application/json");
                return;
            }

            auto j = nlohmann::json::parse(req.body);

            std::vector<std::tuple<std::vector<nlohmann::json>, std::string, std::string>> errors;

            if (!j.contains("id") || !j["id"].is_string()) {
                errors.emplace_back(std::vector<nlohmann::json>{"id"}, "Field required", "value_error.missing");
            }

            if (!errors.empty()) {
                auto error_response = build_validation_error(errors);
                res.status = 422;
                res.set_content(error_response.dump(), "application/json");
                return;
            }

            std::string user_id = j["id"];


            // TODO: Implement recommendation logic
            nlohmann::json response = {
                {"recommendations", nlohmann::json::array()}
            };

            res.set_content(response.dump(), "application/json");
            res.status = 200;

        }
        catch (const nlohmann::json::parse_error&) {
            auto error = build_validation_error({ {
                {{"body"}}, "Invalid JSON", "value_error.jsondecode"
            } });
            res.status = 422;
            res.set_content(error.dump(), "application/json");
        }
        catch (const std::exception& e) {
            res.status = 500;
            res.set_content(R"({"error":")" + std::string(e.what()) + R"("})", "application/json");
        }
    });
    std::cerr << "[routes] POST /api/recommend\n";
}

int main() {
                                
    std::signal(SIGINT, signal);
    std::signal(SIGTERM, signal);

    httplib::Server server;

    server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::cerr << "[httplib] " << req.method << " " << req.path << " -> " << res.status << "\n";
        std::cerr << "  Content-Type: " << req.get_header_value("Content-Type") << "\n";
        std::cerr << "  Body length: " << req.body.size() << "\n";
        if (!req.body.empty()) {
            std::cerr << "  Body: '" << req.body << "'\n";
        }
        });
    server.set_payload_max_length(10 * 1024 * 1024);
                          
    server.set_exception_handler([](const httplib::Request& /*req*/, httplib::Response& res, std::exception_ptr eptr) {
        try {
            if (eptr) std::rethrow_exception(eptr);
        }
        catch (const std::exception& e) {
            res.status = 500;
            std::string msg = std::string("{\"error\":\"internal server error: ") + e.what() + "\"}";
            res.set_content(msg, "application/json");
            return;
        }
        catch (...) {
            res.status = 500;
            res.set_content("{\"error\":\"internal server error\"}", "application/json");
            return;
        }
        });

    setup_routes(server);

    const char* host = "0.0.0.0";
    int port = 7070;
    std::cerr << "Starting server on " << host << ":" << port << std::endl;

    std::thread server_thread([&]() {
        if (!server.listen(host, port)) {
            std::cerr << "Error: failed to start server (port may be in use)" << std::endl;
            status = false;
        }
        });

    while (status) { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); }

    std::cerr << "Stopping server..." << std::endl;
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    return 0;
}
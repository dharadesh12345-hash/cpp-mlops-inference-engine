#define _WIN32_WINNT 0x0A00 
#include "httplib.h" 
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#include <onnxruntime_cxx_api.h> 

int main() {
    std::cout << "Loading AI...\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FraudDetection");
    Ort::SessionOptions session_options;
    const wchar_t* model_path = L"fraud_model.onnx";
    Ort::Session session(env, model_path, session_options);

    std::cout << "AI Loaded successfully!\n";

    httplib::Server svr;

    svr.Get("/predict", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            // 1. DEFAULT VALUES (Normal transaction)
            float amount = 15.0f; 
            float age = 30.0f;

            // 2. READ FROM THE URL
            if (req.has_param("amount")) {
                amount = std::stof(req.get_param_value("amount"));
            }
            if (req.has_param("age")) {
                age = std::stof(req.get_param_value("age"));
            }

            std::cout << "\n---> Running AI for Amount: $" << amount << ", Age: " << age << "\n";
            
            // 3. PUT DYNAMIC NUMBERS INTO AI
            std::vector<float> input_data = {amount, age}; 
            std::vector<int64_t> input_shape = {1, 2}; 

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {"float_input"};
            const char* output_names[] = {"output_label"}; 

            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            int64_t* result = output_tensors.front().GetTensorMutableData<int64_t>();

            if (result[0] == 1) {
                res.set_content("WARNING: FRAUD DETECTED!", "text/plain");
            } else {
                res.set_content("Transaction is Normal.", "text/plain");
            }
            
            // 4. SAVE TO DATABASE
            std::ofstream log_file("transactions_log.csv", std::ios::app);
            log_file << amount << "," << age << "," << result[0] << "\n";
            log_file.close();

        } catch (const Ort::Exception& exception) {
            std::cout << "\n[ONNX ERROR]: " << exception.what() << "\n";
            res.set_content("Server Error", "text/plain");
        }
    });

    std::cout << "Server starting...\n";
    std::cout << "Try this URL: http://localhost:8082/predict?amount=9000&age=21\n";
    
    svr.listen("localhost", 8082); 
    return 0;
}







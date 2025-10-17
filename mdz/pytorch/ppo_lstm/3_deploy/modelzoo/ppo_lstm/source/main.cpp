#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>
#include<chrono>
#include<dlfcn.h>
#include<filesystem>
#include<iostream>
#include<fstream>
#include<numeric>


void ftmp2tensor(const float* tensor_ptr, int64_t fp32_size, const std::string& ftmp_path)
{
	std::ifstream ftmp_stream(ftmp_path, std::ios::binary);

	if (!ftmp_stream) {
		throw std::runtime_error("[Error in Ftmp2XrtTensor] FtmpFile is invalid");
	}

	ftmp_stream.read((char*)tensor_ptr, fp32_size * 4);
};


int main()
{
	try{
		/* ==== Run Network ====*/
		//ppo_lstm_actor
		const std::string model_dir = "../imodel";
		const std::string model_name = "ppo_lstm_carracing_run.onnx";
		const std::string model_path = model_dir + "/" + model_name;
		std::cout<< "load model from " << model_path <<std::endl;
		
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ppo_lstm");
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(4);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::Session session(env, model_path.c_str(), session_options);
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		
		//Ort::Value::CreateTensor<float>(encoder_input, input_lstm_states[0], input_lstm_states[1]);
		std::vector<Ort::Value> inputs;
		std::vector<Ort::Value> outputs;
		//input_path
		const std::string input_dir = "../io/inputs";
		const std::string input0_name = "encoder_input.ftmp";
		const std::string input1_name = "lstm_states0.ftmp";
		const std::string input2_name = "lstm_states1.ftmp";
		const std::string input0_path = input_dir + "/" + input0_name;
		const std::string input1_path = input_dir + "/" + input1_name;
		const std::string input2_path = input_dir + "/" + input2_name;
		std::cout<< "load input from " << input_dir <<std::endl;
		//encoder_input
		std::vector<int64_t> encoder_input_dims = {1, 2, 64, 64};
		size_t encoder_input_numel = std::accumulate(encoder_input_dims.begin() + 1, encoder_input_dims.end(), (int)!encoder_input_dims.empty(), std::multiplies<int>());
		std::vector<float> encoder_input_data(encoder_input_numel);
		ftmp2tensor(encoder_input_data.data(), encoder_input_numel, input0_path);
		Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_input_data.data(), encoder_input_numel, encoder_input_dims.data(), encoder_input_dims.size());
		inputs.push_back(std::move(encoder_input_tensor));
		//lstm_states0
		std::vector<int64_t> lstm_states0_dims = {1, 1, 128};
		size_t lstm_states0_numel = std::accumulate(lstm_states0_dims.begin() + 1, lstm_states0_dims.end(), (int)!lstm_states0_dims.empty(), std::multiplies<int>());
		std::vector<float> lstm_states0_data(lstm_states0_numel);
		ftmp2tensor(lstm_states0_data.data(), lstm_states0_numel, input1_path);
		Ort::Value lstm_states0_tensor = Ort::Value::CreateTensor<float>(memory_info, lstm_states0_data.data(), lstm_states0_numel, lstm_states0_dims.data(), lstm_states0_dims.size());
		inputs.push_back(std::move(lstm_states0_tensor));
		//lstm_states1
		std::vector<int64_t> lstm_states1_dims = {1, 1, 128};
		size_t lstm_states1_numel = std::accumulate(lstm_states1_dims.begin() + 1, lstm_states1_dims.end(), (int)!lstm_states1_dims.empty(), std::multiplies<int>());
		std::vector<float> lstm_states1_data(lstm_states1_numel);
		ftmp2tensor(lstm_states1_data.data(), lstm_states1_numel, input2_path);
		Ort::Value lstm_states1_tensor = Ort::Value::CreateTensor<float>(memory_info, lstm_states1_data.data(), lstm_states1_numel, lstm_states1_dims.data(), lstm_states1_dims.size());
		inputs.push_back(std::move(lstm_states1_tensor));

		std::vector<const char*> input_names = {"obs.1", "onnx::LSTM_1", "onnx::LSTM_2"};
		std::vector<const char*> output_names = {"mean_actions", "108", "109"};
		//inference time start
		for (int i = 0; i < 100; i++){
			auto start = std::chrono::high_resolution_clock::now();
			outputs = session.Run(
				Ort::RunOptions{nullptr}, 
				input_names.data(), 
				inputs.data(), 
				inputs.size(),
				output_names.data(),
				output_names.size()
			);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout<< "num_"<< i << " : inference time: " << duration.count() << " μs." <<std::endl;
		}
		// print output 
		float* output1 = outputs[0].GetTensorMutableData<float>();
		std::cout<< "mean_actions: " <<std::endl;
		for (size_t j = 0; j < 3; j++)
		{
			std::cout<< output1[j] <<" ";
		}
					
	}
	catch(std::exception& e){
		std::cout<< e.what()<<std::endl;
	}
	

	return 0;
}
#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>
#include<chrono>
#include<dlfcn.h>
#include<filesystem>
#include<iostream>
#include<fstream>
#include<numeric>

std::string getFtmpPath(int data_idx, const std::string& ftmp_name){
	std::string ftmp_path = "./io/" + std::to_string(data_idx) + "/";
	if(ftmp_name == "3" || ftmp_name == "4" || ftmp_name == "mask"){
		ftmp_path += "ori/";
	}
	ftmp_path += ftmp_name + ".ftmp";
	if (!std::filesystem::exists(ftmp_path)) {
		std::cout<< "ftmppath:" <<ftmp_path<<std::endl;
		throw std::runtime_error("[Error in Ftmp2MNNTensor] FtmpFile does not exist");
	}
	return ftmp_path;
}

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
		//drlpp_step20, drlpp_no_gru_step20
		const std::string model_dir = "./imodel";
		const std::string model_name = "drl_tsp_step1.onnx";
		const std::string model_path = model_dir + "/" + model_name;
		
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "drl_tsp");
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(4);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::Session session(env, model_path.c_str(), session_options);
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		
		for(int data_idx = 0; data_idx<20; ++data_idx){
			//Ort::Value::CreateTensor<float>(mem_info, data.get(), num_elements, shape.data(), shape.size());
			std::vector<Ort::Value> inputs;
			//static
			std::vector<int64_t> static_dims = {1, 2, 20};
			size_t static_numel = std::accumulate(static_dims.begin() + 1, static_dims.end(), (int)!static_dims.empty(), std::multiplies<int>());
			std::vector<float> static_data(static_numel);
			ftmp2tensor(static_data.data(), static_numel, getFtmpPath(data_idx, "1"));
			Ort::Value static_tensor = Ort::Value::CreateTensor<float>(memory_info, static_data.data(), static_numel, static_dims.data(), static_dims.size());
			inputs.push_back(std::move(static_tensor));
			//dynamic
			std::vector<int64_t> dynamic_dims = {1, 1, 20};
			size_t dynamic_numel = std::accumulate(dynamic_dims.begin() + 1, dynamic_dims.end(), (int)!dynamic_dims.empty(), std::multiplies<int>());
			std::vector<float> dynamic_data(dynamic_numel);
			ftmp2tensor(dynamic_data.data(), dynamic_numel, getFtmpPath(data_idx, "2"));
			Ort::Value dynamic_tensor = Ort::Value::CreateTensor<float>(memory_info, dynamic_data.data(), dynamic_numel, dynamic_dims.data(), dynamic_dims.size());
			inputs.push_back(std::move(dynamic_tensor));
			//decoder
			std::vector<int64_t> decoder_dims = {1, 2, 1};
			size_t decoder_numel = std::accumulate(decoder_dims.begin() + 1, decoder_dims.end(), (int)!decoder_dims.empty(), std::multiplies<int>());
			std::vector<float> decoder_data(decoder_numel);
			ftmp2tensor(decoder_data.data(), decoder_numel, getFtmpPath(data_idx, "3"));
			Ort::Value decoder_tensor = Ort::Value::CreateTensor<float>(memory_info, decoder_data.data(), decoder_numel, decoder_dims.data(), decoder_dims.size());
			inputs.push_back(std::move(decoder_tensor));
			//last hh
			std::vector<int64_t> last_hh_dims = {1, 1, 128};
			size_t last_hh_numel = std::accumulate(last_hh_dims.begin() + 1, last_hh_dims.end(), (int)!last_hh_dims.empty(), std::multiplies<int>());
			std::vector<float> last_hh_data(last_hh_numel);
			ftmp2tensor(last_hh_data.data(), last_hh_numel, getFtmpPath(data_idx, "4"));
			Ort::Value last_hh_tensor = Ort::Value::CreateTensor<float>(memory_info, last_hh_data.data(), last_hh_numel, last_hh_dims.data(), last_hh_dims.size());
			inputs.push_back(std::move(last_hh_tensor));
			//mask
			std::vector<int64_t> mask_dims = {1, 20};
			size_t mask_numel = std::accumulate(mask_dims.begin() + 1, mask_dims.end(), (int)!mask_dims.empty(), std::multiplies<int>());
			std::vector<float> mask_data(mask_numel);
			ftmp2tensor(mask_data.data(), mask_numel, getFtmpPath(data_idx, "mask"));
			Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(memory_info, mask_data.data(), mask_numel, mask_dims.data(), mask_dims.size());
			inputs.push_back(std::move(mask_tensor));

			std::vector<const char*> input_names = {"static", "dynamic", "decoder_input", "last_hh", "mask"};
			std::vector<const char*> output_names = {"ptr", "gru_out"};
			std::vector<int> tour_idx;
			//inference time start
			auto start = std::chrono::high_resolution_clock::now();
			for(int step = 0; step < 20; ++step){
				auto outputs = session.Run(
					Ort::RunOptions{nullptr}, 
					input_names.data(), 
					inputs.data(), 
					inputs.size(),
					output_names.data(),
					output_names.size()
				);
				//get ptr value
				auto ptr_data = outputs[0].GetTensorData<int>();
				auto ptr_value = (int)ptr_data[0];
				//update mask
				auto mask_data = inputs[4].GetTensorMutableData<float>();
				mask_data[ptr_value] = 0;
				//update decoder
				auto decoder_data = inputs[2].GetTensorMutableData<float>();
				auto static_data = inputs[0].GetTensorMutableData<float>();
				decoder_data[0] = static_data[ptr_value];
				decoder_data[1] = static_data[ptr_value + 20];
				//update last hh
				auto last_hh_data = inputs[3].GetTensorMutableData<float>();
				auto gru_out_data = outputs[1].GetTensorData<float>();
				for(int i = 0; i<128 ; ++i){
					last_hh_data[i] = gru_out_data[i];
				}
				//tour idx update
				tour_idx.push_back(ptr_value);
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			std::cout<< "Input Index: " << data_idx;
			std::cout<< " inference time: " << duration.count() << "milliseconds." <<std::endl;
			std::cout<< " tour idx: ";
			for(auto&& tour_idx_value: tour_idx){
				std::cout<< tour_idx_value <<" ";
			}
			std::cout<< std::endl << std::endl;					
		}
	}
	catch(std::exception& e){
		std::cout<< e.what()<<std::endl;
	}
	

	return 0;
}
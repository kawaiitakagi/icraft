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


float cosineSimilarity(const float* a, const float* b, int size) {
    float dot = 0.0;
    float magA = 0.0;
    float magB = 0.0;

    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }

    magA = std::sqrt(magA);
    magB = std::sqrt(magB);

    return dot / (magA * magB);
}


int main()
{
	try{
		/* ==== Run Network ====*/
		//ppo_lstm_actor
		const std::string model_dir = "../imodel";
		const std::string model_name = "ae.onnx";
		const std::string model_path = model_dir + "/" + model_name;
		std::cout<< "load model from " << model_path <<std::endl;
		
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ae");
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(4);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::Session session(env, model_path.c_str(), session_options);
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		
		//Ort::Value::CreateTensor<float>(encoder_input);
		std::vector<Ort::Value> inputs;
		std::vector<Ort::Value> outputs;
		//input_path
		const std::string input_dir = "../io/inputs/ae";
		const std::string input0_name = "encoder_input.ftmp";
		const std::string input0_path = input_dir + "/" + input0_name;
		std::cout<< "load input from " << input_dir <<std::endl;
		//encoder_input
		std::vector<int64_t> encoder_input_dims = {1, 3, 80, 160};
		size_t encoder_input_numel = std::accumulate(encoder_input_dims.begin() + 1, encoder_input_dims.end(), (int)!encoder_input_dims.empty(), std::multiplies<int>());
		std::vector<float> encoder_input_data(encoder_input_numel);
		ftmp2tensor(encoder_input_data.data(), encoder_input_numel, input0_path);
		Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_input_data.data(), encoder_input_numel, encoder_input_dims.data(), encoder_input_dims.size());
		inputs.push_back(std::move(encoder_input_tensor));

		std::vector<const char*> input_names = {"input_image"};
		std::vector<const char*> output_names = {"feature_output"};
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
		// print cosine Similarity 
		float* output_onnx = outputs[0].GetTensorMutableData<float>();
		auto output_base = new float[64];
		const std::string output_path = "../io/inputs/ae/ae_output.ftmp";
		std::ifstream output_file(output_path, std::ios::binary);
		output_file.read(reinterpret_cast<char*>(output_base), 129 * sizeof(float));
		output_file.close();
		float cosine_sim = cosineSimilarity(output_base, output_onnx, 64);
        std::cout << " cosine_similarity: " << cosine_sim << std::endl;
		// std::cout<< "output1: " <<std::endl;
		// for (size_t j = 0; j < 64; j++)
		// {
		// 	std::cout<< output1[j] <<" ";
		// }
		// std::cout<<std::endl;			
	}
	catch(std::exception& e){
		std::cout<< e.what()<<std::endl;
	}
	

	return 0;
}
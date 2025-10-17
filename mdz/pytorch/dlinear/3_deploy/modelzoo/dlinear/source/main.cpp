#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>
#include<chrono>
#include<dlfcn.h>
#include<filesystem>
#include<iostream>
#include<fstream>
#include<numeric>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
   //cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++

void ftmp2tensor(const float* tensor_ptr, int64_t fp32_size, const std::string& ftmp_path)
{
	std::ifstream ftmp_stream(ftmp_path, std::ios::binary);

	if (!ftmp_stream) {
		throw std::runtime_error("[Error in Ftmp2XrtTensor] FtmpFile is invalid");
	}

	ftmp_stream.read((char*)tensor_ptr, fp32_size * 4);
};




int main(int argc, char* argv[])
{
	try{
		/* ==== Run Network ====*/
		//ppo_lstm_actor
		std::string task_name = argv[1];
		std::cout<< task_name <<std::endl;
		if(task_name == "long_term_forecasting"){
			const std::string model_dir = "../imodel/ltf";
			const std::string model_name = "dlinear_ltf.onnx";
			const std::string model_path = model_dir + "/" + model_name;
			std::cout<< "load model from " << model_path <<std::endl;
			
			Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dlinear");
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
			const std::string input_dir = "../io/ltf/ftmp";
			const std::string label_dir = "../io/ltf/label";
			const std::string input_name = "0.ftmp";
			const std::string input_path = input_dir + "/" + input_name;
			const std::string label_path = label_dir + "/" + input_name;
			std::cout<< "load input from " << input_dir <<std::endl;
			//encoder_input
			std::vector<int64_t> encoder_input_dims = {1, 96, 7};
			// std::vector<int64_t> labelt_dims = {1, 144, 7};
			size_t encoder_input_numel = std::accumulate(encoder_input_dims.begin() + 1, encoder_input_dims.end(), (int)!encoder_input_dims.empty(), std::multiplies<int>());
			std::vector<float> encoder_input_data(encoder_input_numel);
			ftmp2tensor(encoder_input_data.data(), encoder_input_numel, input_path);
			// std::cout<< encoder_input_data.size() <<std::endl;

			//label load
			// std::vector<int64_t> encoder_input_dims = {1, 96, 7};
			std::vector<int64_t> labelt_dims = {1, 144, 7};
			size_t labelt_numel = std::accumulate(labelt_dims.begin() + 1, labelt_dims.end(), (int)!labelt_dims.empty(), std::multiplies<int>());
			std::vector<float> label_data(labelt_numel);
			ftmp2tensor(label_data.data(), labelt_numel, label_path);
			std::cout<< label_data.size() <<std::endl;

			Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_input_data.data(), encoder_input_numel, encoder_input_dims.data(), encoder_input_dims.size());
			inputs.push_back(std::move(encoder_input_tensor));
			
			std::vector<const char*> input_names;
			std::vector<const char*> output_names;



			Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
			input_names.push_back(input_name_Ptr.get());
			Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(0, allocator);
			output_names.push_back(output_name_Ptr.get());

			//inference time start
			for (int i = 0; i < 10; i++){
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

			int dec_in=7; 
			int dec_out=7; 
			int pre_len = 96;

			std::vector<float> input_seq;
			std::vector<float> output_seq;
			std::vector<float> label_seq;
			//show
			float* output1 = outputs[0].GetTensorMutableData<float>();

						// auto res_ptr = (float*)res_tensor.data().cptr();
				
			for (int i = dec_out - 1; i < pre_len * dec_out; i += dec_out) {
				output_seq.push_back(*(output1 + i));
			}

		


			
			for (auto i = encoder_input_data.begin()+dec_out - 1; i < encoder_input_data.end(); i += dec_out) {
				input_seq.push_back(*i);

				}

			for (auto i = label_data.begin()+dec_out - 1; i < label_data.end(); i += dec_out) {
				label_seq.push_back(*i);

			}

			//merge input and label for show
			input_seq.insert(input_seq.end(), label_seq.end()- pre_len, label_seq.end());


			int show_w = (((int)input_seq.size() * 4) / 100 + 1) * 100;
			int show_h = show_w * 0.75;
			cv::Mat img(show_h, show_w, CV_8UC3, cv::Scalar(255, 255, 255));
			std::vector<cv::Point> label_points;
			std::vector<cv::Point> output_points;
			

			//normalize
			auto maxE = std::max_element(input_seq.begin(), input_seq.end());
			auto minE = std::min_element(input_seq.begin(), input_seq.end());
			float scale = show_h*0.8 / (*maxE - *minE);
			float zero_point = (*maxE + *minE) / 2;



			// std::cout<< "output1: " <<std::endl;

			for (int p = 0; p < input_seq.size(); p++) {
				label_points.push_back(cv::Point((p+2) * 4, -(input_seq[p] - zero_point) * scale + show_h/2));
			}
			for (int p = 0; p < output_seq.size(); p++) {
				output_points.push_back(cv::Point((p+2+pre_len) * 4, -(output_seq[p] - zero_point) * scale + show_h / 2));
			}




			for (size_t i = 0; i < label_points.size() - 1; ++i) {
				cv::line(img, label_points[i], label_points[i + 1], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
			}
			for (size_t i = 0; i < output_points.size() - 1; ++i) {
				cv::line(img, output_points[i], output_points[i + 1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			}
			std::string id = input_name.substr(0,input_name.size()-5);
			std::cout<< id <<std::endl;
			cv::imwrite("../io/"+id+".png", img);
		}
		if(task_name == "classification"){
			const std::string model_dir = "../imodel/cls";
			const std::string model_name = "dlinear_cls.onnx";
			const std::string model_path = model_dir + "/" + model_name;
			std::cout<< "load model from " << model_path <<std::endl;
			
			Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dlinear");
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
			const std::string input_dir = "../io/cls/ftmp";
			const std::string label_dir = "../io/cls/label";
			const std::string input_name = "0.ftmp";
			const std::string input_path = input_dir + "/" + input_name;
			const std::string label_path = label_dir + "/" + input_name;
			std::cout<< "load input from " << input_dir <<std::endl;
			//encoder_input
			std::vector<int64_t> encoder_input_dims = {1, 405, 61};
			size_t encoder_input_numel = std::accumulate(encoder_input_dims.begin() + 1, encoder_input_dims.end(), (int)!encoder_input_dims.empty(), std::multiplies<int>());
			std::vector<float> encoder_input_data(encoder_input_numel);
			ftmp2tensor(encoder_input_data.data(), encoder_input_numel, input_path);

			//label load
			std::vector<int64_t> labelt_dims = {1, 1};
			size_t labelt_numel = std::accumulate(labelt_dims.begin() + 1, labelt_dims.end(), (int)!labelt_dims.empty(), std::multiplies<int>());
			std::vector<float> label_data(labelt_numel);
			ftmp2tensor(label_data.data(), labelt_numel, label_path);


			Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_input_data.data(), encoder_input_numel, encoder_input_dims.data(), encoder_input_dims.size());
			inputs.push_back(std::move(encoder_input_tensor));
			
			std::vector<const char*> input_names;
			std::vector<const char*> output_names;

			Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
			input_names.push_back(input_name_Ptr.get());
			Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(0, allocator);
			output_names.push_back(output_name_Ptr.get());

			//inference time start
			for (int i = 0; i < 10; i++){
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

			int dec_in=61; 
			int dec_out=2; 
			int pre_len = 1;

			std::vector<float> input_seq;
			std::vector<float> output_seq;
			std::vector<float> label_seq;

			//show
			float* output1 = outputs[0].GetTensorMutableData<float>();
			
			for (int i = 0; i < pre_len * dec_out; i +=1) {

				output_seq.push_back(*(output1 + i));
			}
			auto max_it = std::max_element(output_seq.begin(), output_seq.end());
			int cls = std::distance(output_seq.begin(), max_it);
			std::cout<< "预测类别:"<<cls<< std::endl;
			std::cout<< "标签类别:"<<label_data[0]<< std::endl;

		


		



			// //merge input and label for show
			// input_seq.insert(input_seq.end(), label_seq.end()- pre_len, label_seq.end());





		}

	}

	catch(std::exception& e){
		std::cout<< e.what()<<std::endl;
	}
	

	return 0;
}
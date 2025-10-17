#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>
#include <regex>
#include "icraft-xir/ops/align_axis.h"
#include "icraft-xir/ops/prune_axis.h"
#include "icraft-xir/ops/cast.h"
#include "icraft-xir/core/network.h"
#include "icraft-xir/serialize/json.h"
#include "icraft-xir/core/data.h"
#include "icraft-xrt/dev/host_device.h"
#include "icraft-xrt/dev/buyi_device.h"
#include "icraft-backends/hostbackend/utils.h"
#include "icraft-backends/hostbackend/cuda/device.h"
#include "et_device.hpp"
ICRAFT_CONSOLE_USE_UTF8
namespace fs = std::filesystem;
using namespace icraft::xrt;
using namespace icraft::xir;
// 合并算子之后的计时 ，因为合并后id改了 所以要从getForwards里获取实际的算子属性
void calctime_detail(icraft::xrt::Session& session) {
	auto network_name = session->network_view.network()->name;
	checkDir("./logs/");
	std::string filePath = "./logs/" + network_name + "_time" + ".txt";
	std::ofstream ofs(filePath.c_str(), std::ios::out);

	float total_hard_time = 0;
	float total_time = 0;
	float total_memcpy_time = 0;
	float total_other_time = 0;
	float hardop_total_time = 0;
	float hardop_hard_time = 0;
	float hardop_memcpy_time = 0;
	
	bool imk_on = false;
	bool post_on = false;

	float out_cast_time = 0;
	float icore_in_time = 0;
	float icore_out_time = 0;
	float icore_time = 0;
	float cpu_time = 0;
	float customop_total_time = 0;
	float customop_hard_time = 0;
	std::string in_fpgaop = "cdma";
	std::string out_fpgaop = "cdma";
	std::string icore_fpgaop = "Null";
	std::string cpu_op = "Null";
	std::vector<std::tuple<std::string, float, float>> customops;
	std::map<std::string, float> customop_total_times;
	std::map<std::string, float> customop_hard_times;
	auto result = session.timeProfileResults();
	for (auto k_v : result) {
		//std::cout << k_v.first << std::endl;
		//std::cout << session->network_view.network().getOpById(k_v.first)->typeKey() << std::endl;
		//std::cout << session->network_view.network().getOpById(k_v.first)->name << std::endl;
		auto& [time1, time2, time3, time4] = k_v.second;

		for (auto& [op, _, _1, _2, _3] : session.getForwards()) {
			if (op->op_id == k_v.first) {
				auto op_typekey = op->typeKey();
				auto op_name = op->name;
				//std::cout << op->typeKey() << std::endl;
				//std::cout << op->name << std::endl;
				ofs << fmt::format("op_id: {}, op_type: {}, op_name: {}, total_time: {}, memcpy_time: {}, hard_time: {}, other_time: {}\n",
					k_v.first, op_typekey, op->name
					, time1, time2, time3, time4);
				total_time += time1;
				total_memcpy_time += time2;
				total_hard_time += time3;
				total_other_time += time4;
				if (op_typekey == "icraft::xir::HardOpNode") {
					hardop_total_time += time1;
					//hardop_total_time -= time2;
					hardop_memcpy_time += time2;
					hardop_hard_time += time3;
				}
				if (op_typekey == "icraft::xir::CastNode") {
					if (time2 > 0.001) {
						out_cast_time += time2;
					}
				}
				if (op_typekey.find("customop") != std::string::npos) {
					if (op_typekey.find("ImageMake") != std::string::npos) {
						imk_on = true;
						icore_in_time += time3;
						in_fpgaop = "ImageMake";
					}
					else if (op_typekey.find("Post") != std::string::npos) {
						post_on = true;
						icore_out_time += time1;
						if (out_fpgaop == "cdma") {
							out_fpgaop = op_typekey.substr(0, op_typekey.size() - 4).substr(10);
						}
						else if(out_fpgaop.find(std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10))) == std::string::npos) {
							out_fpgaop = out_fpgaop +";" + std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));
						}
					}
					else {
						icore_time += time1;
						if (icore_fpgaop == "Null") {
							icore_fpgaop = std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));

						}
						else if(icore_fpgaop.find(std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10))) == std::string::npos) {
							icore_fpgaop = icore_fpgaop + ";" + std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));

						}
					}
					 customop_total_time += time1;
					 customop_hard_time += time3;
					if (customop_total_times.find(std::string(op_typekey)) != customop_total_times.end()) {

						customop_total_times[std::string(op_typekey)] += time1;
						customop_hard_times[std::string(op_typekey)] += time3;

					}
					else {
						customop_total_times[std::string(op_typekey)] = time1;
						customop_hard_times[std::string(op_typekey)] = time3;
					}
				}

			}
		}
		// ofs << fmt::format("op_id: {}, op_type: {}, op_name: {}, total_time: {}, memcpy_time: {}, hard_time: {}, other_time: {}\n",
		//    k_v.first, session->network_view.network().getOpById(k_v.first)->typeKey(), session->network_view.network().getOpById(k_v.first)->name
		//    , time1, time2, time3, time4);

	}
	if (!post_on) {
		icore_out_time = out_cast_time;
		cpu_time = total_time - hardop_total_time - customop_total_time - icore_out_time;
	}
	else {
		cpu_time = total_time - hardop_total_time - customop_total_time;
	}
	if (!imk_on) {	
		hardop_total_time -= hardop_memcpy_time;
		icore_in_time = hardop_memcpy_time;
	}

	if (cpu_time < 0) cpu_time = 0;
	ofs << "************************************" << std::endl;
	ofs << fmt::format("Total_TotalTime: {}, Total_MemcpyTime: {}, Total_HardTime: {}, Total_OtherTime: {}\n",
		total_time, total_memcpy_time, total_hard_time, total_other_time);
	ofs << fmt::format("Hardop_Total_Time: {} ms, Hardop_Hard_Time : {} ms.\n",
		hardop_total_time, hardop_hard_time);
	// ofs << fmt::format("Customop_Total_Time: {} ms, Customop_Hard_Time : {} ms.",
	// 	customop_total_time, customop_hard_time);    

	std::cout << "\n" << fmt::format("Total_TotalTime: {} ms, Total_MemcpyTime : {} ms, Total_HardTime : {} ms, Total_OtherTime : {} ms .",
		total_time, total_memcpy_time, total_hard_time, total_other_time) << std::endl;
	std::cout << fmt::format("Hardop_TotalTime: {} ms, Hardop_HardTime : {} ms.",
		hardop_total_time, hardop_hard_time) << std::endl;
	// std::cout << fmt::format("Customop_Total_Time: {} ms, Customop_Hard_Time : {} ms.",
	// 	customop_total_time, customop_hard_time) << std::endl;
	icore_time += hardop_total_time;
	for (const auto& pair : customop_total_times) {

		ofs << fmt::format("Customop: {},TotalTime: {} ms, HardTime : {} ms.\n",
			pair.first.substr(0, pair.first.size() - 4).substr(10), pair.second, customop_hard_times[pair.first]);
		std::cout << fmt::format("Customop: {},TotalTime: {} ms, HardTime : {} ms.",
			pair.first.substr(0, pair.first.size() - 4).substr(10), pair.second, customop_hard_times[pair.first]) << std::endl;
	}
	ofs << "******************************************************\n";
	std::cout << "******************************************************\n";
	ofs << "统计分析结果如下(The analysis results are as follows):\n";
	std::cout << "统计分析结果如下(The analysis results are as follows):\n";
	ofs << "数据传入耗时(Data input time consumption):\n";
	std::cout << "数据传入耗时(Data input time consumption):\n";
	ofs << "Time(ms):" << icore_in_time << "     Device:" << in_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_in_time << "     Device:" << in_fpgaop << std::endl;
	ofs << "icore[npu]耗时(Icore [npu] time-consuming):\n";
	std::cout << "icore[npu]耗时(Icore [npu] time-consuming):\n";
	ofs << "Time(ms):" << icore_time << "     Device:" << icore_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_time << "     Device:" << icore_fpgaop << std::endl;
	ofs << "数据传出耗时(Data output time consumption):\n";
	std::cout << "数据传出耗时(Data output time consumption):\n";
	ofs << "Time(ms):" << icore_out_time << "     Device:" << out_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_out_time << "     Device:" << out_fpgaop << std::endl;
	ofs << "cpu算子耗时(CPU operator time consumption):\n";
	std::cout << "cpu算子耗时(CPU operator time consumption):\n";
	ofs << "Time(ms):" << cpu_time << "     Device:" << cpu_op << std::endl;
	std::cout << "Time(ms):" << cpu_time << "     Device:" << cpu_op << std::endl;
	std::cout << "******************************************************\n";
	ofs.close();

	std::cout << "For details about running time meassage of the network, check the " + network_name + "_time" + ".txt" + " in path: " + "./logs/" << std::endl;
};

void calctime(icraft::xrt::Session& session) {
	auto network_name = session->network_view.network()->name;
	float total_hard_time = 0;
	float total_time = 0;
	float total_memcpy_time = 0;
	float total_other_time = 0;
	auto result = session.timeProfileResults();
	for (auto k_v : result) {
		auto& [time1, time2, time3, time4] = k_v.second;
		total_time += time1;
		total_memcpy_time += time2;
		total_hard_time += time3;
		total_other_time += time4;
	}
	std::cout << "=======TimeProfileResults of " << network_name << "=========" << std::endl;
	std::cout << fmt::format("Total_Time: {} ms, Total_MemcpyTime: {} ms , Total_HardTime: {} ms , Total_OtherTime: {}ms",
		total_time, total_memcpy_time, total_hard_time, total_other_time) << std::endl;
}


std::string getJrPath(const bool& run_sim, std::string& folderPath, std::string targetFileName) {
#ifdef _WIN32
	if (!run_sim) {
		targetFileName = "BY.json";
	}
	else {
		if (STAGE.count(targetFileName) > 0)
			targetFileName = STAGE[targetFileName] + ".json";
		else
			throw std::runtime_error("imodel stage not right ,please check yaml:imodel:dir");

	}
#elif __linux__
	targetFileName = "BY.json";
#endif

	for (const auto& entry : fs::directory_iterator(folderPath)) {
		//std::cout << entry.path().filename().string() << std::endl;
		if (entry.is_regular_file() && entry.path().filename().string().find(targetFileName) != std::string::npos) {
			spdlog::info("Found model file at:{}", entry.path().string());

			return entry.path().string();
		}
	}
	throw std::runtime_error("imodel path not right ,please check yaml:imodel:dir");
}

Device openDevice(const bool& run_sim, const std::string& ip, bool mmu_Mode = true,bool cuda_Mode = false, const std::string& npu_addr = "0x40000000", const std::string& dma_addr = "0x80000000") {

#ifdef _WIN32
	if (run_sim) {
		if (cuda_Mode){
			return CudaDevice::Default();
		}
		return HostDevice::Default();
	}
	std::string URL_PATH = "socket://ql100aiu@" + ip + ":9981?npu=" + npu_addr + "&dma=" + dma_addr;
	Device device;
	device = Device::Open(URL_PATH);
	device.cast<BuyiDevice>().mmuModeSwitch(mmu_Mode);
	return device;
#elif __linux__
	std::string URL_PATH = "axi://ql100aiu?npu=" + npu_addr + "&dma=" + dma_addr;
	Device device;
	device = Device::Open(URL_PATH);
	device.cast<BuyiDevice>().mmuModeSwitch(mmu_Mode);
	return device;
#endif


}

Network loadNetwork(const std::string& JSON_PATH, const std::string& RAW_PATH) {
	auto network = Network::CreateFromJsonFile(JSON_PATH);
	network.lazyLoadParamsFromFile(RAW_PATH);
	return network;
}

Session initSession(const bool& run_sim, const NetworkView& network, Device& device, bool mmu,
	bool open_speedmode = false, bool open_compressFtmp = false) {
#ifdef __linux__
	auto session = Session::Create<BuyiBackend, HostBackend>(network, { device, HostDevice::Default() });
	if (mmu) return session;
	auto buyi_backend = session->backends[0].cast<BuyiBackend>();
	if (open_compressFtmp)
		buyi_backend.compressFtmp();
	if (open_speedmode)
		buyi_backend.speedMode();
	return session;
#endif
	if (run_sim) {
		auto session = Session::Create<HostBackend>(network, { device });


		return session;
	}
	else {
		auto session = Session::Create<BuyiBackend, HostBackend>(network, { device, HostDevice::Default() });
		if (mmu) return session;
		auto buyi_backend = session->backends[0].cast<BuyiBackend>();
		if (open_compressFtmp)
			buyi_backend.compressFtmp();
		if (open_speedmode)
			buyi_backend.speedMode();
		return session;
	}

}

Tensor CvMat2Tensor(cv::Mat& img, const Network& network) {
	// 获取输入的value 用于从 cvMat 构造 输入tensor
	auto input_value = network.inputs()[0];
	// 将cv Mat构造为输入网络的TENSOR
	auto out_dtype = input_value.tensorType().clone();
	auto out_stor_type = out_dtype->element_dtype.getStorageType();
	cv::Mat converted;
	if (out_stor_type.is<xir::FloatType>()) {
		auto float_stor_type = out_stor_type.cast<xir::FloatType>();
		if (float_stor_type.isFP32()) {
			img.convertTo(converted, CV_32F);
		}
		else if (float_stor_type.isFP16()) {
			img.convertTo(converted, CV_16F);
		}
		else {
			ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", float_stor_type->typeKey());
		}
	}
	else if (out_stor_type.is<xir::IntegerType>()) {
		auto int_stor_type = out_stor_type.cast<xir::IntegerType>();
		if (int_stor_type.isSInt8()) {
			img.convertTo(converted, CV_8S);
		}
		else if (int_stor_type.isUInt8()) {
			img.convertTo(converted, CV_8U);
		}
		else if (int_stor_type.isSInt16()) {
			img.convertTo(converted, CV_16S);
		}
		else if (int_stor_type.isUInt16()) {
			img.convertTo(converted, CV_16U);
		}
		else if (int_stor_type.isSInt32()) {
			img.convertTo(converted, CV_32S);
		}
		else {
			ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", int_stor_type->typeKey());
		}
	}
	else {
		ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", out_stor_type->typeKey());
	}
	int H = converted.rows;
	int W = converted.cols;
	int C = converted.channels();
	//define output tensor
	std::vector<int64_t> output_shape = { 1, H, W, C };
	auto tensor_layout = xir::Layout("NHWC");
	out_dtype.setShape(output_shape);
	auto img_tensor = xrt::Tensor(out_dtype).mallocOn(xrt::HostDevice::MemRegion());
	//data copy
	memcpy(img_tensor.data().cptr(), converted.data, H * W * C * out_dtype->element_dtype.bits() / 8);
	//std::cout << img_tensor.dtype()->shape << std::endl;
	return img_tensor;
}

template <typename T>
Tensor data2Tensor(const T* input_data, const xir::Value& input_value) {
	TensorType out_dtype;
	if (input_value.tensorType()->shape[0] == -1) {
		out_dtype = input_value.getUsesOp()[0]->outputs[0].tensorType().clone();
	}
	else {
		out_dtype = input_value.tensorType().clone();
	}
	auto size = out_dtype.numElements();

	auto out_stor_type = out_dtype->element_dtype.getStorageType();

	auto ele_dtype = out_dtype->element_dtype;

	if (ele_dtype.isUInt(8)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(uint8_t)); //malloc on host
		auto trans_data = (uint8_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (uint8_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isSInt(8)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(int8_t)); //malloc on host
		auto trans_data = (int8_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (int8_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isUInt(16)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(uint16_t)); //malloc on host
		auto trans_data = (uint16_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (uint16_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isSInt(16)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(int16_t)); //malloc on host
		auto trans_data = (int16_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (int16_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isUInt(32)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(uint32_t)); //malloc on host
		auto trans_data = (uint32_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (uint32_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isSInt(32)) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(int32_t)); //malloc on host
		auto trans_data = (int32_t*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (int32_t)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else if (ele_dtype.isFP32()) {
		auto param_chunk = icraft::xrt::HostDevice::MemRegion().malloc(size * sizeof(float)); //malloc on host
		auto trans_data = (float*)param_chunk->begin.cptr();
		std::transform((T*)input_data, (T*)input_data + size, trans_data, [](auto d) {return (float)d; });
		return icraft::xrt::Tensor(out_dtype, param_chunk);
	}
	else {
		ICRAFT_LOG(EXCEPT).append("[Error in HostBackend::GenTensorFromParams] Unsupported dtype {}, can't convert to torch tensor.", ele_dtype->typeKey());
	}



}


// 删除输出分支上的指定pattern（cast-Pruneaxis），并按照原来output算子的ifm顺序重新连接hardop <->output；
// idx_list用于指定分支删除cast&Pruneaxis算子，例如：指定第1条分支删除cast&Pruneaxis算子：idx_list={0}
void removeOutputCast(icraft::xir::Network& network, bool mmu, Array<int> idx_list = {}) {
	auto codegen_speedmode = Downcast<Bool>(network.getTag("speedmode").value())->value;
	auto codegen_compressFtmp = Downcast<Bool>(network.getTag("compressFtmp").value())->value;
	bool codegen_mmu = codegen_speedmode || codegen_compressFtmp;
	if(codegen_mmu || mmu)	ICRAFT_LOG(WARNING).append("Open MMU will lock the order of ftmp's physical address, and this may affect network connection!");

	auto cast_p = IsOp<Cast>();
	auto prune_axis_p = IsOp<PruneAxis>(cast_p[0]).setConstraint([](const Operation& op) {
		auto prune_axis = op.cast<PruneAxis>();
		PATTERN_REQUIRE(prune_axis.consumers().size() == 1);
		PATTERN_REQUIRE(prune_axis.consumers()[0]->isInstance<OutputNode>());
		return true;
		});

	network.rewrite(prune_axis_p, [&](Network& network, const MatchGroup& result) {

		auto cast = result.at(cast_p);
		auto prune_axis = result.at(prune_axis_p);
		auto output = prune_axis.consumers()[0];
		auto hardop = cast.producers()[0];

		// 匹配到的是第index个输出
		auto index = output.getInputIndex(prune_axis[0]);
		auto it = std::find(idx_list.begin(), idx_list.end(), *(index.begin()));

		//可指定分支，去除cast&Pruneaxis；若不输入指定分支，默认去除所有分支的cast&Pruneaxis
		if (it != idx_list.end() || idx_list.size() == 0) {
			// 重新连接hardop<->output
			output.setInput(*(index.begin()), hardop[0]);
			// 删除Cast&PruneAxis
			network.removeOpById(prune_axis->op_id);
			network.removeOpById(cast->op_id);

		}
		// 如果不是指定分支，不做任何操作
		else {
			network.rewriter().Continue();
		}

	});
}
// 删除输入分支上的指定pattern（Alignaxis-cast）, 并按照原来input算子的ofm顺序重新连接hardop<->input；
// idx_list用于指定分支删除Alignaxis&cast算子，例如：指定第1条分支删除Alignaxis&cast算子：idx_list={0}
void removeInputCast(icraft::xir::Network& network, bool mmu, Array<int> idx_list = {}) {
	auto codegen_speedmode = Downcast<Bool>(network.getTag("speedmode").value())->value;
	auto codegen_compressFtmp = Downcast<Bool>(network.getTag("compressFtmp").value())->value;
	bool codegen_mmu = codegen_speedmode || codegen_compressFtmp;
	if (codegen_mmu || mmu)	ICRAFT_LOG(WARNING).append("Open MMU will lock the order of ftmp's physical address, and this may affect network connection!");
	
	auto input_p = IsOp<Input>();
	auto align_axis_p = IsOp<AlignAxis>(input_p);
	auto cast_p = IsOp<Cast>(align_axis_p[0]);

	network.rewrite(cast_p, [&](Network& network, const MatchGroup& result) {
		auto input = result.at(input_p);
		auto align_axis = result.at(align_axis_p);
		auto cast = result.at(cast_p);

		// 提前记录下来cast要连接到地方
		auto cast_uses_info = network.getUsesInfoExceptMatch(cast[0], result);

		// 匹配到的是第index个输出
		auto index = align_axis->inputs[0].index();
		auto it = std::find(idx_list.begin(), idx_list.end(), index);

		//可指定分支，去除cast&Alignaxis；若不输入指定分支，默认去除所有分支的cast&Alignaxis
		if (it != idx_list.end() || idx_list.size() == 0) {
			// 拷贝一份cast的输入，重置一下v_id，防止重名
			auto new_value = cast[0].clone(-1).setId(-1);
			// 重新连接hardop<->input
			input.setOutput(index, new_value);
			// 删除AlignAxis&Cast
			network.removeOpById(align_axis->op_id);
			network.removeOpById(cast->op_id);

			// Input的第index个输入连接到原来cast要连接到地方
			network.connect(input[index], cast_uses_info);
				
		}
		// 如果不是指定分支，不做任何操作
		else {
			network.rewriter().Continue();
		}

	});
}
 std::vector<float> getOutputNormratio(icraft::xir::NetworkView network) {
 	auto network_outp = network.outputs();
 	std::vector<float> ret;
 	ret.reserve(network_outp.size());
 	for (auto&& value : network_outp) {
		try {
			auto b = value->dtype.getNormratio().value();
			ret.emplace_back(b[0]);
		}
		catch (const std::exception& e) {
			std::cout << "the output of network/networkview have no Normratio" << std::endl;;
		}
 	}
 	return ret;
 }


 std::vector<float> getInputNormratio(icraft::xir::NetworkView network) {
 	auto network_inp = network.inputs();
 	std::vector<float> ret;
 	ret.reserve(network_inp.size());
 	for (auto&& value : network_inp) {
		try {
			auto b = value->dtype.getNormratio().value();
			ret.emplace_back(b[0]);
		}
		catch (const std::exception& e) {
			std::cout << "the input of network/networkview have no Normratio" << std::endl;;
		}
 	}
 	return ret;
 }

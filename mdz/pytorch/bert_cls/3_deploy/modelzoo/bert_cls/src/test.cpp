#include <iostream>
#include <fstream>
#include <string>
#include <codecvt>
#include <locale>
#include <unordered_map>
#include <vector>


std::unordered_map<std::wstring, int64_t> buildVocabMap(const std::string& filename) {
    std::unordered_map<std::wstring, int64_t> vocabMap;
    std::wifstream file(filename);
    std::wstring line;
    int64_t id = 0;

    if (!file.is_open()) {
        std::wcerr << L"无法打开文件: " << filename.c_str() << std::endl;
        return vocabMap;
    }

    file.imbue(std::locale(file.getloc(), new std::codecvt_utf8<wchar_t>));

    while (std::getline(file, line)) {
        // 去除行末的换行符
        line.erase(line.find_last_not_of(L"\r\n") + 1);
        vocabMap[line] = id++;
    }

    file.close();
    return vocabMap;
}


int main() {

    // 获取输入id
    std::locale::global(std::locale("en_US.UTF-8"));
    std::wcout.imbue(std::locale("en_US.UTF-8"));

    std::string inputFilePath = "../io/input/test.txt";
    std::wifstream inputFile(inputFilePath);

    // 设置文件流为 UTF-8 编码
    inputFile.imbue(std::locale(inputFile.getloc(), new std::codecvt_utf8<wchar_t>));

    std::unordered_map<std::wstring, int64_t> vocabMap = buildVocabMap("../io/input/vocab.txt");

    if (!inputFile.is_open()) {
        std::wcerr << L"无法打开文件: " << inputFilePath.c_str() << std::endl;
        return 1;
    }

    std::wstring text;
    if (std::getline(inputFile, text)) {
        std::wcout << L"读取到的文本: " << text << std::endl;
        size_t pos = text.find_last_not_of(L"0123456789");
        std::wstring cleanedText = (pos != std::wstring::npos && pos < text.size() - 1) ? text.substr(0, pos + 1) : text;
        std::wcout << L"输入：" << cleanedText << std::endl;
        std::vector<int64_t> input_ids;
        input_ids.push_back(101);  // 首字符填充特殊字符：[UNK]->101
        // 遍历 cleanedText 中的每一个字符
        for (wchar_t ch : cleanedText) {
            // 将字符转换为字符串并输出
            std::wstring charAsString(1, ch);
            if (vocabMap.find(charAsString) != vocabMap.end()) {
                input_ids.push_back(vocabMap.at(charAsString));
                std::wcout << L"匹配正确： " << charAsString  << L" ,id:"<< input_ids.back() << std::endl;
            } else {
                std::wcerr << L"字符 '" << charAsString << L"' 未在词汇表中找到。" << std::endl;
                // input_ids.push_back(0);  // 不匹配，不添加
            }
        }
        // 调整 input_ids 到 32 个字符
        while (input_ids.size() < 32) {
            input_ids.push_back(0);  // 补 0
        }
        if (input_ids.size() > 32) {
            input_ids.resize(32);  // 删除多余的
        }

        // 输出调整后的 input_ids
        std::wcout << L"调整后的 input_ids: ";
        for (int64_t id : input_ids) {
            std::wcout << id << L" ";
        }
        std::wcout << std::endl;
    } else {
        std::wcout << L"无法读取文件内容" << std::endl;
    }

    inputFile.close();
    return 0;
}
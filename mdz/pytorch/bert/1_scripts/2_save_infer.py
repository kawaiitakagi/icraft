from PIL import Image
import numpy as np
import collections
import torch


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# 输入
embedding_out = np.fromfile("../2_compile/qtset/bert/embedding_out.ftmp", dtype=np.float32).reshape(1,128,768)
input_ids = np.fromfile("../2_compile/qtset/bert/input_ids.ftmp", dtype=np.float32).reshape(1,128)
attention_mask =  np.fromfile("../2_compile/qtset/bert/attention_mask.ftmp", dtype=np.float32).reshape(1,128)

# 特征提取阶段socket推理
model = torch.jit.load("../2_compile/fmodel/bert_mask_traced.pt")
outputs = model(torch.tensor(embedding_out), torch.tensor(input_ids), torch.tensor(attention_mask))


# 获取输入的的[MASK]位置下标
mask_token_index = 4
# 获取[MASK]位置对应的预测结果
masked_token_logits = outputs[0][0, mask_token_index, :].detach().numpy()
# 获取预测结果中概率最高的token索引
predicted_token_index = np.argmax(masked_token_logits)

# 输入文本
text = "I want to [MASK] a new car."
# 根据字典将预测id映射为文本
vocab_file = R'../weights/bert-base-cased/vocab.txt'
vocab = load_vocab(vocab_file)
ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in vocab.items()])
predicted_token = ids_to_tokens[predicted_token_index]
# 预测的单词替换mask
completed_text = text.replace('[MASK]', predicted_token)
# 打印补全后的文本
print("Tnput text:", text)
print("Completed text:", completed_text)
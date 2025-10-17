import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM


pretrain_model_bert = R'../weights/bert-base-cased'
# 获取tokenizer
tokenzier = BertTokenizer.from_pretrained(pretrain_model_bert)

# 加载bert mask 预训练模型
model = BertForMaskedLM.from_pretrained(pretrain_model_bert)

text = "I want to [MASK] a new car."
text_tokens = tokenzier(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

model_outputs = model(**text_tokens)

mask_token_index = torch.where(text_tokens["input_ids"] == tokenzier.mask_token_id)[1]
masked_token_logits = model_outputs[0][0, mask_token_index, :]

# 获取预测结果中概率最高的token索引
predicted_token_index = torch.argmax(masked_token_logits, dim=1).item()

# 根据预测的token索引获取预测的token
predicted_token = tokenzier.convert_ids_to_tokens([predicted_token_index])[0]

# 在原始文本中替换[MASK]为预测的token
completed_text = text.replace('[MASK]', predicted_token)

# 打印补全后的文本
print("Tnput text:", text)
print("Output text:", completed_text)

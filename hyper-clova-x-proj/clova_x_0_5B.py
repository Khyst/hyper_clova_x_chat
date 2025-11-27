import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 1. 로컬 경로 설정 ---
# Git Clone으로 다운로드한 모델 폴더의 절대 또는 상대 경로를 지정합니다.
local_model_path = "../models/HyperCLOVAX-SEED-Text-Instruct-0.5B"

# 2. 하드웨어 설정 (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# --- 3. 로컬 경로에서 토크나이저 및 모델 로드 ---
# from_pretrained() 함수에 로컬 경로를 직접 전달합니다.
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

print(f"HyperCLOVAX-SEED-Text-Instruct-0.5B 모델이 로컬 경로 ({local_model_path})에서 성공적으로 로드되었습니다.")

# 4. 프롬프트 작성 및 실행 (이후 과정은 동일)
instruction = "이 상품의 장점을 3가지로 요약해 줘."
user_input = "이 로봇청소기는 흡입력도 좋고, 물걸레 기능도 있어."
prompt = f"### System: {instruction}\n ### User: {user_input}\n ### Assistant: "

inputs = tokenizer(prompt, return_tensors="pt")

# input_ids와 attention_mask를 디바이스로 이동
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device) # 새로 추가된 부분

# 6. 텍스트 생성 (generate 함수에 attention_mask 전달)
with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask, # <-- 이 부분을 추가해야 합니다.
        max_length=256,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
response = generated_text[len(prompt):].strip()

print("-" * 50)
print(f"**생성된 응답:**\n{response}")
print("-" * 50)
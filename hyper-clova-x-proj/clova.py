import torch
import time
import os
import re # 정규 표현식 모듈
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# LangChain RAG 및 Core 모듈 임포트
from langchain_core.prompts import PromptTemplate 
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from operator import itemgetter

LOCAL_MODEL_PATH = "../models/HyperCLOVAX-SEED-Text-Instruct-1.5B" 
# --- 1. 설정 및 로컬 모델 로드 (HuggingFace LLM) ---

def load_model(local_model_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        print("LLM이 로컬 경로에서 성공적으로 로드되었습니다.")
        return model, tokenizer
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n사용 장치: {device}")

model, tokenizer = load_model(LOCAL_MODEL_PATH)

# --- 2. 기존 로직을 함수로 정의 ---
def generate_response(chat):
    """chat 메시지를 받아서 모델로 생성하고 결과를 반환하는 함수"""
    start_time = time.time()  # 시작 시간 측정
    
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(device)
    output_ids = model.generate(
        **inputs,
        max_length=256,
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
    )

    # 입력 토큰의 길이 (프롬프트의 길이)를 구합니다.
    input_len = inputs["input_ids"].shape[1]

    # 생성된 토큰 시퀀스에서 입력 부분을 제외한, 모델이 새로 생성한 답변 토큰만 추출합니다.
    generated_ids = output_ids[:, input_len:]

    # 모델의 답변 부분만 디코딩합니다.
    generated_text = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    end_time = time.time()  # 종료 시간 측정
    elapsed_time = end_time - start_time  # 소요 시간 계산
    
    return {
        "answer": generated_text,
        "elapsed_time": elapsed_time
    }

# --- 3. LangChain Runnable Chain 생성 ---
# RunnableLambda를 사용하여 기존 함수를 체인으로 만듭니다
llm = RunnableLambda(generate_response)

# --- 4. 테스트용 채팅 데이터 ---
chats = [
    [{"role": "system", "content": "당신은 친절하게 사용자의 질문에 대해 답변하는 꿈돌이 로봇입니다, 묻는 바에 대해 최대한 간결하게 세 문장 이내로 알려주세요."},
    {"role": "user", "content": "파이썬에서 time 모듈에 대해 알려줘"}],
]

# --- 5. Chain invoke 실행 ---
for chat in chats:
    result = llm.invoke(chat)
    print(result["answer"])
    print(f"처리 시간: {result['elapsed_time']:.4f}초\n")
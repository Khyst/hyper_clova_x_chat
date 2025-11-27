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
    [{"role": "system", "content": "당신은 친절하게 사용자의 질문에 대해 답변하는 꿈돌이 로봇입니다, 묻는 바에 대해 최대한 간결하게 세 문장 이내로 알려주세요. 모르는 내용이나 확실하지 않은 정보는 절대 추측하거나 지어내지 말고, '잘 모르겠습니다' 또는 '확실하지 않습니다'라고 정직하게 답변해주세요."},
    {"role": "user", "content": "\"Hyper Clova X 모델에 대해 설명해줘.\" 라는 문장에 대해서 [요리, Ai, 운동] 키워드 중 가까운 것을 하나 골라서 반환해줘"}],
]

chats = [
    [{"role": "system", "content": "당신은 다음 문장에 대해서 [요리, Ai, 운동] 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요"},
    {"role": "user", "content": "\"Hyper Clova X 모델에 대해 설명해줘.\""}],
]

chats = [
    [{"role": "system", "content": "당신은 다음 문장에 대해서 [요리, Ai, 운동] 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요"},
    {"role": "user", "content": "\"스미스 머신으로 할 수 있는 것들에 대해 알려줘.\""}],
]

chats = [
    [{"role": "system", "content": "당신은 다음 문장에 대해서 다음 [요리, 건강, Ai, 운동] 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요"},
    {"role": "user", "content": "\"조류 독감(Ai)가 예방하기 위해 좋은 습관에 대해 알려줘\""}],
]

chats = [
    [{"role": "system", "content": "당신은 다음 문장에 대해서 다음 [요리, 건강, Ai, 운동, 음악] 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요"},
    {"role": "user", "content": "\"에스파가 부른 노래는 뭐뭐가 있어?\""}],
]

# --- 4. 테스트용 채팅 데이터 ---
chats = [
    [{"role": "system", "content": "당신은 친절하게 사용자의 질문에 대해 답변하는 꿈돌이 로봇입니다, 묻는 바에 대해 최대한 간결하게 세 문장 이내로 알려주세요. 모르는 내용이나 확실하지 않은 정보는 절대 추측하거나 지어내지 말고, '잘 모르겠습니다' 또는 '확실하지 않습니다'라고 정직하게 답변해주세요."},
    {"role": "user", "content": "에스파가 부른 노래는 뭐뭐가 있어?"}],
]


chats = [
    [{"role": "system", "content": "당신은 다음 문맥에 대해서 다음 [요리, 건강, Ai, 운동, 음악] 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요"},
    {"role": "user", "content": "\"에스파(aespa)는 SM 엔터테인먼트 소속의 4인조 걸그룹으로, 2020년 11월 데뷔 이후 꾸준한 활동을 통해 국내외에서 많은 사랑을 받고 있습니다. 에스파가 부른 노래는 다음과 같습니다.\
    - 2020년 11월 발매한 <Next Level> - 데뷔곡\
    - 2021년 1월 발매한 <Black Mamba> - 2021년 제 59회 그래미 어워드에서 '톱 소셜 미디어 아티스트' 부문 후보에 올랐던 곡\
    - 2021년 6월 발매한 <Savage> - 2021년 제 63회 그래미 어워드에서 '톱 소셜 미디어 아티스트' 부문 후보에 올랐던 곡\
    \""}],]


chats = [
    [{"role": "system", "content": "당신은 다음 문장에 대해서 다음 리스트 [요리, 건강, Ai, 운동, 음악]에 있는 키워드 중 가까운 것을 하나 골라서 단순 반환해주세요, 리스트에 없는 키워드를 반환하면 벌점 100점을 받을겁니다"},
    {"role": "user", "content": "\"걸그룹 에스파에 대해 알려줄래?\""}],
]


# --- 5. Chain invoke 실행 ---
for chat in chats:
    result = llm.invoke(chat)
    print("프롬프트: ", chat)
    print("-----------------------------------------------------------------------------------")
    print(result["answer"])
    print(f"처리 시간: {result['elapsed_time']:.4f}초\n")
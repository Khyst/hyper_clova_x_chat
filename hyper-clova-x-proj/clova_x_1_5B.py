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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter

# --- 출력 클리닝 함수 최종 강화 ---
def clean_llm_response(text):
    """
    LLM 응답에서 불필요한 태그, 반복, 코드 블록을 제거하고 텍스트를 정리합니다.
    """
    text = text.strip()
    
    # 1. 반복적인 Chat Template 태그 및 불필요한 시작 키워드 정의
    # 사용자가 보고한 특정 반복 문구 포함
    keywords_to_remove = [
        "답변:", "assistant:", "<|im_start|>assistant", "user", "system",
        "이를 통해 사용자는 복잡한 작업을 더 쉽게 수행할 수 있습니다.assistant",
        "LangChain은 언어 모델 체인을 통해 다양한 언어 모델과 도구를 통합하여 작업 수행을 자동화하는 프레임워크입니다."
    ]
    
    # 최대 5번 반복 제거 시도 (더 공격적으로)
    for _ in range(5): 
        original_text = text
        
        # 키워드 제거
        for keyword in keywords_to_remove:
            if text.startswith(keyword):
                text = text[len(keyword):].strip()
        
        # 특정 반복 패턴 제거 (문장 자체 제거)
        if text.startswith("이를 통해 사용자는 복잡한 작업을 더 쉽게 수행할 수 있습니다."):
             text = text[len("이를 통해 사용자는 복잡한 작업을 더 쉽게 수행할 수 있습니다."):].strip()
        
        if text == original_text:
            break

    # 2. 정규 표현식을 사용하여 코드 블록 (````...````) 및 마크다운 형식 제거
    # re.DOTALL 플래그는 .이 줄 바꿈 문자도 포함하도록 합니다.
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```markdown.*?$', '', text, flags=re.DOTALL).strip() # 마크다운 시작 태그 및 그 이후 텍스트 제거
    
    # 3. 불필요한 빈 줄 제거 및 줄 바꿈 정리
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n'.join(text_lines)
    
    return text.strip()


# --- 환경 설정 및 가상 문서 파일 생성 ---

RAG_DOC_PATH = "rag_source_document.txt"
if not os.path.exists(RAG_DOC_PATH):
    with open(RAG_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("LangChain은 LLM(대규모 언어 모델)을 기반으로 한 애플리케이션 개발을 위한 프레임워크입니다. "
                "주요 목적은 언어 모델을 외부 데이터 소스 및 컴퓨팅과 연결하여 기능을 확장하는 것입니다. "
                "LangChain의 핵심 구성 요소는 모델 I/O, 검색(Retrieval), 체인(Chains) 등입니다. "
                "RAG는 외부 지식(문서)을 검색하여 LLM이 더 정확하고 최신 정보가 반영된 답변을 생성하도록 돕습니다. "
                "\n\n슈뢰딩거 방정식은 양자역학의 기본 방정식으로, 시간에 따라 파동함수(Wave function, $\Psi$)가 어떻게 진화하는지를 설명합니다. "
                "파동함수의 절댓값 제곱($|\Psi|^2$)은 특정 위치에서 입자를 발견할 확률 밀도를 나타냅니다. "
                "양자역학에서 슈뢰딩거 방정식은 이 확률적 행동을 예측하는 핵심 도구입니다.")
    print(f"'{RAG_DOC_PATH}' 파일이 생성되었습니다.")


# --- 1. 설정 및 로컬 모델 로드 (HuggingFace LLM) ---

local_model_path = "../models/HyperCLOVAX-SEED-Text-Instruct-1.5B" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n사용 장치: {device}")

try:
    model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print("LLM이 로컬 경로에서 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")


# --- 2. Hugging Face Pipeline 및 LLM 객체 생성 ---

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device.index if device.type == "cuda" else -1,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.7,
    return_full_text=False # 입력 프롬프트 제외
)

llm = HuggingFacePipeline(pipeline=pipe)
print("HuggingFace LLM 파이프라인 객체 생성 완료.")


# --- 3. RAG 시스템 구성 ---

loader = TextLoader(RAG_DOC_PATH, encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"총 {len(texts)}개의 문서 청크를 생성했습니다.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 
print("Chroma 벡터스토어 및 Retriever 객체 생성 완료.")


# --- 4. RAG Chain 구성 (LCEL 사용 - 출처 포함) ---

RAG_TEMPLATE = """당신은 주어진 맥락(Context)을 사용하여 질문(Question)에 답변하는 AI입니다. 
제공된 맥락과 관련이 없는 질문이라면, 관련 정보가 없다고 정중하게 답변하십시오.
답변은 항상 사용된 맥락에 근거해야 합니다.

맥락(Context):
{context}

질문(Question): {question}

답변:
"""
RAG_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context_pipe = itemgetter("question") | retriever

answer_pipe = (
    RunnableParallel({
        "context": context_pipe | format_docs, 
        "question": itemgetter("question")
    })
    | RAG_PROMPT
    | llm
)

full_rag_chain = RunnableParallel(
    context=context_pipe, 
    answer=answer_pipe
)
print("RAG Chain 파이프라인 (답변 및 출처 포함) 구축 완료.")


# --- 5. RAG 모델 테스트: Chain Invoke 및 정갈한 CLI 출력 ---

prompts = [
    "LangChain은 어떤 목적으로 사용되는 프레임워크야?",
    "슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘.",
    "프랑스 혁명은 언제 일어났니?",
]

# LLM 출력을 멈출 시퀀스 정의
STOP_SEQUENCES = ["\n\n답변:", "답변:\n\n", "맥락(Context):", "<|endofturn|>", "<|stop|>"]

print("\n" + "=" * 70)
print("RAG Chain 테스트 시작")
print("=" * 70)

for i, prompt in enumerate(prompts):
    print("\n" + "#" * 70)
    print(f"   👉 {i+1}. 사용자 질문: {prompt}")
    print("#" * 70)
    
    start_time = time.time()
    
    # invoke 호출 시 config를 통해 STOP 시퀀스 전달
    result = full_rag_chain.invoke(
        {"question": prompt},
        config={
            "stop": STOP_SEQUENCES 
        }
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 1. 답변 텍스트 정리 및 클리닝 (강화된 함수 사용)
    answer_text = clean_llm_response(result['answer'])
    
    # 2. 정갈한 답변 출력
    print("-" * 30 + " 🤖 답변 " + "-" * 30)
    print(answer_text)
    print("-" * 70)
    
    # 3. 출처 정보 출력
    print("📚 사용된 출처 정보:")
    
    if result['context']:
        sources = set([doc.metadata.get('source', '알 수 없음') for doc in result['context']])
        for src in sources:
            print(f"   - {src}")
    else:
        print("   - RAG 체인을 통해 검색된 문서가 없습니다.")
    
    print("-" * 70)
    print(f"✅ 처리 완료 | 소요 시간: {elapsed_time:.2f}초")


print("\n" + "=" * 70)
print("RAG Chain 테스트가 완료되었습니다.")
print("=" * 70)
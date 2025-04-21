import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import requests
import webbrowser
import threading
import time

# 1. PDF → 텍스트
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# 2. 텍스트 → 문단 나누기
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# 3. 문단 → 임베딩 + 인덱싱
def build_faiss(chunks, embed_model):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# 4. 질의 → 유사 문단 검색
def search_similar(query, embed_model, index, chunks, k=3):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# 5. 로컬 LLM에 프롬프트 요청
def query_local_llm(prompt, model_name, api_url="http://127.0.0.1:1234/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "당신은 문서를 분석하고 설명해주는 조수입니다."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        res = requests.post(api_url, json=payload, headers=headers)
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# PDF 파일 목록을 가져오는 함수
def get_pdf_files():
    directory = "./docs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

# 그라디오 인터페이스 함수
def analyze_pdf(pdf_filename, user_question, model_name, api_url):
    if not pdf_filename:
        return "PDF 파일을 선택해주세요."
    
    if not user_question:
        return "질문을 입력해주세요."
    
    if not model_name:
        return "LLM 모델 이름을 입력해주세요."
    
    if not api_url:
        api_url = "http://127.0.0.1:1234/v1/chat/completions"
    
    try:
        # 전체 경로 구성
        pdf_path = os.path.join("./docs", pdf_filename)
        
        # 임베딩 모델 로드
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # PDF 텍스트 추출
        progress_text = f"📄 '{pdf_filename}' 파일을 처리 중...\n"
        raw_text = extract_text_from_pdf(pdf_path)
        
        # 문단 나누기
        chunks = chunk_text(raw_text)
        
        # 임베딩 및 인덱싱
        progress_text += "📐 임베딩 및 인덱싱 중...\n"
        index, _, chunks = build_faiss(chunks, embed_model)
        
        # 유사 문단 검색
        progress_text += "🔍 질문에 관련된 문단 검색 중...\n"
        related = search_similar(user_question, embed_model, index, chunks)
        
        # 로컬 AI에 질의
        progress_text += f"🧠 로컬 AI ({model_name})에게 질의 중...\n\n"
        context = "\n\n".join(related)
        prompt = f"다음 문서를 기반으로 질문에 답해주세요:\n\n{context}\n\n질문: {user_question}"
        answer = query_local_llm(prompt, model_name, api_url)
        
        # 결과 반환
        result = f"{progress_text}-------- 분석 결과 --------\n\n{answer}"
        return result
    
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# 브라우저 자동 실행 함수
def open_browser(port=7860):
    def _open_browser():
        # 서버가 시작될 시간을 주기 위해 2초 대기
        time.sleep(2)
        # 기본 브라우저로 Gradio 페이지 열기
        webbrowser.open(f'http://127.0.0.1:{port}')
    
    # 별도의 스레드에서 브라우저 열기
    threading.Thread(target=_open_browser).start()

# Gradio 인터페이스 설정
with gr.Blocks(title="PDF 분석기") as app:
    gr.Markdown("# 📄 PDF 분석기")
    gr.Markdown("docs 폴더에 있는 PDF 파일을 선택하고 질문을 입력하세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # PDF 파일 선택 드롭다운
            pdf_dropdown = gr.Dropdown(
                label="PDF 파일 선택", 
                choices=get_pdf_files(),
                interactive=True
            )
            
            # 새로고침 버튼
            refresh_btn = gr.Button("📂 파일 목록 새로고침")
            
            # 질문 입력
            question_input = gr.Textbox(
                label="질문 입력",
                placeholder="예: 이 문서의 핵심 내용을 요약해줘",
                lines=3
            )
            
            # LLM 모델 이름 입력 (텍스트 필드)
            model_input = gr.Textbox(
                label="LM Studio 모델 이름 입력",
                placeholder="모델 이름을 입력하세요 (예: llama-dna-1.0-8b-instruct)",
                value="llama-dna-1.0-8b-instruct"
            )
            
            # API URL 입력
            api_url_input = gr.Textbox(
                label="API URL (선택사항)",
                placeholder="http://127.0.0.1:1234/v1/chat/completions",
                value="http://127.0.0.1:1234/v1/chat/completions"
            )
            
            # 분석 버튼
            analyze_btn = gr.Button("🔍 분석하기", variant="primary")
        
        with gr.Column(scale=2):
            # 결과 출력
            result_output = gr.Textbox(
                label="분석 결과",
                lines=15,
                placeholder="분석 결과가 여기에 표시됩니다..."
            )
    
    # 새로고침 버튼 이벤트
    def refresh_pdf_list():
        return gr.Dropdown(choices=get_pdf_files())
    
    refresh_btn.click(refresh_pdf_list, outputs=[pdf_dropdown])
    
    # 분석 버튼 이벤트
    analyze_btn.click(
        analyze_pdf, 
        inputs=[pdf_dropdown, question_input, model_input, api_url_input], 
        outputs=[result_output]
    )
    
    # 도움말 표시
    gr.Markdown("""
    ## 사용 방법
    1. docs 폴더에 분석할 PDF 파일을 넣으세요.
    2. PDF 파일을 선택하고 질문을 입력하세요.
    3. LM Studio에서 로드한 모델의 이름을 정확히 입력하세요.
    4. API URL이 기본값과 다른 경우 수정하세요.
    5. '분석하기' 버튼을 클릭하세요.
    
    ## 주의사항
    - LM Studio에서 사용하는 모델 이름을 정확히 입력해야 합니다.
    - LM Studio API 서버가 실행 중이어야 합니다.
    """)

# 그라디오 앱 실행
if __name__ == "__main__":
    # 브라우저 자동 실행 함수 호출
    open_browser()
    
    # Gradio 앱 실행
    app.launch(share=False)
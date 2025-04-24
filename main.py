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
import pytesseract
from PIL import Image
import io
import torch
import cv2
import concurrent.futures
from tqdm import tqdm

# GPU 초기화 및 확인
def init_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("GPU not available, using CPU instead")
        return False

# 이미지 전처리 함수 (GPU 가속)
def preprocess_image(img):
    # OpenCV로 이미지 처리 (GPU 가속)
    if isinstance(img, bytes) or isinstance(img, io.BytesIO):
        if isinstance(img, bytes):
            img = io.BytesIO(img)
        img = np.array(Image.open(img))
    
    # BGR로 변환 (OpenCV 형식)
    if img.shape[2] == 4:  # RGBA 이미지인 경우
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.shape[2] == 3:  # RGB 이미지인 경우
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 이미지 개선 (GPU 가속 활용)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:  # CUDA 지원 확인
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        
        # 그레이스케일 변환 (GPU)
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거 (GPU)
        gpu_denoised = cv2.cuda.fastNlMeansDenoising(gpu_gray)
        
        # 대비 향상 (GPU)
        gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gpu_enhanced = gpu_clahe.apply(gpu_denoised)
        
        # CPU로 다시 가져오기
        enhanced_img = gpu_enhanced.download()
    else:
        # CPU 대체 처리
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(denoised)
    
    return enhanced_img

# 병렬 OCR 처리 함수
def process_page_with_ocr(page, use_ocr=True):
    # 일반 텍스트 추출 시도
    text = page.get_text()
    
    # 텍스트가 거의 없고 OCR이 활성화된 경우, 이미지 기반 OCR 수행
    if use_ocr and len(text.strip()) < 100:  # 텍스트가 적다면 이미지일 가능성 높음
        # 페이지를 고해상도 이미지로 렌더링
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 해상도 3배 증가
        img_bytes = pix.tobytes("png")
        
        # 이미지 전처리
        enhanced_img = preprocess_image(img_bytes)
        
        # 이미지를 PIL로 변환하여 OCR 처리
        ocr_text = pytesseract.image_to_string(
            Image.fromarray(enhanced_img), 
            lang='kor+eng',  # 한국어+영어 OCR
            config='--oem 1 --psm 6 -c tessedit_do_invert=0'  # LSTM OCR 엔진, 균일한 텍스트 블록
        )
        
        # OCR 결과가 더 많은 텍스트를 제공하면 사용
        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text
    
    return text

# OCR 기능을 통합한 PDF 텍스트 추출 (GPU 가속 및 병렬 처리)
def extract_text_from_pdf(pdf_path, use_ocr=True):
    # GPU 초기화
    has_gpu = init_gpu()
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 병렬 처리를 위한 ThreadPoolExecutor 설정
    # PDF 처리는 I/O 바운드와 CPU 바운드 작업이 섞여 있으므로 ThreadPoolExecutor가 적합
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 각 페이지를 병렬로 처리
        future_to_page = {executor.submit(process_page_with_ocr, doc[page_num], use_ocr): page_num for page_num in range(total_pages)}
        
        # 결과 수집 (페이지 순서 유지)
        text_content = [""] * total_pages
        
        # tqdm으로 진행 상황 표시
        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=total_pages, desc="OCR 처리 중"):
            page_num = future_to_page[future]
            try:
                text_content[page_num] = future.result()
            except Exception as e:
                print(f"페이지 {page_num} 처리 중 오류 발생: {e}")
                text_content[page_num] = ""
    
    return "\n".join(text_content)

# 2. 텍스트 → 문단 나누기
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# 3. 문단 → 임베딩 + 인덱싱 (GPU 가속)
def build_faiss(chunks, embed_model):
    # GPU 사용 가능 여부 확인
    use_gpu = torch.cuda.is_available()
    
    # 임베딩 생성 (GPU 사용)
    # SentenceTransformer는 자동으로 GPU를 감지하고 사용함
    embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True)
    
    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    
    if use_gpu:
        # GPU 가속 인덱스 생성
        # 먼저 CPU에서 인덱스 생성
        index = faiss.IndexFlatL2(dimension)
        
        # GPU로 인덱스 이동
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # 임베딩 추가
        gpu_index.add(np.array(embeddings).astype(np.float32))
        
        # 검색을 위해 CPU로 다시 이동 (선택사항)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        # CPU 인덱스 생성
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype(np.float32))
    
    return index, embeddings, chunks

# 4. 질의 → 유사 문단 검색 (GPU 가속)
def search_similar(query, embed_model, index, chunks, k=3):
    # GPU 사용 가능 여부 확인 (임베딩 계산에 사용)
    use_gpu = torch.cuda.is_available()
    
    # 쿼리 임베딩 계산 (GPU 사용 가능시 자동 활용)
    q_emb = embed_model.encode([query])
    
    # GPU로 검색
    if use_gpu and 'StandardGpuResources' in dir(faiss):
        # GPU로 인덱스 이동
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # GPU로 검색
        D, I = gpu_index.search(np.array(q_emb).astype(np.float32), k)
    else:
        # CPU로 검색
        D, I = index.search(np.array(q_emb).astype(np.float32), k)
    
    # 유사도 점수와 함께 결과 반환
    results = []
    for i, dist in zip(I[0], D[0]):
        if i < len(chunks):
            results.append({
                'chunk': chunks[i],
                'similarity_score': 1.0 / (1.0 + dist)  # 거리를 유사도 점수로 변환
            })
    
    # 가장 유사한 문단만 반환
    return [item['chunk'] for item in results]

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
def analyze_pdf(pdf_filename, user_question, model_name, api_url, use_ocr, use_gpu):
    if not pdf_filename:
        return "PDF 파일을 선택해주세요."
    
    if not user_question:
        return "질문을 입력해주세요."
    
    if not model_name:
        return "LLM 모델 이름을 입력해주세요."
    
    if not api_url:
        api_url = "http://127.0.0.1:1234/v1/chat/completions"
    
    try:
        # GPU 가용성 확인
        gpu_available = torch.cuda.is_available()
        if use_gpu and not gpu_available:
            return "GPU 사용을 선택했지만 사용 가능한 GPU가 없습니다. GPU 드라이버 설치 및 CUDA 호환성을 확인해주세요."
        
        # 전체 경로 구성
        pdf_path = os.path.join("./docs", pdf_filename)
        
        # 실행 정보 및 진행 상황
        progress_text = f"📄 '{pdf_filename}' 파일 처리 중...\n"
        
        # GPU 정보 표시
        if use_gpu and gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위로 변환
            progress_text += f"🚀 GPU 가속 활성화: {gpu_name} ({gpu_memory:.2f} GB)\n"
        
        # 임베딩 모델 로드 (GPU 자동 사용)
        progress_text += "🔄 임베딩 모델 로드 중...\n"
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # GPU/CUDA 확인
        if use_gpu and gpu_available:
            embed_model = embed_model.to(torch.device("cuda"))
        
        # PDF 텍스트 추출 (OCR 사용 여부에 따라)
        if use_ocr:
            progress_text += "🔍 OCR을 사용하여 이미지 내 텍스트 추출 중 (병렬 처리)...\n"
        else:
            progress_text += "📑 텍스트 추출 중...\n"
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 텍스트 추출
        raw_text = extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
        # 텍스트 추출 시간 계산
        extraction_time = time.time() - start_time
        progress_text += f"⏱️ 텍스트 추출 완료 ({extraction_time:.2f}초)\n"
        
        # 문단 나누기
        chunks = chunk_text(raw_text)
        progress_text += f"📋 총 {len(chunks)}개 문단으로 분할\n"
        
        # 임베딩 및 인덱싱 시간 측정 시작
        start_time = time.time()
        
        # 임베딩 및 인덱싱
        progress_text += "📐 임베딩 및 인덱싱 중 (GPU 가속)...\n"
        index, embeddings, chunks = build_faiss(chunks, embed_model)
        
        # 임베딩 시간 계산
        embedding_time = time.time() - start_time
        progress_text += f"⏱️ 임베딩 완료 ({embedding_time:.2f}초)\n"
        
        # 유사 문단 검색
        progress_text += "🔍 질문에 관련된 문단 검색 중...\n"
        related = search_similar(user_question, embed_model, index, chunks)
        
        # 로컬 AI에 질의
        progress_text += f"🧠 로컬 AI ({model_name})에게 질의 중...\n\n"
        context = "\n\n".join(related)
        prompt = f"""다음 문서 내용을 기반으로 질문에 답해주세요:

문서 내용:
---
{context}
---

질문: {user_question}

답변을 할 때는 주어진 문서 내용에서 찾을 수 있는 정보만 사용하세요. 확실하지 않은 정보는 포함하지 마세요."""
        
        # LLM 응답 시간 측정
        start_time = time.time()
        answer = query_local_llm(prompt, model_name, api_url)
        llm_time = time.time() - start_time
        
        # 총 소요 시간 정보 추가
        progress_text += f"⏱️ AI 응답 생성 완료 ({llm_time:.2f}초)\n"
        
        # 결과 반환
        result = f"{progress_text}\n-------- 분석 결과 --------\n\n{answer}"
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"오류가 발생했습니다: {str(e)}\n\n상세 오류: {error_details}"

# 브라우저 자동 실행 함수
def open_browser(port=7860):
    def _open_browser():
        # 서버가 시작될 시간을 주기 위해 2초 대기
        time.sleep(2)
        # 기본 브라우저로 Gradio 페이지 열기
        webbrowser.open(f'http://127.0.0.1:{port}')
    
    # 별도의 스레드에서 브라우저 열기
    threading.Thread(target=_open_browser).start()

# 시스템 정보 확인 함수
def get_system_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "cpu_count": os.cpu_count(),
    }
    
    # OpenCV CUDA 지원 확인
    if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
        info["opencv_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    else:
        info["opencv_cuda"] = False
    
    # FAISS GPU 지원 확인
    info["faiss_gpu"] = hasattr(faiss, 'StandardGpuResources')
    
    return info

# GPU 사용 가능성 확인
def check_gpu_availability():
    system_info = get_system_info()
    if system_info["cuda_available"]:
        gpu_names = ", ".join(system_info["gpu_names"])
        return f"✅ GPU 가속 사용 가능: {gpu_names}"
    else:
        return "❌ GPU 가속 불가: CUDA 지원 GPU가 감지되지 않았습니다."


# Gradio 인터페이스 설정
with gr.Blocks(title="PDF 분석기 (GPU 가속)") as app:
    gr.Markdown("# 🚀 PDF 분석기 (OCR + GPU 가속)")
    
    # 시스템 정보 확인
    system_info = get_system_info()
    if system_info["cuda_available"]:
        gpu_info = f"💻 시스템: CPU {system_info['cpu_count']}코어, GPU {system_info['gpu_count']}대 ({', '.join(system_info['gpu_names'])})"
        gr.Markdown(f"{gpu_info}\n\n스캔된 문서는 OCR을 활성화하고, 처리 속도 향상을 위해 GPU 가속을 사용하세요.")
    else:
        gr.Markdown("💻 GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다.\n\n스캔된 문서는 OCR을 활성화하세요.")
    
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
            
            with gr.Row():
                # OCR 사용 여부 체크박스
                ocr_checkbox = gr.Checkbox(
                    label="OCR 활성화",
                    value=True,
                    info="스캔된 문서나 이미지 PDF에 텍스트를 추출할 때 OCR을 사용합니다."
                )
                
                # GPU 사용 여부 체크박스
                gpu_checkbox = gr.Checkbox(
                    label="GPU 가속",
                    value=system_info["cuda_available"],  # GPU 사용 가능하면 기본 활성화
                    interactive=system_info["cuda_available"],  # GPU 없으면 비활성화
                    info="CUDA GPU를 사용하여 처리 속도를 향상시킵니다."
                )
            
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
            
            # API URL 입력 - 이 부분이 잘렸었습니다
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
        inputs=[pdf_dropdown, question_input, model_input, api_url_input, ocr_checkbox, gpu_checkbox], 
        outputs=[result_output]
    )
    
    # 도움말 표시
    gr.Markdown("""
    ## 사용 방법
    1. docs 폴더에 분석할 PDF 파일을 넣으세요.
    2. PDF 파일을 선택하고 질문을 입력하세요.
    3. 스캔된 문서나 이미지 기반 PDF인 경우 "OCR 활성화" 체크박스를 선택하세요.
    4. GPU가 있는 경우 "GPU 가속" 체크박스를 선택하면 처리 속도가 향상됩니다.
    5. LM Studio에서 로드한 모델의 이름을 정확히 입력하세요.
    6. API URL이 기본값과 다른 경우 수정하세요.
    7. '분석하기' 버튼을 클릭하세요.
    
    ## 주의사항
    - OCR 처리는 시간이 더 오래 걸릴 수 있지만, GPU 가속을 사용하면 속도가 향상됩니다.
    - 한국어와 영어 OCR을 지원합니다.
    - GPU 가속을 사용하려면 CUDA와 호환되는 NVIDIA GPU가 필요합니다.
    - LM Studio에서 사용하는 모델 이름을 정확히 입력해야 합니다.
    - LM Studio API 서버가 실행 중이어야 합니다.
    - OCR 사용을 위해 pytesseract와 Tesseract OCR 설치가 필요합니다.
    """)

# 그라디오 앱 실행
if __name__ == "__main__":
    # 브라우저 자동 실행 함수 호출
    open_browser()
    
    # Gradio 앱 실행
    app.launch(share=False)

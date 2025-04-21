# 설치
1. 깃을 설치합니다.
2. 파이썬 3.10버전을 설치합니다.
3. "설치.BAT" 배치파일로 원클릭 설치하면 끝.



# 실행시 주의할 점
1. 로컬 LLM 어플리케이션이랑 같이 쓰는 구성입니다. 즉, 올라마라든지 LM스튜디오를 반드시 설치해주시기 바랍니다.
2. 그리고 로컬에서 모델명과 API를 불러올 수 있는 주소를 적어도 맞춰야 한다는건 기초입니다, 제발!
![image](https://github.com/user-attachments/assets/0998506d-337e-4b3a-a1ae-fb2a5281dc4f)



# 현재 추천드리는 한국어 로컬 모델 
모델명 | 파라미터 수 | 주요 특징 | 라이선스
GECKO-7B | 7B | 한국어-영어 이중언어 모델로, KMMLU에서 우수한 성능을 보임 | 
Gukbap Qwen2.5 7B | 7B | 한국어 논리 및 언어 이해에서 GPT-4를 능가하는 성능을 보임 | 
KoLlama2 7B | 7B | LLaMA2 기반으로 LoRA를 활용한 한국어 성능 향상 모델 | 
LLaMA-Pro-Ko 8B | 8B | LLaMA2-7B에 한국어 전용 토크나이저를 추가하여 한국어 성능 강화 | 
Llama 3.1 Korean 8B Instruct | 8B | LLaMA 3.1 기반으로 고품질 한국어 데이터셋을 활용한 지시어 튜닝 모델 | 
DNA 1.0 8B Instruct | 8B | LLaMA 3.1 기반으로 한국어와 영어 모두에서 우수한 성능을 보이는 이중언어 모델 | 



# LocaRag_Easy
A RAG-based PDF analyzer that retrieves relevant chunks from documents and uses a local LLM to answer user questions through a Gradio GUI interface.

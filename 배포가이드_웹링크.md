# 🌐 챗봇을 사이트로 배포하기 (Streamlit Cloud, 무료)

배포하면 `https://내앱이름.streamlit.app` 주소가 생기고, **누구나 로그인·API 키 없이** LLM 답변까지 쓸 수 있습니다.
API 키는 사이트 설정(Secrets)에 숨겨 두기 때문에 화면이나 코드에 노출되지 않습니다.

준비물: 구글 계정(Gemini 키), GitHub 계정 (없으면 https://github.com 에서 무료 가입)

---

## 1단계. Gemini 무료 API 키 만들기
1. https://aistudio.google.com 접속 → 구글 로그인
2. 왼쪽 **API keys** → **Create API key** → 키(`AIza...`) 복사
3. ⚠️ 결제 정보는 연결하지 마세요 (무료 등급 유지 = 요금 0원)

## 2단계. GitHub에 코드 올리기
1. https://github.com 로그인 → 오른쪽 위 **+** → **New repository**
2. Repository name: `grad-chatbot` / **Public** 선택 → **Create repository**
3. 화면의 **uploading an existing file** 클릭
4. 압축 푼 `grad_chatbot` 폴더 **안의 내용물 전부**를 드래그해서 올리기
   - `app.py`, `requirements.txt`, `rag/`, `data/`, `docs/` 가 저장소 맨 위에 보여야 함
   - ⚠️ `.env` 파일이 있다면 올리지 마세요 (키 노출)
5. **Commit changes** 클릭

## 3단계. Streamlit Cloud에 배포
1. https://share.streamlit.io 접속 → **Continue with GitHub** 로 로그인
2. **Create app** → **Deploy a public app from GitHub**
3. 입력:
   - Repository: `내아이디/grad-chatbot`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: 원하는 이름 (예: `sch-grad-chatbot`)
4. **Advanced settings** → **Secrets** 칸에 아래를 그대로 붙여넣고 `AIza...` 부분만 내 키로 바꾸기:
   ```toml
   OPENAI_API_KEY = "AIza...여기에_내_Gemini_키"
   OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
   CHAT_MODEL = "gemini-3.8-flash"
   EMBED_MODEL = "gemini-embedding-2"
   ```
5. **Deploy** → 2~5분 기다리면 사이트가 열림

## 4단계. 확인
- 사이드바에 **"LLM 연결됨: gemini-3.8-flash"** 가 보이면 성공
- 질문 탭에서 예시 질문 2~3개, 자가진단 탭에서 판정 1번 해보기
- 브라우저 **시크릿 창**에서 주소를 열어 로그인 없이 열리는지 확인

## 문제가 생기면
| 증상 | 해결 |
|---|---|
| 사이드바에 "오프라인 모드" | Secrets 에 `OPENAI_API_KEY` 가 제대로 들어갔는지 확인 → 앱 메뉴 ⋮ → Reboot app |
| 답변 중 `429` / `limit: 0` 오류 | 무료 사용량 초과 또는 모델 미지원 → Secrets 의 `CHAT_MODEL` 을 다른 flash 모델로 변경 |
| 임베딩 관련 오류 | Secrets 에 `USE_EMBEDDINGS = "0"` 추가 (TF-IDF 검색만 사용, 기능은 그대로) |
| 사이트가 잠자는 화면 | 한동안 접속이 없으면 잠들 수 있음 → 버튼 눌러 깨우기. **제출 직전에 한 번 접속**해 두기 |

---

## 제출 문구 예시
> **챗봇 사이트:** https://(내앱이름).streamlit.app
> - 접속 방법: 링크를 열면 바로 사용할 수 있습니다 (로그인·설치·API 키 불필요).
> - 왼쪽에서 학과와 학번을 선택한 뒤,
>   ① **졸업요건 Q&A** 탭: 졸업요건을 질문하면 안내문 근거와 함께 답합니다. 들은 과목과 활동을 문장으로 적으면 졸업 충족 여부도 계산합니다.
>   ② **졸업 자가진단** 탭: 수강 과목을 붙여넣고 활동을 체크하면 점수와 부족 항목을 보여줍니다.
> - 사이트가 잠든 화면이 나오면 버튼을 눌러 1분 정도 기다려 주세요.
>
> **코드:** https://github.com/(내아이디)/grad-chatbot (또는 첨부 zip)

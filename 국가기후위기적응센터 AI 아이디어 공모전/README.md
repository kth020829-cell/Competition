# Aider 🌿
### AI 기반 기후적응 헬스케어 플랫폼
> *"위기 상황에서 가장 먼저 반응하는 도우미"*

[![Status](https://img.shields.io/badge/status-demo--deployed-brightgreen)]()
[![Stack](https://img.shields.io/badge/AI-LightGBM%20%7C%20Autoencoder%20%7C%20Transformer-blue)]()
[![Infra](https://img.shields.io/badge/AWS-Lambda%20%7C%20API%20Gateway%20%7C%20DynamoDB%20%7C%20CloudFront-orange)]()

---

## 🚀 라이브 데모

실제로 AWS에 배포되어 동작하는 데모입니다. (시뮬레이션 모드 — 실제 사용자 데이터 없이 동작)

| 항목 | 링크 |
|------|------|
| 웹 대시보드 | https://d1qw2blagzq2nu.cloudfront.net |
| API 문서 (Swagger) | https://s68pb10tzf.execute-api.ap-northeast-2.amazonaws.com/prod/docs |
| API 헬스체크 | https://s68pb10tzf.execute-api.ap-northeast-2.amazonaws.com/prod/health |

> 데모/포트폴리오 목적의 개인 AWS 계정에 배포되어 있어, 예고 없이 내려가거나 URL이 바뀔 수 있습니다.

---

## 📌 프로젝트 개요

**Aider**는 폭염·한파·미세먼지 등 극단적 기후 환경에서 **노인, 만성질환자, 1인가구 등 기후 취약계층**을 보호하기 위한 **AI 기반 기후적응형 돌봄 플랫폼**입니다.

기존의 기후변화 대응이 온실가스 감축 등 **'대응(mitigation)' 중심**이었다면, Aider는 변화된 환경 속에서 스스로를 보호하기 위한 **'적응(adaptation)' 중심** 전략에 초점을 맞춥니다.

웨어러블 생체신호, IoT 환경센서, 공공 기후데이터를 융합하여 자체 개발한 **CHAI(Climate Health Adaptation Index)** 알고리즘으로 개인별 기후건강 위험지수를 실시간 산출하고, 돌봄 자원을 선제적으로 배분합니다.

이 저장소의 [`aider/`](aider/) 폴더에는 위 아이디어를 실제로 구현하여 **AWS 서버리스 인프라에 배포한 풀스택 프로토타입**의 전체 소스코드가 포함되어 있습니다.

---

## 🎯 해결하고자 하는 문제

- 폭염·한파로 인한 온열·한랭질환자 매년 증가
- 기후 취약계층(고령자, 만성질환자)의 건강 악화 및 돌봄 공백
- 현재 돌봄 체계는 **'재난 발생 후 대응' 중심** → 사전 예측·선제 관리 부재
- 돌봄 인력의 우선순위 없는 비효율적 대응
- '기후'와 '건강'을 통합 분석하는 플랫폼의 부재

---

## 💡 핵심 아이디어: CHAI 모델

CHAI는 기후로 인한 건강 위험이 단일 요인이 아닌, **'기후 변화 ↔ 신체 반응 ↔ 생활환경 ↔ 개인 취약성'** 네 가지 요소가 상호작용한 결과라는 관점에서 설계되었습니다.

### CHAI 수식

```
CHAIₜ = σ( f(Cₜ, Bₜ, Eₜ) + w_personal )
```

| 변수 | 의미 | 설명 | 구현체 |
|------|------|------|--------|
| **Cₜ** (Climate) | 기후 스트레스 | 기온·습도·예보·급변 신호를 함수화한 외부 기후 부담 | 기상청 동네예보/특보 API 연동 |
| **Bₜ** (Bio-behavioral) | 생체 반응 | HR, HRV, 체온 등 신체의 단기 반응 패턴 | Transformer 인코더 + Autoencoder 이상탐지 |
| **Eₜ** (Environmental) | 환경 노출 | 실내 온·습도, CO₂, WBGT 등 누적 부담 | 에어코리아 CAI API 연동 |
| **w_personal** | 개인 취약성 보정값 | 나이, 기저질환, BMI 등 개인 고유 민감도 | 가중 임베딩 (α·V + β·S + γ·R) |

### Personalization Layer

```
w_personal = α · V_personal + β · Sₜ + γ · Rₜ
```

- **V_personal**: 장기 취약성 임베딩 (기저질환, 연령 등)
- **Sₜ**: 실시간 생체 변화 (HR, HRV, SpO₂ 등)
- **Rₜ**: 현재 환경 강도 (WBGT, PM 농도 등)
- **가중치**: α=0.5, β=0.3, γ=0.2 (선행 연구 기반 고정값, [`config.py`](aider/backend/app/config.py))

### 최종 예측 파이프라인

1. **Transformer 인코더**: 생체 시계열(HR/HRV/피부온/SpO₂)을 patch embedding + self-attention으로 인코딩해 Bₜ 산출
2. **Autoencoder 이상탐지**: 정상 패턴 대비 재구성 오차(recon error)로 급변 징후를 별도 신호로 병행 포착
3. **LightGBM**: Cₜ, Bₜ, Eₜ, w_personal을 통합 입력으로 받아 최종 CHAI 위험 점수(0~1, Sigmoid) 산출
4. **분류**: `low / moderate / high / critical` 4단계 위험도 라벨링

### 출력

- 0~1 사이 위험 점수 (Sigmoid)
- 4단계 분류: **low / moderate / high / critical**
- 이상탐지 여부 및 재구성 오차값 함께 제공 (모델 판단 근거 투명화)

---

## 🏗 실제 배포 아키텍처 (AWS)

```
┌──────────────────────────┐        ┌────────────────────────────────────────────┐
│   사용자 브라우저          │  HTTPS │              CloudFront (CDN)               │
│  (React 대시보드)         ├───────►│  Origin: S3 (정적 호스팅, React 빌드 산출물)  │
└──────────────────────────┘        └────────────────────────────────────────────┘

┌──────────────────────────┐        ┌───────────────────┐   ┌───────────────────────┐
│   사용자 브라우저          │  HTTPS │   API Gateway      │   │  Lambda (Container)   │
│  (axios API 호출)        ├───────►│   (REST, CORS)     ├──►│  FastAPI + Mangum      │
└──────────────────────────┘        └───────────────────┘   │  LightGBM/Autoencoder/ │
                                                              │  Transformer 모델 로드  │
                                                              └─────────┬─────────────┘
                                                                        │
                                       ┌────────────────────────────────┼────────────────────┐
                                       ▼                                ▼                     ▼
                              ┌────────────────┐             ┌──────────────────┐   ┌──────────────────┐
                              │   DynamoDB      │             │  기상청 Open API   │   │  에어코리아 API    │
                              │ Users/Records/  │             │ (동네예보/특보)     │   │  (CAI 대기지수)    │
                              │ Alerts 테이블    │             │  ─ 키 미설정 시     │   │  ─ 키 미설정 시    │
                              └────────────────┘             │    자동 mock 응답   │   │    자동 mock 응답  │
                                                              └──────────────────┘   └──────────────────┘
```

- **인프라 정의**: [`infra/template.yaml`](aider/infra/template.yaml) (AWS SAM / CloudFormation)
- **배포 자동화**: [`deploy.ps1`](aider/deploy.ps1) 스크립트 하나로 Docker 빌드 → ECR 푸시 → SAM 배포 → 프론트엔드 빌드 → S3 업로드 → CloudFront 캐시 무효화까지 일괄 수행
- **콜드스타트 대응**: torch/LightGBM 임포트로 인한 Lambda 초기화 지연을 고려해 메모리 3008MB 할당(Lambda는 메모리에 비례해 CPU 배정)
- **CORS 방어**: 미처리 예외 발생 시에도 CORS 헤더가 포함된 JSON 응답을 반환하도록 전역 예외 핸들러 구성 ([`main.py`](aider/backend/app/main.py))

---

## 🧩 주요 기능 (구현 완료)

### 1. 시민(취약계층) 대시보드 — React 웹앱
- 실시간 CHAI 위험도 게이지 및 4단계 위험 라벨
- 기후 스트레스 시나리오 선택(폭염/한파/정상) 및 지속시간 시뮬레이션
- Cₜ/Bₜ/Eₜ/개인 요인별 기여도 시각화 (Feature Bar)
- 기상청 예보 카드, 시뮬레이션된 생체신호(HR/HRV/체온/SpO₂) 카드
- 폭염·한파 특보 알림 배너
- 이상탐지(Autoencoder) 결과 및 재구성 오차 표시

### 2. 백엔드 API (FastAPI)
- `POST /predictions/chai`: 실측 생체/환경 입력 기반 CHAI 예측
- `GET /predictions/simulate`: 시나리오 기반 시뮬레이션 예측 (데모용)
- `GET /predictions/history/{user_id}`: 사용자별 예측 이력 조회 (DynamoDB)
- `GET /alerts/forecast`, `GET /alerts/warnings`: 기상청 연동 예보/특보
- `POST /users`, `GET /users/{user_id}`: 사용자 등록/조회
- `GET /health`: 모델 로딩 상태 헬스체크

### 3. 데이터/모델 계층
- LightGBM 최종 위험도 예측 모델 (`training/train_lgbm.py`)
- PyTorch Autoencoder 이상탐지 모델 (`training/train_autoencoder.py`)
- PyTorch Transformer 인코더 (`training/train_transformer.py`)
- 외부 API 키 미설정 시 자동으로 목(mock) 데이터로 폴백 → 별도 키 없이도 전체 플로우 데모 가능

---

## 📊 데이터 소스

| 데이터 | 연동 상태 | 비고 |
|--------|----------|------|
| 기상청 동네예보/특보 API | ✅ 실제 연동 (키 미설정 시 mock) | `KMA_API_KEY` 환경변수 |
| 에어코리아 통합대기환경지수(CAI) API | ✅ 실제 연동 (키 미설정 시 mock) | `AIRKOREA_API_KEY` 환경변수 |
| 웨어러블 생체신호 (HR/HRV/체온/SpO₂) | 🧪 시뮬레이터로 대체 | 실제 기기 연동 없이 시나리오 기반 합성 데이터로 모델 검증 |
| 65세 이상 노인 다빈도 상병 데이터, 경로당 현황 등 | 📝 기획 단계 | V_personal 고도화를 위한 향후 확장 데이터로 계획 |

---

## 🛠 기술 스택

| 영역 | 사용 기술 |
|------|----------|
| **Backend** | Python 3.11, FastAPI, Mangum(ASGI→Lambda 어댑터), Pydantic v2 |
| **AI/ML** | LightGBM, PyTorch(Autoencoder, Transformer Encoder), scikit-learn, SHAP |
| **Frontend** | React 18, TypeScript, CRACO, Recharts, Axios |
| **인프라(IaC)** | AWS SAM / CloudFormation, Docker(Lambda 컨테이너 이미지) |
| **컴퓨팅/스토리지** | AWS Lambda(Container Image), API Gateway, DynamoDB, S3, CloudFront |
| **로컬 개발** | Docker Compose, DynamoDB Local, nvm(Node 18) |
| **외부 API** | 기상청 공공데이터포털, 에어코리아 |

---

## 📂 프로젝트 구조

```
aider/
├── deploy.ps1                  # AWS 배포 자동화 스크립트 (Windows PowerShell)
├── docker-compose.yml          # 로컬 개발용 (backend + DynamoDB Local + frontend)
├── infra/
│   └── template.yaml           # AWS SAM 템플릿 (Lambda/API GW/DynamoDB/S3/CloudFront)
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 진입점
│   │   ├── config.py           # 환경변수 기반 설정 (Pydantic Settings)
│   │   ├── api/routes/         # health, users, predictions, alerts 라우터
│   │   ├── models/             # lgbm_model, autoencoder, transformer_encoder
│   │   ├── services/           # KMA/에어코리아 API 클라이언트 (data_collector.py)
│   │   └── db/                 # DynamoDB 테이블 정의/CRUD
│   ├── training/                # 모델 학습 스크립트 (LightGBM/Autoencoder/Transformer)
│   ├── tests/                   # pytest 기반 API 테스트
│   ├── lambda_handler.py        # Lambda 진입점 (Mangum)
│   ├── Dockerfile.lambda        # Lambda 컨테이너 이미지 빌드용
│   └── Dockerfile                # 로컬/일반 컨테이너 실행용
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── components/          # RiskGauge, FeatureBar, WeatherCard, BioCard, AlertBanner
    │   ├── hooks/useSimulation.ts
    │   └── services/api.ts      # REACT_APP_API_URL 기반 axios 클라이언트
    └── craco.config.js
```

---

## ⚙️ 로컬 개발 환경 실행

```bash
# 1. 백엔드 + DynamoDB Local + 프론트엔드 동시 실행
cd aider
cp .env.example .env   # 필요 시 KMA/AirKorea 키 입력 (미입력 시 mock 데이터로 동작)
docker-compose up --build

# 접속
#   프론트엔드: http://localhost:3000
#   백엔드 API: http://localhost:8000/docs
#   DynamoDB Admin UI: http://localhost:8002
```

모델 아티팩트가 없다면 먼저 학습 스크립트를 실행합니다.

```bash
cd aider/backend
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -r requirements.txt
python training/train_lgbm.py
python training/train_autoencoder.py
python training/train_transformer.py
```

---

## ☁️ AWS 배포 방법

**사전 요구사항**: AWS CLI(자격증명 설정 완료), AWS SAM CLI, Docker Desktop, Node.js 18(nvm-windows 권장)

```powershell
# 학습된 모델 아티팩트가 backend/artifacts/ 에 있는지 먼저 확인 후
.\deploy.ps1 -KMAKey "<기상청 API 키, 없으면 생략>" -AirKey "<에어코리아 API 키, 없으면 생략>"
```

`deploy.ps1`이 자동으로 수행하는 작업:

1. AWS 계정/리전 확인, ECR 리포지토리 확인·생성
2. `Dockerfile.lambda` 기반 백엔드 이미지 빌드 → ECR 푸시
3. `sam deploy`로 CloudFormation 스택 생성/업데이트 (Lambda, API Gateway, DynamoDB, S3, CloudFront)
4. 배포 결과에서 API 엔드포인트·S3 버킷·CloudFront 정보를 조회
5. 조회한 API 주소를 프론트엔드 `.env.production`에 자동 반영 후 프로덕션 빌드
6. 빌드 산출물을 S3에 업로드하고 CloudFront 캐시 무효화

> KMA/AirKorea 키를 생략(또는 `mock`으로 입력)하면 백엔드가 자동으로 목(mock) 데이터 경로로 동작해, 실제 공공데이터 API 키 없이도 전체 데모를 시연할 수 있습니다.

---

## 🌍 기대 효과

- **환경적**: 기후 취약 지역 데이터 축적 → 지자체 기후 적응 정책 기반 자료
- **사회적**: 응급 출동·입원 감소, 돌봄 사각지대 해소, 복지 효율화
- **경제적**: 의료비 절감, 구독형 SaaS 모델 기반 지속가능 운영
- **확장성**: KOICA/GGGI 협력 통한 동남아·아프리카 기후취약 도시 확산 가능

---

## ⚠ 리스크 및 대응

| 리스크 | 대응 방안 |
|-------|----------|
| AI 예측 오차 | MLOps 기반 정기 재학습 및 검증 |
| 기기 보급 한계 | 지자체 협력 무상 대여 시범사업 |
| 고령층 디지털 접근성 | 음성 알림, 돌봄요원/가족 대리 입력 |
| 운영비 지속성 | 보험사·복지기관 연계 수익 다각화 |
| 개인정보 유출 | Edge AI 처리, 가명화·암호화, 접근권한 제한 |

---

## 🗺 향후 로드맵

- [x] CHAI 모델 프로토타입 (LightGBM + Autoencoder + Transformer)
- [x] 백엔드 API 및 AWS 서버리스 배포 파이프라인
- [x] 프론트엔드 시뮬레이션 대시보드
- [ ] 실제 웨어러블/IoT 기기 연동
- [ ] 돌봄요원 앱 (방문 우선순위 자동 배정)
- [ ] 관리자 대시보드 (행정동 단위 히트맵)
- [ ] 실사용자 데이터 기반 V_personal 메타러닝 고도화

---

## 👥 팀: 알파고 (알고 보면 파릇파릇한 고인물)

국민대학교 AI빅데이터융합경영학과 3학년

| 이름 | 역할 |
|------|------|
| 김태현 | 아이디어 제시 및 구체화 |
| 이용찬 | 데이터 탐색 및 분석 |
| 허지원 | 어플리케이션 구현 및 대시보드 시스템 |
| 윤성철 | 차별화 전략 수립 및 서비스 관리 |

---

## 📚 참고문헌

- [Lancet Regional Health – Western Pacific] Impacts of heat exposure on cardiovascular and respiratory outcomes (2025)
- [한국보건사회연구원] 폭염과 사망률의 상관관계 분석 연구
- [Frontiers in Public Health] Extreme heat exposure and health risks among vulnerable populations (2025)
- [Environmental Health Perspectives] Air pollution (PM2.5) and short-term cardiovascular/respiratory impacts (2023)
- [대한의사협회지(JKMA)] 대기오염과 건강 영향 종합 리뷰

---

## 📄 라이선스

본 프로젝트는 2025 국가기후위기적응센터 AI 아이디어 공모전 출품작을 기반으로 합니다.

---

> 🌱 *기후변화 시대의 새로운 사회 안전망, Aider*

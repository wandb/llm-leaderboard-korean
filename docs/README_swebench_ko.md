# SWE-bench 평가 가이드

SWE-bench는 실제 오픈소스 프로젝트의 버그 수정 능력을 평가하는 벤치마크입니다.  
모델이 생성한 unified diff 패치를 Docker 환경에서 적용하고, 테스트 통과 여부로 채점합니다.

---

## 목차

- [아키텍처](#아키텍처)
- [채점 서버](#채점-서버)
  - [요구 사항](#요구-사항)
  - [설치](#설치)
  - [서버 시작](#서버-시작)
  - [환경변수](#환경변수)
- [클라이언트 설정](#클라이언트-설정)
- [평가 실행](#평가-실행)
- [데이터셋 정보](#데이터셋-정보)
- [평가 흐름](#평가-흐름)
- [출력 형식](#출력-형식-중요)
- [서버 API](#서버-api-엔드포인트)
- [트러블슈팅](#트러블슈팅)
- [부록: AWS EC2로 서버 운영](#부록-aws-ec2로-서버-운영)
- [참고 자료](#참고-자료)

---

## 아키텍처

```
┌─────────────────────────────┐
│ horangi (클라이언트)        │
│  - 문제 설명 → LLM 호출     │
│  - unified diff 생성        │
└──────────────┬──────────────┘
               │ HTTP API
               ▼
┌─────────────────────────────┐
│ SWE-bench Server            │
│  - Docker 환경에서 패치 적용 │
│  - 테스트 실행 및 채점       │
└─────────────────────────────┘
```

---

## 채점 서버

Docker가 설치된 Linux 머신이면 어디서든 채점 서버를 띄울 수 있습니다.  
자체 서버가 없는 경우 [부록: AWS EC2로 서버 운영](#부록-aws-ec2로-서버-운영)을 참고하세요.

### 요구 사항

- Linux (Docker 필요 — macOS Docker Desktop은 미지원)
- Python 3.12+
- 최소 4 vCPU / 16GB RAM / 100GB 디스크 권장

### 설치

```bash
# 방법 1: pip 직접 설치
pip install fastapi "uvicorn[standard]" swebench

# 방법 2: 프로젝트 optional dependency
uv add fastapi "uvicorn[standard]" swebench --optional swebench-server
```

### 서버 시작

```bash
# 포어그라운드 실행
uv run python src/server/swebench_server.py --host 0.0.0.0 --port 8000

# 백그라운드 실행
nohup python src/server/swebench_server.py \
  --host 0.0.0.0 --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown

# 헬스체크
curl http://localhost:8000/health
```

### 환경변수

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `SWE_API_KEY` | (없음) | API 인증 키 (선택사항) |
| `SWE_MAX_JOBS` | `4` | 동시 실행 최대 작업 수 |
| `SWE_JOB_TIMEOUT` | `1800` | 작업 타임아웃 (초, 30분) |
| `SWE_PREBUILD_IMAGES` | `true` | Docker 이미지 사전 빌드 |
| `SWE_WORKERS` | `4` | 동시 평가 워커 수 |

---

## 클라이언트 설정

클라이언트(horangi)는 macOS 등 Docker 없는 환경에서도 실행 가능합니다.  
채점 서버 URL만 알면 됩니다.

### 환경변수 설정

```bash
export SWE_SERVER_URL=http://YOUR_SERVER:8000

# (선택) API 키가 있는 경우
export SWE_API_KEY=your-api-key
```

### 설정 파일 (`configs/base_config.yaml`)

```yaml
benchmarks:
  swebench:
    server_url: http://YOUR_SERVER:8000
    timeout: 1800  # 30분
```

---

## 평가 실행

```bash
# 단일 실행 (테스트)
uv run horangi swebench_verified_official_80 --config gpt-4o -T limit=5

# 전체 평가 (80개 샘플)
uv run horangi swebench_verified_official_80 --config gpt-4o

# run_eval.py를 통한 실행
uv run python run_eval.py --config <model_config> --only swebench_verified_official_80

# resume으로 swebench만 재실행
uv run python resume_swebench.py --workers 4
```

---

## 데이터셋 정보

| 이름 | 설명 | 샘플 수 | 입력 토큰 |
|-----|------|--------:|----------|
| `swebench_verified_official_80` | 검증된 80개 인스턴스 | 80 | < 7,000 |

> **참고**: 원본 SWE-bench Verified (500개)에서 입력 토큰 7,000 미만으로 필터링하고, 난이도 분포를 유지하며 샘플링한 서브셋입니다.

### 난이도 분포

| 난이도 | 원본 (500) | 80개 서브셋 |
|-------|----------:|------------:|
| < 15분 | 38.8% | 46.2% |
| 15분 ~ 1시간 | 52.2% | 50.0% |
| 1~4시간 | 8.4% | 3.8% |
| > 4시간 | 0.6% | 0.0% |

---

## 평가 흐름

1. **입력**: 문제 설명(Issue), 힌트, 관련 코드
2. **생성**: 모델이 unified diff 패치 생성
3. **적용**: 서버에서 Docker 환경에 패치 적용
4. **채점**: 테스트 실행 후 Pass/Fail 판정

### 상세 흐름

```
┌────────────────────────────┐
│ Dataset (80 items)         │
└───────────────┬────────────┘
                │ Input (Issue/PR context, relevant snippets,
                │         reproduction/expected tests, constraints)
                ▼
┌────────────────────────────────────────┐
│ Generation (LLM)                       │
│  - Prompt shaping (CRITICAL sentence)  │
│  - unified diff 생성                   │
└───────────────────┬────────────────────┘
                    │ Unified diff (line numbers in @@ required)
                    ▼
┌────────────────────────────────────────┐
│ Preprocessing (expansion & normalization) │
│  - Extract minimal patch                │
│  - Hunk header expansion                │
│  - Filename normalization / merge dups  │
└───────────────────┬────────────────────┘
                    │ Apply patch (git apply / patch --fuzz)
                    ▼
┌────────────────────────────────────────┐
│ Evaluation runner                      │
│  - Docker 환경에서 실행                 │
│  - Unit tests 실행                     │
└───────────────────┬────────────────────┘
                    │ Pass/Fail
                    ▼
            Resolved / Not Resolved
                    ↓
             SWE-Bench Score
           (Resolved rate = pass rate)
```

---

## 출력 형식 (중요!)

모델은 반드시 **올바른 hunk header**를 포함한 unified diff를 생성해야 합니다:

```diff
--- a/file.py
+++ b/file.py
@@ -10,5 +10,7 @@
 def function():
-    old_code()
+    new_code()
+    additional_fix()
```

### CRITICAL

- **라인 번호 필수**: `@@ -start,count +start,count @@` 형식의 hunk header가 반드시 필요합니다.
- 라인 번호 없이 `@@ @@`만 사용하면 패치 적용이 실패합니다.

### 올바른 예시

```diff
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -245,1 +245,1 @@
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

---

## 서버 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 헬스 체크 |
| `/v1/jobs` | POST | 평가 작업 생성 |
| `/v1/jobs/{job_id}` | GET | 작업 상태 조회 |
| `/v1/jobs/{job_id}/logs` | GET | 작업 로그 조회 |
| `/v1/jobs/{job_id}/report` | GET | 평가 결과 조회 |
| `/v1/summary` | GET | 전체 큐 상태 요약 |

### 작업 생성 예시

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "astropy__astropy-12907",
    "patch_diff": "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new",
    "model_name_or_path": "gpt-4o"
  }'
```

### 응답 예시

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### 작업 상태 조회

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "finished",
  "instance_id": "astropy__astropy-12907",
  "created_at": 1702800000.0,
  "finished_at": 1702800300.0
}
```

### 결과 조회

```bash
curl http://localhost:8000/v1/jobs/{job_id}/report
```

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "instance_id": "astropy__astropy-12907",
  "resolved_ids": ["astropy__astropy-12907"],
  "unresolved_ids": [],
  "error_ids": []
}
```

---

## 트러블슈팅

### 패치 적용 실패

- hunk header에 라인 번호가 없는지 확인하세요.
- `git apply`가 실패하면 서버가 자동으로 `patch --fuzz=10/20`을 시도합니다.

### 타임아웃

- 기본 타임아웃은 30분(1800초)입니다.
- 복잡한 테스트는 더 오래 걸릴 수 있으니 `SWE_JOB_TIMEOUT` 환경변수로 조정하세요.

### 서버 프로세스가 안 뜰 때

```bash
# 로그 확인
cat /tmp/swebench_server.out

# 포트 사용 중인 프로세스 확인/제거
sudo lsof -t -i:8000 | xargs sudo kill -9
```

### Docker 이미지 디스크 부족

```bash
sudo docker system prune -af
```

---

## 부록: AWS EC2로 서버 운영

Docker가 돌아가는 Linux 서버가 없으면 AWS EC2를 빌려 쓸 수 있습니다.

### 기존 인스턴스 사용

프로젝트 전용 EC2 인스턴스가 이미 세팅되어 있습니다.

| 항목 | 값 |
|---|---|
| Instance ID | `i-00796b3844c7f4a25` |
| 타입 | m5.xlarge (4 vCPU, 16GB RAM, 100GB gp3) |
| 리전 | ap-northeast-2 (서울) |
| 보안그룹 | `sg-08535256fcd8b3e7a` (SSH 22, API 8000 개방) |
| 키페어 | `swebench-server` (`~/.ssh/swebench-server.pem`) |

```bash
# 인스턴스 시작
aws ec2 start-instances --instance-ids i-00796b3844c7f4a25 --region ap-northeast-2

# Public IP 확인 (시작할 때마다 바뀜)
aws ec2 describe-instances --instance-ids i-00796b3844c7f4a25 --region ap-northeast-2 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH로 채점 서버 프로세스 시작
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP> \
  'sudo bash -c "export PATH=/root/.local/bin:\$PATH; cd /opt/llm-leaderboard-korean; export SWE_WORKERS=4; nohup uv run python src/server/swebench_server.py --port 8000 >/tmp/swebench_server.out 2>&1 &"'

# 헬스체크
curl http://<PUBLIC_IP>:8000/health

# 사용 끝나면 인스턴스 중지
aws ec2 stop-instances --instance-ids i-00796b3844c7f4a25 --region ap-northeast-2
```

### 새 인스턴스 만들기

#### 1. EC2 인스턴스 생성

```bash
# 키페어 생성
aws ec2 create-key-pair --key-name swebench-server \
  --query 'KeyMaterial' --output text > ~/.ssh/swebench-server.pem
chmod 600 ~/.ssh/swebench-server.pem

# 보안그룹 생성
VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?IsDefault].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group \
  --group-name swebench-sg --description "SWE-bench server" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Ubuntu 22.04 AMI 찾기
AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

# 인스턴스 생성
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type m5.xlarge \
  --key-name swebench-server \
  --security-group-ids $SG_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=swebench-server}]' \
  --count 1
```

#### 2. 서버 환경 구축

```bash
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP>

sudo apt-get update -y
sudo apt-get install -y docker.io python3-pip python3-venv git curl
sudo systemctl enable docker && sudo systemctl start docker

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

cd /opt
sudo git clone https://github.com/wandb/llm-leaderboard-korean.git
cd llm-leaderboard-korean
sudo uv python install 3.12
sudo uv sync --python 3.12
sudo uv pip install uvicorn swebench
```

#### 3. 채점 서버 시작

```bash
sudo bash -c '
export PATH=/root/.local/bin:$PATH
cd /opt/llm-leaderboard-korean
export SWE_WORKERS=4
nohup uv run python src/server/swebench_server.py --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
'
```

### EC2 비용

| 상태 | 비용 |
|---|---|
| Running (m5.xlarge) | ~$0.19/hr (~$136/월) |
| Stopped | ~$8/월 (100GB EBS만) |

사용하지 않을 때는 반드시 Stop 해두세요.

### Public IP 고정 (Elastic IP)

Stop/Start 하면 IP가 바뀝니다. 고정하려면:

```bash
ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
aws ec2 associate-address --instance-id i-00796b3844c7f4a25 --allocation-id $ALLOC_ID
```

---

## 참고 자료

- [SWE-bench 공식 레포](https://github.com/princeton-nlp/SWE-bench)
- [Nejumi LLM Leaderboard SWE-bench 가이드](https://github.com/wandb/llm-leaderboard/blob/main/docs/README_swebench.md)

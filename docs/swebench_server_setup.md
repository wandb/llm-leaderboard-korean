# SWE-bench 평가 서버 셋업 가이드

## 개요

SWE-bench 평가는 Docker 기반 서버가 필요합니다. 모델이 생성한 패치를 서버에 제출하면, 서버가 Docker 컨테이너에서 공식 harness를 실행하고 테스트 통과 여부를 반환합니다.

## 현재 인프라

- **Instance ID**: `i-00796b3844c7f4a25`
- **타입**: m5.xlarge (4 vCPU, 16GB RAM, 100GB gp3)
- **리전**: ap-northeast-2 (서울)
- **보안그룹**: `sg-08535256fcd8b3e7a` (SSH 22, API 8000 개방)
- **키페어**: `swebench-server` (`~/.ssh/swebench-server.pem`)

## 기존 서버 시작/중지

```bash
# 서버 시작
aws ec2 start-instances --instance-ids i-00796b3844c7f4a25

# Public IP 확인 (시작 후 변경됨)
aws ec2 describe-instances --instance-ids i-00796b3844c7f4a25 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH 접속
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP>

# swebench 서버 프로세스 시작
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP> \
  'sudo bash -c "export PATH=/root/.local/bin:\$PATH; cd /opt/llm-leaderboard-korean; export SWE_WORKERS=4; nohup uv run python src/server/swebench_server.py --port 8000 >/tmp/swebench_server.out 2>&1 &"'

# 헬스체크
curl http://<PUBLIC_IP>:8000/health

# 서버 중지 (사용 안 할 때 — EBS 비용만 발생)
aws ec2 stop-instances --instance-ids i-00796b3844c7f4a25
```

## 평가 실행

```bash
# 환경변수 설정 후 평가 실행
SWE_SERVER_URL=http://<PUBLIC_IP>:8000 uv run python run_eval.py \
  --config <model_config> --only swebench_verified_official_80

# resume으로 swebench만 재실행
SWE_SERVER_URL=http://<PUBLIC_IP>:8000 uv run python resume_swebench.py --workers 4
```

## 새 서버 처음부터 만들기

### 1. EC2 인스턴스 생성

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

### 2. 서버 환경 구축

```bash
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP>

# 시스템 패키지
sudo apt-get update -y
sudo apt-get install -y docker.io python3-pip python3-venv git curl
sudo systemctl enable docker && sudo systemctl start docker

# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# 레포 클론 및 의존성 설치
cd /opt
sudo git clone https://github.com/wandb/llm-leaderboard-korean.git
cd llm-leaderboard-korean
sudo uv python install 3.12
sudo uv sync --python 3.12
sudo uv pip install uvicorn swebench
```

### 3. 서버 시작

```bash
sudo bash -c '
export PATH=/root/.local/bin:$PATH
cd /opt/llm-leaderboard-korean
export SWE_WORKERS=4  # 동시 평가 워커 수
nohup uv run python src/server/swebench_server.py --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
'
```

## 서버 API

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/health` | GET | 헬스체크 |
| `/v1/jobs` | POST | 평가 job 제출 |
| `/v1/jobs/{job_id}` | GET | job 상태 조회 |
| `/v1/jobs/{job_id}/logs` | GET | job 로그 조회 |
| `/v1/jobs/{job_id}/report` | GET | 평가 리포트 (resolved/unresolved) |
| `/v1/summary` | GET | 전체 큐 상태 요약 |

## 비용

| 상태 | 비용 |
|---|---|
| Running (m5.xlarge) | ~$0.19/hr (~$136/월) |
| Stopped | ~$8/월 (100GB EBS만) |

사용하지 않을 때는 반드시 Stop 해두세요.

## 트러블슈팅

### 서버 프로세스가 안 뜰 때
```bash
# 로그 확인
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP> 'sudo cat /tmp/swebench_server.out'

# 포트 사용 중인 프로세스 확인/제거
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP> 'sudo lsof -t -i:8000 | xargs sudo kill -9'
```

### Docker 이미지 디스크 부족
```bash
ssh -i ~/.ssh/swebench-server.pem ubuntu@<PUBLIC_IP> 'sudo docker system prune -af'
```

### Public IP 변경 시
Stop/Start 하면 IP가 바뀝니다. Elastic IP를 할당하면 고정 가능:
```bash
ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
aws ec2 associate-address --instance-id i-00796b3844c7f4a25 --allocation-id $ALLOC_ID
```

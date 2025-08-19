# PyPI 배포 체크리스트

## ✅ 완료된 항목들

### 1. 기본 설정 파일들
- [x] `pyproject.toml` - 패키지 메타데이터 및 의존성 설정
- [x] `setup.py` - 호환성을 위한 최소 설정
- [x] `MANIFEST.in` - 추가 파일 포함 설정
- [x] `LICENSE` - Apache-2.0 라이선스
- [x] `README.md` - 상세한 프로젝트 설명

### 2. 패키지 구조
- [x] `llm_eval/` - 메인 패키지 디렉토리
- [x] `llm_eval/__init__.py` - 버전 정보 및 주요 exports
- [x] 모든 하위 모듈에 `__init__.py` 파일 존재

### 3. 메타데이터
- [x] 패키지 이름: `haerae-evaluation-toolkit`
- [x] 버전: `0.1.0` (pyproject.toml과 __init__.py 일치)
- [x] 작성자 정보
- [x] 프로젝트 설명
- [x] 키워드 및 분류자
- [x] 프로젝트 URL들 (Homepage, Repository, Issues, Documentation)

### 4. 의존성 관리
- [x] 모든 의존성에 최소 버전 명시
- [x] Python 버전 요구사항: `>=3.10`
- [x] 개발 의존성 분리 (`dev` optional-dependencies)

### 5. 라이선스
- [x] SPDX 형식으로 라이선스 명시: `Apache-2.0`
- [x] 라이선스 분류자 제거 (최신 표준 준수)

### 6. 빌드 테스트
- [x] Wheel 빌드 성공
- [x] Source distribution 빌드 성공
- [x] 패키지 내용 검증

### 7. 문서화
- [x] `DEPLOYMENT_GUIDE.md` - 배포 가이드
- [x] `PYPI_CHECKLIST.md` - 이 체크리스트

## 🔄 배포 전 확인사항

### 1. 최종 테스트
```bash
# 빌드 테스트
python -m build --wheel --sdist

# 패키지 내용 확인
python -m tarfile -l dist/haerae_evaluation_toolkit-*.tar.gz

# 기본 import 테스트 (가상환경에서)
pip install dist/haerae_evaluation_toolkit-*.whl
python -c "import llm_eval; print(llm_eval.__version__)"
```

### 2. PyPI 업로드
```bash
# TestPyPI에 먼저 업로드 (선택사항)
python -m twine upload --repository testpypi dist/*

# 실제 PyPI에 업로드
python -m twine upload dist/*
```

## 📋 수정된 파일 목록

1. **pyproject.toml**
   - 의존성 버전 명시
   - 라이선스 현대화 (`Apache-2.0`)
   - 프로젝트 URL 추가
   - 분류자 개선

2. **llm_eval/__init__.py**
   - 버전을 0.1.0으로 통일

3. **README.md**
   - 로고 이미지 경로 수정 (`logo.png.png` → `logo.png`)

4. **requirements.txt**
   - pyproject.toml과 일치하도록 의존성 버전 업데이트

5. **새로 생성된 파일들**
   - `MANIFEST.in` - 패키지에 포함할 파일 명시
   - `setup.py` - 호환성을 위한 최소 설정
   - `DEPLOYMENT_GUIDE.md` - 배포 가이드
   - `PYPI_CHECKLIST.md` - 이 체크리스트
   - `.pypirc` - PyPI 설정 템플릿

## 🚀 다음 단계

1. PyPI 계정 생성 및 API 토큰 발급
2. `twine` 설치: `pip install twine`
3. 테스트 업로드 (선택사항)
4. 실제 PyPI 업로드
5. 설치 테스트: `pip install haerae-evaluation-toolkit`

## ⚠️ 주의사항

- 한 번 업로드된 버전은 삭제할 수 없습니다
- 버전 번호를 신중하게 관리하세요
- 업로드 전에 반드시 테스트를 수행하세요
- API 토큰을 안전하게 보관하세요
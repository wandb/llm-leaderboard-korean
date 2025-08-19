# PyPI 배포 가이드

이 문서는 haerae-evaluation-toolkit을 PyPI에 배포하는 방법을 설명합니다.

## 사전 준비

1. **PyPI 계정 생성**
   - https://pypi.org 에서 계정을 생성하세요
   - API 토큰을 생성하세요 (Account settings > API tokens)

2. **필요한 도구 설치**
   ```bash
   pip install build twine
   ```

## 배포 과정

### 1. 버전 업데이트
배포 전에 다음 파일들의 버전을 업데이트하세요:
- `pyproject.toml`의 `version` 필드
- `llm_eval/__init__.py`의 `__version__` 변수

### 2. 패키지 빌드
```bash
# 이전 빌드 파일 정리
rm -rf dist/ build/ *.egg-info

# 새로운 패키지 빌드
python -m build --wheel --sdist
```

### 3. 빌드 결과 확인
```bash
# 생성된 파일 확인
ls -la dist/

# 패키지 내용 확인
python -m tarfile -l dist/haerae_evaluation_toolkit-*.tar.gz
```

### 4. 테스트 업로드 (선택사항)
```bash
# TestPyPI에 먼저 업로드하여 테스트
python -m twine upload --repository testpypi dist/*
```

### 5. 실제 PyPI 업로드
```bash
# 실제 PyPI에 업로드
python -m twine upload dist/*
```

## 업로드 후 확인

1. **PyPI 페이지 확인**
   - https://pypi.org/project/haerae-evaluation-toolkit/ 에서 패키지가 정상적으로 업로드되었는지 확인

2. **설치 테스트**
   ```bash
   # 새로운 환경에서 설치 테스트
   pip install haerae-evaluation-toolkit
   ```

3. **기본 import 테스트**
   ```python
   import llm_eval
   print(llm_eval.__version__)
   ```

## 주의사항

- **버전 관리**: 한 번 업로드된 버전은 삭제할 수 없으므로 신중하게 버전을 관리하세요
- **의존성 확인**: 모든 의존성이 올바르게 설정되어 있는지 확인하세요
- **라이선스**: Apache-2.0 라이선스가 올바르게 설정되어 있는지 확인하세요
- **README**: PyPI 페이지에 표시될 README.md가 올바르게 작성되어 있는지 확인하세요

## 문제 해결

### 일반적인 오류들

1. **403 Forbidden**: API 토큰이 올바르지 않거나 권한이 없는 경우
2. **400 Bad Request**: 패키지 메타데이터에 문제가 있는 경우
3. **409 Conflict**: 이미 존재하는 버전을 업로드하려는 경우

### 도움말

- PyPI 공식 문서: https://packaging.python.org/
- Twine 문서: https://twine.readthedocs.io/
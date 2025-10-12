## HalluLens

### 1\. 주요 기능 (What I did)

  - **halluLens 벤치마크**를 활용하여 한국어 모델의 \*\*Hallucination(환각 현상)\*\*을 평가하는 기능을 추가합니다.
      - 기존 halluLens의 결과물을 번역하고 프롬프트를 한국어 버전으로 생성하여 한국어 모델에 대한 정밀한 평가를 진행합니다.
      - 평가 방식이 다양하고 독립적이라서 `llm_eval/external/hallulens` 경로에 별도로 구현했습니다.
      - 자세한 사용 예시는 `test_hallulens_benchmark.py`를 참고해 주세요.

### 2\. 주요 평가 항목 (Key Evaluation Metrics)

1.  **`precise_wikiqa`**: 단답형 질문에 대한 사실적 정확성
2.  **`longwiki_qa`**: 서술형 질문에 대한 답변의 일관성 및 정확성
3.  **`Non_entity_refusal`**: 존재하지 않는 개체(entity)에 대한 질문 시, 답변을 거부하는 능력

### 3\. 실행 방법 (How to Run)

  - 모든 코드의 실행 시작점은 `llm_eval/external/hallulens/tasks/hallulens_runner.py` 입니다.

### 4\. 중요 공지 (Important Notice)

#### 4.1. 데이터 준비 (Getting ready with data)

  - **(한국어) 데이터 다운로드**

      - **⭐️ 중요\!\!**: `donwload.sh`로 데이터 다운로드 시 `.db` 파일이 정상적으로 받아지지 않을 수 있습니다.
      - `script/download_data.sh` 스크립트로 다운로드를 시도하거나, **[이 링크](https://www.google.com/search?q=%EC%88%98%EB%8F%99_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C_%EB%A7%81%ED%81%AC_%EC%82%BD%EC%9E%85)** 에서 직접 다운로드해 주세요.
      - 다운로드한 파일은 반드시 다음 경로에 저장해야 합니다.
          - **경로**: `llm_eval/external/providers/hallulens/data/wiki_data/.cache/enwiki-20230401.db`

  - **Data Download**

      - We provide a script to download all data needed for the three tasks. This will download all necessary data into the `/data` folder.
      - **⚠️Notice**: The Wikipedia dump is large (\~16GB), so please ensure you have enough space. The download may fail via the script.
        ```bash
        bash scripts/download_data.sh
        ```
      - This script includes:
          - [Wikirank](https://wikirank-2024.di.unimi.it/)
          - [GoodWiki](https://huggingface.co/datasets/euirim/goodwiki)
          - Processed Wikipedia dump (from [FactScore](https://arxiv.org/abs/2305.14251))
          - [ITIS taxonomy](https://www.itis.gov/)
          - [250k Medicines Usage, Side Effects and Substitutes](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes)

#### 4.2. 커스터마이징 및 설정 변경 (Customization & Configuration)

  - **VLLM 사용 및 모델 변경**:
      - `inference_method` 파라미터를 `'vllm'`으로 변경하고, `model`에 허깅페이스 모델명을 입력하세요.
  - **LLM as Judge 방식 변경 (VLLM, Custom 등)**:
      - 코드 내 `call_together_api` 함수를 `call_vllm_api` 또는 `custom_api` 함수로 hallulens 파일에서 전체 변경해야 합니다.
  - **새로운 LLM 호스팅 방식 추가**:
      - `llm_eval/external/hallulens/utils/lm.py` 파일의 `custom_api`와 `generate` 함수를 수정하여 구현할 수 있습니다.

#### 4.3. 트러블슈팅 (Troubleshooting)

  - **Together.ai Rate Limit**: `together.ai` 호스팅 사용 시 API 요청 제한(Rate Limit)이 발생하여 속도를 낮췄습니다. `Max_worker` 파라미터를 높이거나 지연 시간을 줄이면 Rate Limit이 발생할 수 있습니다.
  - **성능 낮은 모델의 평가 불가**: 성능이 낮은 모델은 평가 가능한 답변 형식(올바른 Json 형태)을 생성하지 못해 `longwiki_qa` 또는 `precise_wikiqa` 평가가 실패할 수 있습니다.
  - **`precise_wikiqa` Abstain 문제**: `precise_wikiqa` 태스크에서 모델 추론 실패나 `abstain` 문제가 반복된다면, 불완전하게 생성된 `output` 폴더의 대상 모델 결과물(.jsonl 파일)을 삭제 후 다시 시도해 주세요. 이전의 잘못된 결과물을 계속 참조하여 문제가 발생할 수 있습니다.

### 5\. 라이선스 (License)

The majority of HalluLens is licensed under CC-BY-NC. However, portions of the project are available under separate license terms:

  - [FActScore](https://github.com/shmsw25/FActScore) is licensed under the MIT license.
  - VeriScore is licensed under the Apache 2.0 license.
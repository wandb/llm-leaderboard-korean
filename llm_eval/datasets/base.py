from typing import List, Dict, Any, Optional

class BaseDataset:
    """
    (개발 용이를 위해 개발 단계에서는 한국어로 작성, 이후 영어로 대체)
    모든 데이터셋 클래스가 상속해야 할 기본(추상) 클래스
    목적:
      1) 평가 파이프라인에서 기대하는 일관된 인터페이스(특히 'input'과 'reference') 제공
        1-1) input: LLM에게 전달되는 prompt, context, instruction 등을 모두 합쳐 "모델이 실제로 받게 될 텍스트"
        1-2) reference: 정답, 정답 label, gold output 등에 해당하는 "모델이 만들어야 하는 기대 출력"
      2) 각 데이터셋별 로드/전처리 로직을 쉽게 커스터마이징징

    필수 구현 메서드
      - load(): 데이터를 로드하고 최종적으로 [{"input":..., "reference":...}, ...] 형태 반환.
    
    선택적 구현 메서드
      - get_raw_samples(): 원본(raw) 데이터 접근
      - info(): 데이터셋 정보
    """

    def __init__(self, dataset_name: str, split: str = "test", subset:str = None, **kwargs):
        """
        Args:
            dataset_name (str): 
                데이터셋 고유 식별자
            split (str): 
                train/validation/test 등을 구분하기 위한 문자열 (load 시)
            subset (str): 
                하위 태스크나 config (예: "abstract_algebra")
            kwargs: 
                기타 데이터셋 로딩에 필요한 파라미터(예: HF config, version, 인증 토큰 등)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.kwargs = kwargs  # 확장 가능하도록 저장

    def load(self) -> List[Dict[str, Any]]:
        """
        (필수) 전제 pipeline 에서 사용할 데이터 List 반환.
        각 element는 {"input": str, "reference": str, (optional) ...} 형태.
        """
        raise NotImplementedError("Subclasses must implement load().")

    def get_raw_samples(self) -> Any:
        """
        (선택) 원본 데이터(raw)를 반환하거나, 
        필요하다면 객체 형태로 caching하여 접근할 수 있도록 제공.
        """
        raise NotImplementedError("This is optional. Override if needed.")

    def info(self) -> Dict[str, Any]:
        """
        (선택) 데이터셋에 대한 메타 정보(고민중..)를 딕셔너리로 반환.
        """
        return {"name": self.dataset_name, "split": self.split}

import logging
import sys
from typing import Optional

def get_logger(
    name: str = "llm_eval",
    level: int = logging.INFO,
    log_to_stdout: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    간단한 로거 설정 함수.

    Args:
        name (str): 로거 이름 (보통 패키지명이나 모듈명)
        level (int): 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_stdout (bool): True면 stdout으로 출력, False면 핸들러 추가 안 함
        log_format (str): 커스텀 로그 포맷 지정. None이면 기본 포맷 사용.

    Returns:
        logging.Logger: 설정이 완료된 로거 객체
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 이미 핸들러가 있다면(이중 로그 방지 위해) 초기화
    if logger.hasHandlers():
        logger.handlers.clear()

    if not log_format:
        log_format = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # STDOUT 핸들러
    if log_to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 파일 로깅이 필요할진 모르겠으나, 혹시 모르니 일단 두기:
    # file_handler = logging.FileHandler("llm_eval.log")
    # file_handler.setLevel(level)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger

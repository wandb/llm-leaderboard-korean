# What code was I trying to put in here? 
# probably.. THE LANGUAGE PENALIZER

from langdetect import detect

def language_penalizer(text: str, target_lang: str = 'ko') -> float:
    """
    Returns 1.0 if the detected language of the input text matches the target language,
    otherwise returns 0.0. In cases where language detection fails due to text length or
    encoding issues, the function returns 0.0.
    
    Parameters:
    - text (str): The text whose language is to be detected.
    - target_lang (str): The target language code to compare against (default is 'ko' for Korean).
    
    Returns:
    - float: 1.0 if the detected language matches the target language; otherwise, 0.0.
    """
    try:
        return 1.0 if detect(text) == target_lang else 0.0
    except:
        return 0.0

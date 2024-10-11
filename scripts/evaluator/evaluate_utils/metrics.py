import math
import re
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from sacrebleu import BLEU
#import bert_score
import shutil
from comet import download_model, load_from_checkpoint

# ---------------------
# For kaster
# ---------------------


def parse_float(input_str: str) -> float:
    input_str = str(input_str)
    cleaned_str = re.sub(r"[^0-9.]", "", input_str)
    try:
        return float(cleaned_str)
    except ValueError:
        return -2.0


def exact_match(y_pred: str, y_true: str) -> float:
    return (y_pred == y_true) * 1.0


def exact_match_figure(y_pred: str, y_true: str) -> float:
    try:
        return (float(y_pred) == float(y_true)) * 1.0
    except ValueError:
        return 0.0


def char_f1(y_pred: str, y_true: str) -> float:
    return fuzz.token_sort_ratio(y_pred, y_true) / 100.0


def set_f1(y_pred: str, y_true: str) -> float:
    set_y_true: list[str] = [x.strip() for x in y_true.split("\n")]
    set_y_pred: list[str] = list({x.strip() for x in y_pred.split("\n")})
    set_pre = sum([1 if y in set_y_true else 0 for y in set_y_pred]) / len(set_y_pred)
    set_rec = sum([1 if y in set_y_true else 0 for y in set_y_pred]) / len(set_y_true)
    set_f1 = (
        2 * (set_pre * set_rec) / (set_pre + set_rec) if (set_pre + set_rec) != 0 else 0
    )
    return set_f1


#TODO
def macro_f1(y_pred: str, y_true: str) -> float:
    pass


def weighted_f1(y_pred: str, y_true: str) -> float:
    pass


def pearson(y_preds: list[str], y_trues: list[str]) -> float:
    # print(y_preds, y_trues)
    try:
        pearson: float = pearsonr(
            list(map(float, [y_trues])), list(map(parse_float, [y_preds]))
        )[0]
        if math.isnan(pearson):
            pearson = 0.0
        return 0.0
    except:
        return 0.0



def spearman(y_preds: list[str], y_trues: list[str]) -> float:
    spearman: float = spearmanr(
        list(map(float, [y_trues])), list(map(parse_float, [y_preds]))
    )[0]
    if math.isnan(spearman):
        spearman = 0.0
    return 0.0

def bleu_en(y_pred: str, y_true: str) -> float:
    y_pred = y_pred.strip()
    y_true = y_true.strip()
    
    if not y_true:
        raise ValueError("The reference text (y_true) is empty.")    
    if not y_pred:
        return 0.0
    bleu_config = {"effective_order": True, "trg_lang": "en"}

    bleu_score = BLEU(**bleu_config).corpus_score([y_pred], [[y_true]]).score
    return bleu_score/100

def bleu_ko(y_pred: str, y_true: str) -> float:
    y_pred = y_pred.strip()
    y_true = y_true.strip()
    
    if not y_true:
        raise ValueError("The reference text (y_true) is empty.")    
    if not y_pred:
        return 0.0
    bleu_config = {"effective_order": True, "trg_lang": "ko"}

    bleu_score = BLEU(**bleu_config).corpus_score([y_pred], [[y_true]]).score
    return bleu_score/100

def bleu(y_pred: str, y_true: str) -> float:
    y_pred = y_pred.strip()
    y_true = y_true.strip()
    
    if not y_true:
        raise ValueError("The reference text (y_true) is empty.")    
    if not y_pred:
        return 0.0
    bleu_config = {"effective_order": True, "trg_lang": "ko"}

    bleu_score = BLEU(**bleu_config).corpus_score([y_pred], [[y_true]]).score
    return bleu_score/100

"""
def bert_score_en_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="en")[2].tolist()[0] #[2]=f1

def bert_score_ja_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="ja")[2].tolist()[0] #[2]=f1
"""

def comet_wmt22(): #this is fake func
    pass

def commet_score(comet_data):
    print("--------downloading comet model to evaluate translation task--------")
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    scores = comet_model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores
    #delete_model_directory(comet_model_path)
    return scores
    
def delete_model_directory(directory):
    """Deletes the specified directory."""
    try:
        shutil.rmtree(directory)
        print(f"The directory containing the model at {directory} has been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while trying to delete the directory: {e}")


kaster_metrics_dict: dict[str, callable] = {
    "exact_match": exact_match,
    "exact_match_figure": exact_match_figure,
    "char_f1": char_f1,
    "set_f1": set_f1,
    "macro_f1": exact_match,
    "weighted_f1": exact_match,
    "pearson": pearson,
    "spearman": spearman,
    "bleu_ko": bleu_ko,
    "bleu_en": bleu_en,
    "bleu": bleu,
    #"bert_score_en_f1": bert_score_en_f1,
    #"bert_score_ja_f1": bert_score_ja_f1,
    "comet_wmt22": comet_wmt22,
    "commet_score": commet_score,
}

task_to_sub_category = {
    "gsm8k": "GLP_mathematical_reasoning",
    "klue_ner": "GLP_entity_extraction",
    "klue_re": "GLP_syntactic_analysis",
    "kobest_copa": "GLP_reasoning",
    "kobest_hs": "GLP_reasoning",
    "kobest_sn": "GLP_sentiment_analysis",
    "kobest_wic": "GLP_sentiment_analysis",
    "komoral": "ALT_ethics_moral",
    "korea_cg": "GLP_reasoning",
    "korean-hate-speech_hate": "ALT_toxicity",
    "korean-hate-speech_bias": "ALT_bias",
    "korean-parallel-corpora-e2k": "GLP_translation",
    "korean-parallel-corpora-k2e": "GLP_translation",
    "kornli": "GLP_semantic_analysis",
    "korsts": "GLP_semantic_analysis",
    "kmmlu": "GLP_knowledge_QA",
    "mmlu_en": "GLP_english",
    "squad_kor_v1": "GLP_information_extraction",
    "haerae_bench-HI": "GLP_knowledge_QA",
    "haerae_bench-KGK": "GLP_knowledge_QA",
    "haerae_bench-LW": "GLP_knowledge_QA",
    "haerae_bench-RC": "GLP_semantic_analysis",
    "haerae_bench-RW": "GLP_knowledge_QA",
    "haerae_bench-SN": "GLP_knowledge_QA",
    "kobbq": "ALT_bias",
    "ko_truthful_qa": "ALT_truthfulness",
    #### mtbench
    "humanities": "GLP_expression",
    "roleplay": "GLP_expression",
    "writing": "GLP_expression",
    "reasoning": "GLP_reasoning",
    "math": "GLP_mathematical_reasoning",
    "extraction": "GLP_entity_extraction",
    "stem": "GLP_knowledge_QA",
    # "coding": "ADVANCED_programing"
}

# ---------------------
# For controllability
# ---------------------
# mawps, mgsm
def is_all_digit(text: str) -> int:
    try:
        float(text)
        return 1
    except ValueError:
        return 0

# kobbq
def is_one_of_AB(text: str) -> int:
    return 1 if text in {"A", "B"} else 0
def is_one_of_ABCD(text: str) -> int:
    return 1 if text in {"A", "B", "C", "D"} else 0

# haerae_bench
def is_one_of_ABCDE(text: str) -> int:
    return 1 if text in {"A", "B", "C", "D", "E"} else 0

# JBLiMP
def is_a_b(text: str) -> int:
    return 1 if text in {"a", "b"} else 0

#kobest_copa
def is_1_2(text: str) -> int:
    return 1 if text in {"1", "2"} else 0
 
def is_0_4(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3", "4"} else 0

# kobest_hs
def is_0_3(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3"} else 0

# komoral, korean-hate-speech_hate, korean-hate-speech_bias
def is_0_1(text: str) -> int:
    return 1 if text in {"0", "1"} else 0

# kobest_sn
def is_pos_neg(text: str) -> int:
    return 1 if text in {"positive", "negative"} else 0

def is_entailment2_format(text: str) -> int:
    return 1 if text in {"entailment", "non-entailment"} else 0

# kornli
def is_entailment3_format(text: str) -> int:
    return 1 if text in {"entailment", "contradiction", "neutral"} else 0

# kobest_wic
def is_yes_no(text: str) -> int:
    return 1 if text in {"예", "아니오"} else 0 

def is_klue_re_format(text: str):
    return 1 if text in {"무관","종료일","설립일","본부소재지","별명","구성원","모임","종교단체","회사생산품","설립자","대표","회원수","출생일","사망일","출생한곳","사망한곳","거주지","출신","회사원","학교학생","별명","부모","자식","형제","배우자","가족","동료","개인생산품","종교","직위"} else 0

def is_klue_re_v11_format(text: str):
    return 1 if text in {"관계없음", "기관:해산", "기관:설립", "기관:본사위치", "기관:대체이름", "기관:소속", "기관:구성원", "기관:정치/종교 성향", "기관:제품", "기관:설립자", "기관:주요구성원/직원", "기관:직원/구성원수", "인물:출생일", "인물:사망일", "인물:출생지", "인물:사망지", "인물:거주지", "인물:출신", "인물:고용주", "인물:출신 학교", "인물:별명", "인물:부모", "인물:자녀", "인물:형제자매", "인물:배우자", "인물:기타 가족", "인물:동료", "인물:제품", "인물:종교", "인물:직함"} else 0

# no_check
def no_check(text: str):
    return None

controllability_dict = {
    "gsm8k": is_all_digit,
    "klue_ner": no_check,
    "klue_re": is_klue_re_format,
    "kobest_copa": is_1_2,
    "kobest_hs": is_0_3,
    "kobest_sn": is_pos_neg,
    "kobest_wic": is_yes_no,
    "komoral": is_0_1,
    "korea_cg": no_check,
    "korean-hate-speech_hate": is_0_1,
    "korean-hate-speech_bias": is_0_1,
    "korean-parallel-corpora-e2k": no_check,
    "korean-parallel-corpora-k2e": no_check,
    "kornli": is_entailment3_format,
    "korsts": is_all_digit,
    "kmmlu": is_one_of_ABCD,
    "mmlu_en": is_one_of_ABCD,
    "squad_kor_v1": no_check,
    "haerae_bench-HI": is_one_of_ABCDE,
    "haerae_bench-KGK": is_one_of_ABCDE,
    "haerae_bench-LW": is_one_of_ABCDE,
    "haerae_bench-RC": is_one_of_ABCDE,
    "haerae_bench-RW": is_one_of_ABCDE,
    "haerae_bench-SN": is_one_of_ABCDE,
    "kobbq": is_one_of_AB,
    "ko_truthful_qa": no_check,
}


kmmlu_dict = {
    'kmmlu_Aviation-Engineering-and-Maintenance': 'kmmlu',
    'kmmlu_Education': 'kmmlu',
    'kmmlu_Chemical-Engineering': 'kmmlu',
    'kmmlu_math': 'kmmlu',
    'kmmlu_Chemistry': 'kmmlu',
    'kmmlu_Maritime-Engineering': 'kmmlu',
    'kmmlu_korean-history': 'kmmlu',
    'kmmlu_Mechanical-Engineering': 'kmmlu',
    'kmmlu_Computer-Science': 'kmmlu',
    'kmmlu_Marketing': 'kmmlu',
    'kmmlu_Health': 'kmmlu',
    'kmmlu_Psychology': 'kmmlu',
    'kmmlu_Geomatics': 'kmmlu',
    'kmmlu_Ecology': 'kmmlu',
    'kmmlu_Electronics-Engineering': 'kmmlu',
    'kmmlu_Information-Technology': 'kmmlu',
    'kmmlu_Gas-Technology-and-Engineering': 'kmmlu',
    'kmmlu_Construction': 'kmmlu',
    'kmmlu_Railway-and-Automotive-Engineering': 'kmmlu',
    'kmmlu_Patent': 'kmmlu',
    'kmmlu_Energy-Management': 'kmmlu',
    'kmmlu_Machine-Design-and-Manufacturing': 'kmmlu',
    'kmmlu_Materials-Engineering': 'kmmlu',
    'kmmlu_Accounting': 'kmmlu',
    'kmmlu_Public-Safety': 'kmmlu',
    'kmmlu_Law': 'kmmlu',
    'kmmlu_Refrigerating-Machinery': 'kmmlu',
    'kmmlu_Taxation': 'kmmlu',
    'kmmlu_Telecommunications-and-Wireless-Technology': 'kmmlu',
    'kmmlu_Criminal-Law': 'kmmlu',
    'kmmlu_Agricultural-Sciences': 'kmmlu',
    'kmmlu_Biology': 'kmmlu',
    'kmmlu_Management': 'kmmlu',
    'kmmlu_Industrial-Engineer': 'kmmlu',
    'kmmlu_Economics': 'kmmlu',
    'kmmlu_Environmental-Science': 'kmmlu',
    'kmmlu_Social-Welfare': 'kmmlu',
    'kmmlu_Political-Science-and-Sociology': 'kmmlu',
    'kmmlu_Nondestructive-Testing': 'kmmlu',
    'kmmlu_Food-Processing': 'kmmlu',
    'kmmlu_Interior-Architecture-and-Design': 'kmmlu',
    'kmmlu_Real-Estate': 'kmmlu',
    'kmmlu_Civil-Engineering': 'kmmlu',
    'kmmlu_Electrical-Engineering': 'kmmlu',
    'kmmlu_Fashion': 'kmmlu'
}

from typing import List, Dict, Any
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method

@register_scaling_method("best_of_n")
class BestOfN(BaseScalingMethod):
    """
    N번 샘플링하여 가장 우수한 답변(또는 첫 답변)을 선택하는 방식.
    * "우수" 판단은 score_fn으로 할 수도 있고, default로 첫 번째만 택할 수도 있음.
    """

    def __init__(self, model: BaseModel = None, n: int = 5, batch_size:int=1, **kwargs):
        super().__init__(model=model, **kwargs)

        self.n = n
        self.batch_size=batch_size
        
        #self.score_fn = score_fn


    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        data 내 각 샘플에 대해:
          - N번 반복하여 후보 생성
          - score_fn이 있으면 최고 점수 후보 선택, 없으면 첫 후보
        """
        if self.model is None:
            raise ValueError("BestOfN requires a 'model' instance.")
        predictions=[]
        for i in range(0, len(data), self.batch_size):
            # prompt = sample["input"]
            # N번 후보 생성 (batch 호출로 최적화 가능)
            candidates = []
            candidates_logits = []

            sample_batch_list=[]
            for k in range(self.batch_size):
                sample_batch_list.append(data[i+k])
                

            for _ in range(self.n):
                # generate_batch는 리스트를 반환하므로 [0]만 취함
                outputs = self.model.generate_batch(sample_batch_list,batch_size=self.batch_size)
                # len(outputs) == batch size
                
                for i2 in range(len(outputs)):
                    candidate_text = outputs[i2]["prediction"]
                    candidates.append(candidate_text)  

                try:

                    result=self.model.score_batch(outputs)

                    for i2 in range(len(outputs)):
                        reward_score=result[i2]["reward"]
                        candidates_logits.append(reward_score)

                # 위는 모델이 multi 일때

                except:
                    
                    for i2 in range(len(outputs)):
                        candidate_text_logit=outputs[i2]["logits"]["sum_log_prob"]
                        candidates_logits.append(candidate_text_logit)

                # 위는 모델이 multi 아닐때

            for k2 in range(self.batch_size):
                    
                    sample = data[i + k2]
                    start_idx = k2
                    max_index = max(range(start_idx, len(candidates_logits), self.n), key=lambda k: candidates_logits[k])                        
                    sample["prediction"] = candidates[max_index]
                    predictions.append(sample)

            """

            batch size 3 이라면 총 3문제에 대한 질문이 한꺼번에 들어가는데 또한 ""best of n 의 n 이 3 이라면""
            [첫문제 에 대한 답1, 두번째 문제에 대한 답1, 세번째 문제 대한 답1, 첫문제에 대한 답2, .... , 세번째 문제에 대한 답3] , 길이 : 9
            이런식의 candidate 리스트가 만들어지고  
            
            [첫문제 에 대한 답1 score, .... , 세번째 문제에 대한 답3 score] , 길이 : 9
            라는 candidates_logits 리스트가 만들어진다. 
            
            여기서 첫문제에 대한 best of n 수행시 0,3,6 번째 index 중 한 답안이 pred가 되고
            두번째 문제는 1,4,7 번째 index 중 한 답안이 pred가 된다. 
            
            첫번째 문제를 예시를 들면, candidate logit의 0,3,6 인덱스 중 logit 이 가장 큰 index 를 candidate 리스트에 전달하고
            candidate 리스트의 그 인덱스로 부터(0,3,6 번째 중 한개의 pred) 첫번째 문제에 대한 best of n pred 결과출력 
            
            """
            
    
        return predictions   
        # return data

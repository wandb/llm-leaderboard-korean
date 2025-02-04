# best of n 실행 예시
# 이코드는 삭제하셔도 됩니다

from llm_eval.evaluator import Evaluator

evaluator = Evaluator(
    default_model_backend="huggingface",   
    default_scaling_method="best_of_n",            
    default_evaluation_method="string_match",
    default_split="test"
)


"""
results = evaluator.run(
    model="huggingface",                    
    dataset="haerae_bench",                  
    subset="csat_geo",                       
    split="test",                            
    dataset_params={"revision":"main"},      
    model_params={"model_name_or_path":"facebook/opt-350m","device":"cuda"},      
    scaling_method="best_of_n",                   
    scaling_params={"n":5,"batch_size":3},                                  
    evaluator_params={}                      
)
"""




results = evaluator.run(
    model="multi",                     
    dataset="haerae_bench",                 
    subset="csat_geo",                       
    split="test",                           
    dataset_params={"revision":"main"},      
    model_params={
          "generate_model": { "name": "huggingface", "params": { "model_name_or_path": "facebook/opt-350m" ,"device":"cuda"} },
          "judge_model": None,
          "reward_model": { "name": "huggingface_reward", "params": { "model_name_or_path": "facebook/opt-350m" ,"device":"cuda"}
        }},    
    scaling_method="best_of_n",                   
    scaling_params={"n":5,"batch_size":3},                      
    evaluator_params={}                    
)


print("Metrics:", results["metrics"])
print("Samples:", results["samples"][0])

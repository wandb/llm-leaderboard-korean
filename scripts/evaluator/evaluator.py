from .kaster import KasterEvaluator
from .haerae_bench_v1 import HaeraeBenchV1Evaluator
from .mtbench import MTBenchEvaluator
from .kobbq import KoBBQEvaluator
from .ko_truthful_qa import KoTruthfulQAEvaluator


class Evaluator():
    def __init__(self, kaster: bool, haerae_bench_v1: bool, mtbench: bool, kobbq: bool, ko_truthful_qa: bool):
        self.evaluators = []
        if kaster:
            self.evaluators.append(KasterEvaluator(few_shots=True))
            self.evaluators.append(KasterEvaluator(few_shots=False))
        if haerae_bench_v1:
            self.evaluators.append(HaeraeBenchV1Evaluator(few_shots=True))
            self.evaluators.append(HaeraeBenchV1Evaluator(few_shots=False))
        if mtbench:
            self.evaluators.append(MTBenchEvaluator(few_shots=False))
        if kobbq:
            self.evaluators.append(KoBBQEvaluator(few_shots=True))
        if ko_truthful_qa:
            self.evaluators.append(KoTruthfulQAEvaluator(few_shots=False))

    def evaluate(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()
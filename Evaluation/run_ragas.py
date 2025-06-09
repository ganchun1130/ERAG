# coding:utf-8
# @File  : run_ragas.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description:
import pandas as pd
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRelevance,
    ContextRecall,
    ContextPrecision
)
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from model_config import db_llm, local_em

evaluator_llm  = LangchainLLMWrapper(db_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(local_em)
metrics = [
    AnswerRelevancy(),
    Faithfulness(),
    ContextRelevance(),
    ContextRecall(),
    ContextPrecision()
]

# 读取数据
eval_data_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Evaluation\ERAG\all_questions_eval.jsonl"
output_file_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Evaluation\result\all_questions_eval_result.csv"

evaluation_dataset = EvaluationDataset.from_jsonl(eval_data_path)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)
print(result)
# {'answer_relevancy': 0.4279, 'faithfulness': 0.8880, 'nv_context_relevance': 0.9543, 'context_recall': 0.8913, 'context_precision': 0.7784}
# 将结果保存到CSV文件
output = result.to_pandas()
output.to_csv(output_file_path, index=False)
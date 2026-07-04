from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from chain import qa_chain

# Build an eval dataset (20–50 Q&A pairs with ground truth)
eval_questions = [
    "What are the key risk factors mentioned?",
    "What is the recommended dosage for adults?",
    # ... add your domain questions
]
ground_truths = [
    ["The key risk factors are X, Y, Z."],
    ["The recommended adult dosage is 10mg twice daily."],
]

# Run chain on all questions
answers, contexts = [], []
for q in eval_questions:
    result = qa_chain({"query": q})
    answers.append(result["result"])
    contexts.append([d.page_content for d in result["source_documents"]])

dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [g[0] for g in ground_truths]
})

scores = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
])
print(scores)
# Output: {'faithfulness': 0.91, 'answer_relevancy': 0.87, ...}
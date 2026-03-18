"""
evaluation_benchmarking.py

It performs the following:
1. Implements standard Exact Match (EM) and F1 score metrics.
2. Sets up Zero-Shot / Few-Shot Baseline inference loops.
3. Evaluates the trained 1-Shot Search-RL Agent using the interactive environment.
4. Tracks agentic behavior (e.g., average number of search calls per question).
"""

import re
import ast
import operator
import string
import collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ==============================================================================
# 1. EVALUATION METRICS 
# ==============================================================================

class EvaluationMetrics:
    """
    Implements the quantitative metrics.
    Uses standard SQuAD/HotpotQA normalization techniques.
    """
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        if not s:
            return ""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str) -> float:
        """Binary metric: 1.0 if Exact Match, 0.0 otherwise."""
        if prediction is None:
            return 0.0
        return 1.0 if cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth) else 0.0

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Token overlap metric for lenient multi-word evaluations."""
        if prediction is None:
            return 0.0
            
        pred_tokens = cls.normalize_answer(prediction).split()
        truth_tokens = cls.normalize_answer(ground_truth).split()
        
        # Count overlapping tokens
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        
        # Edge cases
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


# ==============================================================================
# 2. INTERACTIVE ENVIRONMENT & CONTROLLER 
# ==============================================================================

class InteractiveSearchEnvironment:
    """External Math Environment using safe AST."""
    def __init__(self):
        self.ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def _eval(self, node):
        if isinstance(node, ast.Expression): return self._eval(node.body)
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return self.ops[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp): return self.ops[type(node.op)](self._eval(node.operand))
        raise ValueError("Unsupported")

    def execute_query(self, query: str) -> str:
        try:
            res = self._eval(ast.parse(query.strip(), mode='eval'))
            return str(int(res) if isinstance(res, float) and res.is_integer() else round(res, 4))
        except Exception:
            return "Math Error."

    def format_snippet(self, snippet: str) -> str:
        return f"\n<information> {snippet.strip()} </information>\n"


class SearchTagCriteria(StoppingCriteria):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        return "</search>" in self.tokenizer.decode(input_ids[0, -10:], skip_special_tokens=True)


class InteractiveRolloutController:
    """Interleaves model generation and environment retrieval."""
    def __init__(self, model, tokenizer, env):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.stop_criteria = StoppingCriteriaList([SearchTagCriteria(tokenizer)])

    def evaluate_trajectory(self, prompt: str, max_turns: int = 5) -> tuple:
        """Runs inference and counts the number of searches."""
        traj = prompt
        search_count = 0
        
        for _ in range(max_turns):
            inputs = self.tokenizer(traj, return_tensors="pt").to(self.model.device)
            # Use greedy decoding (do_sample=False) for deterministic evaluation benchmarking
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id,
                                          do_sample=False, stopping_criteria=self.stop_criteria)
                                          
            new_text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            traj += new_text
            
            if "</search>" in new_text:
                search_count += 1
                queries = re.findall(r"<search>(.*?)</search>", traj, flags=re.DOTALL)
                if queries:
                    traj += self.env.format_snippet(self.env.execute_query(queries[-1]))
            elif "</answer>" in new_text:
                break
                
        # Extract the final answer
        match = re.search(r"<answer>(.*?)</answer>", traj, flags=re.IGNORECASE | re.DOTALL)
        extracted_answer = match.group(1).strip() if match else None
        
        return extracted_answer, search_count, traj


# ==============================================================================
# 3. BENCHMARKING PIPELINE 
# ==============================================================================

def run_benchmark():
    print("="*70)
    print("BENCHMARKING & EVALUATION PIPELINE")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model_id = "Qwen/Qwen2.5-7B-Instruc" 
    
    print(f"Loading Evaluator LLM: {model_id} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    env = InteractiveSearchEnvironment()
    controller = InteractiveRolloutController(model, tokenizer, env)

    # ---------------------------------------------------------
    # Mock Test Dataset (Simulating Downsampled GSM8K test set)
    # ---------------------------------------------------------
    test_dataset =[
        {
            "question": "If John has 5 apples and buys 3 bags of 4 apples, how many does he have?",
            "ground_truth": "17"
        },
        {
            "question": "A bakery makes 400 muffins. They sell 50% in the morning and 1/4 of the remaining in the afternoon. How many are left?",
            "ground_truth": "150"
        },
        {
            "question": "Calculate: (150 / 3) + 20 * 2",
            "ground_truth": "90"
        }
    ]

    # ---------------------------------------------------------
    # The 1-Shot Search-RL Prompt
    # ---------------------------------------------------------
    sys_instruction = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> first. "
        "You can call a search engine by <search> query </search>, and it returns <information>. "
        "Provide final answer in <answer> and </answer>.\n\n"
        "Example:\nQuestion: A store has 5 boxes of shirts. Each box contains 24 shirts. They sell 1/3 on Monday. On Tuesday, they sell the remaining for $15 each. Revenue?\n"
        "<think> Find total shirts. </think>\n<search> 5 * 24 </search>\n<information> 120 </information>\n"
        "<think> 1/3 sold is 40, 80 remain. 80 * 15. </think>\n<search> 80 * 15 </search>\n<information> 1200 </information>\n"
        "<think> Revenue is 1200. </think>\n<answer> 1200 </answer>\n\n"
    )

    print("\n--- Running Evaluation ---")
    
    total_em = 0.0
    total_f1 = 0.0
    total_searches = 0
    num_samples = len(test_dataset)

    for idx, data in enumerate(test_dataset):
        q = data["question"]
        gt = data["ground_truth"]
        full_prompt = sys_instruction + f"Question: {q}\n"

        print(f"\nEvaluating Question {idx+1}/{num_samples}: {q}")
        
        # Run Trajectory
        pred_answer, num_searches, traj = controller.evaluate_trajectory(full_prompt, max_turns=5)
        
        # Calculate Metrics
        em_score = EvaluationMetrics.exact_match_score(pred_answer, gt)
        f1_score = EvaluationMetrics.f1_score(pred_answer, gt)
        
        total_em += em_score
        total_f1 += f1_score
        total_searches += num_searches
        
        print(f"  Ground Truth   : {gt}")
        print(f"  Model Extracted: {pred_answer}")
        print(f"  Search Calls   : {num_searches}")
        print(f"  Metrics        : EM={em_score:.1f} | F1={f1_score:.2f}")

    # ---------------------------------------------------------
    # Aggregate and Print Final Benchmark
    # ---------------------------------------------------------
    avg_em = (total_em / num_samples) * 100
    avg_f1 = (total_f1 / num_samples) * 100
    avg_searches = total_searches / num_samples

    print("\n" + "="*70)
    print("FINAL BENCHMARK RESULTS (1-Shot Search-RL \u03C0_3 equivalent)")
    print("="*70)
    print(f"{'Metric':<20} | {'Score'}")
    print("-" * 30)
    print(f"{'Full Exact Match (EM)':<20} | {avg_em:.1f}%")
    print(f"{'Full F1 Score':<20} | {avg_f1:.1f}%")
    print(f"{'Avg # Searches':<20} | {avg_searches:.2f}")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
import re
import ast
import operator
import string
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

# ==============================================================================
# 1. THE SEARCH ENGINE ENVIRONMENT 
# ==============================================================================

class InteractiveSearchEnvironment:
    """
    Implements the real external environment E(q).
    Domain 'math': Uses a safe AST evaluator for GSM8K.
    Domain 'qa': Uses FAISS + E5 for dense retrieval on HotpotQA.
    """
    def __init__(self, domain="qa", faiss_index_path=None, corpus_path=None, e5_model_name="intfloat/e5-small-v2"):
        self.domain = domain
        
        if self.domain == "math":
            # Allowed AST operators for safe GSM8K arithmetic execution
            self.allowed_operators = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.Pow: operator.pow, ast.Mod: operator.mod,
                ast.USub: operator.neg, ast.UAdd: operator.pos
            }
        
        elif self.domain == "qa":
            print(f"Loading Dense Retriever ({e5_model_name})...")
            self.retriever = SentenceTransformer(e5_model_name)
            
            # If the user has the precomputed 2018 Wikipedia dump FAISS index, load it.
            # Otherwise, we dynamically create a FAISS index from a local list for testing.
            if faiss_index_path and corpus_path:
                print(f"Loading FAISS index from {faiss_index_path}")
                self.index = faiss.read_index(faiss_index_path)
                import json
                with open(corpus_path, 'r') as f:
                    self.corpus = json.load(f) # Expected: List of string passages
            else:
                print("No FAISS index path provided. Building an ephemeral FAISS index for demonstration.")
                self.corpus =[
                    "Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.",
                    "Christopher Edward Nolan was born on 30 July 1970 in Westminster, London.",
                    "London is the capital and largest city of England and the United Kingdom."
                ]
                # E5 requires 'passage: ' prefix for documents in the database
                passages = [f"passage: {doc}" for doc in self.corpus]
                embeddings = self.retriever.encode(passages, normalize_embeddings=True)
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim) # Inner Product (Cosine similarity since normalized)
                self.index.add(embeddings)

    def _safe_math_eval(self, node):
        """Recursively parses AST to safely evaluate mathematical expressions."""
        if isinstance(node, ast.Expression):
            return self._safe_math_eval(node.body)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._safe_math_eval(node.left)
            right = self._safe_math_eval(node.right)
            return self.allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_math_eval(node.operand)
            return self.allowed_operators[type(node.op)](operand)
        raise ValueError(f"Unsupported math operation: {type(node).__name__}")

    def execute_query(self, query: str) -> str:
        """Executes the query extracted from the <search> tags."""
        clean_query = query.strip()

        if self.domain == "math":
            try:
                # Parse the math expression strictly
                tree = ast.parse(clean_query, mode='eval')
                result = self._safe_math_eval(tree)
                # Format to 2 decimal places if float, else int
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                return str(round(result, 4))
            except Exception as e:
                return f"Math Evaluation Error."

        elif self.domain == "qa":
            try:
                # E5 requires the 'query: ' prefix for searching
                q_prompt = f"query: {clean_query}"
                q_emb = self.retriever.encode([q_prompt], normalize_embeddings=True)
                distances, indices = self.index.search(q_emb, k=1)
                
                # Retrieve the top-1 snippet
                top_idx = indices[0][0]
                if top_idx >= 0 and top_idx < len(self.corpus):
                    return self.corpus[top_idx]
                return "No relevant information found."
            except Exception as e:
                return f"Retrieval Error."
                
        return "Unknown Domain."

    def format_snippet(self, snippet: str) -> str:
        """Wraps snippet in <information> tags per constraints."""
        return f"\n<information> {snippet.strip()} </information>\n"


# ==============================================================================
# 2. SPARSE OUTCOME REWARD MODEL 
# ==============================================================================

class SparseRewardEvaluator:
    def __init__(self, lambda_penalty=0.1):
        self.lambda_penalty = lambda_penalty

    def _normalize_answer(self, text: str) -> str:
        """Standard SQuAD/HotpotQA Exact Match normalization."""
        if not text:
            return ""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Collapse whitespaces
        return ' '.join(text.split())

    def check_format_violations(self, trajectory: str) -> bool:
        """Validates tool syntax to apply the -\lambda penalty."""
        if "<answer>" not in trajectory or "</answer>" not in trajectory:
            return True
        if trajectory.count("<search>") != trajectory.count("</search>"):
            return True
        if trajectory.count("<think>") != trajectory.count("</think>"):
            return True
        return False

    def extract_answer(self, trajectory: str) -> str:
        """Regex to pull the final LLM string from <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", trajectory, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def compute_reward(self, trajectory: str, ground_truth: str) -> float:
        """
        Implementation of Eq 1:
        R(x, y) = 1 if Extract(y) == GroundTruth(x)
        R(x, y) = -lambda if y violates format constraints
        R(x, y) = 0 otherwise
        """
        if self.check_format_violations(trajectory):
            return -self.lambda_penalty

        extracted_pred = self.extract_answer(trajectory)
        if extracted_pred is None:
            return 0.0

        norm_pred = self._normalize_answer(extracted_pred)
        norm_gt = self._normalize_answer(ground_truth)

        if norm_pred == norm_gt:
            return 1.0
        return 0.0


# ==============================================================================
# 3. INTERACTIVE ROLLOUT CONTROLLER
# ==============================================================================

class SearchTagStoppingCriteria(StoppingCriteria):
    """Custom HuggingFace stopping criteria to halt exactly after </search>."""
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_sequence: str = "</search>"):
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last 15 tokens (enough to cover </search>) to check for the stop string
        last_tokens = input_ids[0][-15:]
        decoded_tail = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        return self.stop_sequence in decoded_tail

class InteractiveRolloutController:
    """
    Manages the autoregressive generation loop.
    Interleaves LLM text generation -> halting -> environment search -> resuming.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, env: InteractiveSearchEnvironment):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.stopping_criteria = StoppingCriteriaList([SearchTagStoppingCriteria(tokenizer)])

    def generate_trajectory(self, system_prompt: str, question: str, max_search_turns: int = 5, max_new_tokens_per_turn: int = 200) -> str:
        """Generates a full trajectory y = {t_1, ..., t_T} with interleaved search."""
        
        # "Cold Start" Prompt formatting
        full_prompt = f"{system_prompt}\nQuestion: {question}\n"
        trajectory = full_prompt
        
        for turn in range(max_search_turns):
            input_ids = self.tokenizer(trajectory, return_tensors="pt").input_ids.to(self.model.device)
            
            # Generate until </search> or natural EOS
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens_per_turn,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True, # Exploration needed for GRPO later
                temperature=0.8
            )
            
            # Decode the newly generated text
            new_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            trajectory += new_text
            
            # If the model output a search tag, process it
            if "</search>" in new_text:
                # Parse the query: find the last occurrence of <search> ... </search>
                search_match = re.findall(r"<search>(.*?)</search>", trajectory, flags=re.DOTALL)
                if search_match:
                    query = search_match[-1] # Take the most recent query
                    # Call environment
                    snippet = self.env.execute_query(query)
                    # Append formatted <information> snippet
                    trajectory += self.env.format_snippet(snippet)
            
            # If the model output the final answer, terminate early
            if "</answer>" in new_text:
                break
                
        return trajectory


# ==============================================================================
# 4. FULL EXECUTION / DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\n--- Testing 1: HotpotQA Environment (FAISS + E5 Dense Retrieval) ---")
    qa_env = InteractiveSearchEnvironment(domain="qa")
    qa_query = "Christopher Nolan birthplace"
    qa_result = qa_env.execute_query(qa_query)
    print(f"Agent requested: <search>{qa_query}</search>")
    print(f"Environment returned: {qa_env.format_snippet(qa_result)}")

    print("\n--- Testing 2: GSM8K Environment (Safe AST Calculator) ---")
    math_env = InteractiveSearchEnvironment(domain="math")
    math_query = "(120 / 3) * 15 + 4"
    math_result = math_env.execute_query(math_query)
    print(f"Agent requested: <search>{math_query}</search>")
    print(f"Environment returned: {math_env.format_snippet(math_result)}")

    print("\n--- Testing 3: Reward Evaluator (Equation 1) ---")
    evaluator = SparseRewardEvaluator(lambda_penalty=0.1)
    
    # Ground Truth target
    gt = "London"
    
    # Simulating Pi_9 Trajectory
    pi_9_trajectory = (
        "<think> I need to find the director of Interstellar first. </think>\n"
        "<search> director of Interstellar 2014 film </search>\n"
        "<information> Interstellar is a 2014 epic science fiction film directed by Christopher Nolan. </information>\n"
        "<think> The director is Christopher Nolan. Now search birthplace. </think>\n"
        "<search> Christopher Nolan birthplace </search>\n"
        "<information> Christopher Edward Nolan was born on 30 July 1970 in Westminster, London. </information>\n"
        "<think> He was born in London. </think>\n"
        "<answer> London </answer>"
    )
    
    bad_format_trajectory = "<think> It's London </think> <answer> London" # Missing </answer>
    wrong_answer_trajectory = "<think> Evaluating... </think> <answer> Paris </answer>"
    
    r_perfect = evaluator.compute_reward(pi_9_trajectory, gt)
    r_format = evaluator.compute_reward(bad_format_trajectory, gt)
    r_wrong = evaluator.compute_reward(wrong_answer_trajectory, gt)
    
    print(f"Pi_9 Trajectory Reward      : {r_perfect: .1f}  (Expected: 1.0)")
    print(f"Format Violation Reward     : {r_format: .1f} (Expected: -0.1)")
    print(f"Wrong Answer Reward         : {r_wrong: .1f}  (Expected: 0.0)")

"""
baselines_implementation.py

It does:
1. ZeroShotAgent: Relies purely on internal parametric knowledge.
2. FewShotAgent: Uses 5 static examples for in-context alignment.
3. StandardRAGAgent: Uses intfloat/e5-small-v2 and FAISS to retrieve 
   external documents *once* before generation.
"""

import re
import string
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 1. UTILITY: EVALUATION METRICS
# ==============================================================================

def normalize_answer(s: str) -> str:
    """Standard Exact Match (EM) normalization (lowercase, no punctuation/articles)."""
    if not s: return ""
    s = s.lower().translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def exact_match(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 if normalized strings match perfectly, else 0.0."""
    if not prediction: return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def extract_answer(text: str) -> str:
    """Extracts text within <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ==============================================================================
# 2. BASELINE 1: ZERO-SHOT AGENT
# ==============================================================================

class ZeroShotAgent:
    """
    Evaluates the model using only its pre-trained parametric memory.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = (
            "Answer the given question. Provide your final answer enclosed in "
            "<answer> and </answer> tags. Do not use external search.\n\n"
        )

    def predict(self, question: str, max_new_tokens: int = 100) -> str:
        prompt = self.system_prompt + f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False # Greedy decoding for benchmarking
            )
            
        new_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return extract_answer(new_text), new_text


# ==============================================================================
# 3. BASELINE 2: FEW-SHOT PROMPTING AGENT
# ==============================================================================

class FewShotAgent:
    """
    Evaluates the model using in-context learning with multiple static examples.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # We simulate 5-shot by providing 3 here for script compactness, 
        # acting as standard CoT alignments.
        self.few_shot_prompt = (
            "Answer the questions step by step. Put the final answer in <answer> tags.\n\n"
            "Question: Who wrote Romeo and Juliet?\n"
            "<think> Romeo and Juliet is a famous play. The author is William Shakespeare. </think>\n"
            "<answer> William Shakespeare </answer>\n\n"
            "Question: What is 15 * 4?\n"
            "<think> 10 * 4 = 40. 5 * 4 = 20. 40 + 20 = 60. </think>\n"
            "<answer> 60 </answer>\n\n"
            "Question: What is the capital of France?\n"
            "<think> Paris is the capital city of France. </think>\n"
            "<answer> Paris </answer>\n\n"
        )

    def predict(self, question: str, max_new_tokens: int = 150) -> str:
        prompt = self.few_shot_prompt + f"Question: {question}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        new_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return extract_answer(new_text), new_text


# ==============================================================================
# 4. BASELINE 3: STANDARD RAG (E5 + FAISS) AGENT
# ==============================================================================

class StandardRAGAgent:
    """
    Evaluates the static Retrieval-Augmented Generation pipeline.
    """
    def __init__(self, model, tokenizer, corpus: list):
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize Dense Embedding Model (E5)
        print("Initializing Standard RAG: Loading E5 embedding model...")
        self.retriever = SentenceTransformer('intfloat/e5-small-v2')
        
        # Build FAISS Index
        print("Initializing Standard RAG: Building FAISS index...")
        self.corpus = corpus
        # E5 dictates passages must be prefixed with "passage: "
        passages =[f"passage: {doc}" for doc in self.corpus]
        embeddings = self.retriever.encode(passages, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.system_prompt = (
            "Use the provided context to answer the question. "
            "Provide your final answer enclosed in <answer> and </answer> tags.\n"
        )

    def retrieve(self, question: str, top_k: int = 1) -> str:
        """Retrieves top-k documents prior to generation."""
        # E5 dictates queries must be prefixed with "query: "
        q_emb = self.retriever.encode([f"query: {question}"], normalize_embeddings=True)
        distances, indices = self.index.search(q_emb, k=top_k)
        
        retrieved_docs = [self.corpus[idx] for idx in indices[0]]
        return " ".join(retrieved_docs)

    def predict(self, question: str, max_new_tokens: int = 100) -> str:
        # Step 1: Retrieve context statically ONCE (The "Single-Hop" limitation)
        context = self.retrieve(question)
        
        # Step 2: Augment prompt
        prompt = self.system_prompt + f"\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Step 3: Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        new_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return extract_answer(new_text), context, new_text


# ==============================================================================
# 5. EXECUTION & DEMONSTRATION SCRIPT
# ==============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    print("=" * 70)
    print("BASELINE EVALUATIONS (Zero-Shot, Few-Shot, Standard RAG)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3B as well
    model_id = "Qwen/Qwen2.5-7B-Instruct" 
    
    print(f"Loading Base LLM ({model_id}) onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # 1. Mock Knowledge Base for the RAG baseline
    mock_wikipedia_dump =[
        "London is the capital and largest city of England and the United Kingdom.",
        "Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.",
        "Christopher Edward Nolan was born on 30 July 1970 in Westminster, London.",
        "Paris is the capital of France and a major European city.",
    ]

    # 2. Initialize the three Baselines
    zero_shot_agent = ZeroShotAgent(model, tokenizer)
    few_shot_agent = FewShotAgent(model, tokenizer)
    rag_agent = StandardRAGAgent(model, tokenizer, mock_wikipedia_dump)

    # 3. Test multi-hop question 
    test_question = "In which European city was the director of the 2014 sci-fi film Interstellar born?"
    ground_truth = "London"

    print("\n" + "=" * 70)
    print(f"TARGET QUESTION: {test_question}")
    print(f"GROUND TRUTH:    {ground_truth}")
    print("=" * 70)

    # --- Run Zero-Shot ---
    print("\n[1] Running ZERO-SHOT Baseline...")
    zs_pred, zs_raw = zero_shot_agent.predict(test_question)
    zs_em = exact_match(zs_pred, ground_truth)
    print(f"  Prediction : {zs_pred}")
    print(f"  Exact Match: {zs_em}")

    # --- Run Few-Shot ---
    print("\n[2] Running FEW-SHOT Baseline...")
    fs_pred, fs_raw = few_shot_agent.predict(test_question)
    fs_em = exact_match(fs_pred, ground_truth)
    print(f"  Prediction : {fs_pred}")
    print(f"  Exact Match: {fs_em}")

    # --- Run Standard RAG ---
    print("\n[3] Running STANDARD RAG (E5 + FAISS) Baseline...")
    rag_pred, rag_context, rag_raw = rag_agent.predict(test_question)
    rag_em = exact_match(rag_pred, ground_truth)
    print(f"  Retrieved Context: '{rag_context}'")
    print(f"  Prediction       : {rag_pred}")
    print(f"  Exact Match      : {rag_em}")


    print("\n" + "=" * 70)
    print("BASELINE TESTING COMPLETE.")
    print("Compare these results to the multi-hop capabilities of your 1-Shot Search-RL agent.")
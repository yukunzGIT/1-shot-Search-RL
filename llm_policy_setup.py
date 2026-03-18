"""
llm_policy_setup.py

This script does:
1. Loads the Policy LLM and Reference LLM (Qwen-Instruct models).
2. Freezes the Reference LLM to prevent gradient updates (used for KL divergence).
3. Constructs the exact 1-Shot "Cold Start" system prompt
4. Executes an interactive multi-turn rollout where the LLM actually calls the search environment.

We can switch it back to 7B via the function arguments.
"""

import re
import ast
import operator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ==============================================================================
# PART A: ENVIRONMENT MODULE
# ==============================================================================

class InteractiveSearchEnvironment:
    """Mock search environment handling QA and Math domains."""
    def __init__(self, domain="qa"):
        self.domain = domain
        # Simple local DB for the HotpotQA demonstration
        self.mock_kb = {
            "director of interstellar 2014 film": "Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.",
            "christopher nolan birthplace": "Christopher Edward Nolan was born on 30 July 1970 in Westminster, London.",
        }

    def execute_query(self, query: str) -> str:
        """Executes the extracted query against the environment."""
        clean_query = query.strip().lower()
        if self.domain == "qa":
            return self.mock_kb.get(clean_query, "No relevant information found.")
        elif self.domain == "math":
            try:
                # Safe math execution using eval for basic arithmetic
                return str(eval(clean_query, {"__builtins__": None}, {}))
            except Exception:
                return "Math Evaluation Error."
        return "Unknown Domain."

    def format_snippet(self, snippet: str) -> str:
        """Wraps the result in <information> tags."""
        return f" <information> {snippet} </information>\n"


# ==============================================================================
# PART B: 1-SHOT PROMPT BUILDER 
# ==============================================================================

def build_1shot_prompt(question: str, domain="qa") -> str:
    """
    Constructs the 1-Shot Cold Start prompt. 
    It includes the System Instruction and the Pi_9 / Pi_3 example 
    """
    # 1. System Instruction verbatim 
    sys_instruction = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. After reasoning, if you find you lack "
        "some knowledge, you can call a search engine by <search> query </search>, and it "
        "will return the top searched results between <information> and </information>. "
        "You can search as many times as you want. If you find no further external knowledge "
        "needed, you can directly provide the answer inside <answer> and </answer> without "
        "detailed illustrations.\n\n"
    )

    # 2. Append the optimal high-variance example based on domain
    if domain == "qa":
        # The Pi_9 Example 
        one_shot_example = (
            "Example:\n"
            "Question: The director of the 2014 sci-fi film Interstellar was born in which European city?\n"
            "<think> I need to find the director of Interstellar first, and then find where they were born. </think>\n"
            "<search> director of Interstellar 2014 film </search>\n"
            "<information> Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. </information>\n"
            "<think> The director is Christopher Nolan. Now I must search for his birthplace. </think>\n"
            "<search> Christopher Nolan birthplace </search>\n"
            "<information> Christopher Edward Nolan was born on 30 July 1970 in Westminster, London. </information>\n"
            "<think> He was born in Westminster, London. London is a European city. </think>\n"
            "<answer> London </answer>\n\n"
        )
    elif domain == "math":
        # The Pi_3 Example 
        one_shot_example = (
            "Example:\n"
            "Question: A store has 5 boxes of shirts. Each box contains 24 shirts. The store sells 1/3 of the shirts on Monday. On Tuesday, they sell the remaining shirts for $15 each. How much revenue did they make on Tuesday?\n"
            "<think> I need to track the inventory across multiple steps. First, find total shirts. </think>\n"
            "<search> 5 * 24 </search>\n"
            "<information> 120 </information>\n"
            "<think> Total is 120. They sold 1/3 on Monday, which is 120 / 3 = 40. The remaining shirts are 120 - 40 = 80. Now, calculate revenue for 80 shirts at $15 each. </think>\n"
            "<search> 80 * 15 </search>\n"
            "<information> 1200 </information>\n"
            "<think> The revenue made on Tuesday is 1200 dollars. </think>\n"
            "<answer> 1200 </answer>\n\n"
        )
    else:
        one_shot_example = ""

    # 3. Append the actual target question to solve
    target_query = f"Question: {question}\n"
    
    return sys_instruction + one_shot_example + target_query


# ==============================================================================
# PART C: LLM POLICY & REFERENCE SETUP
# ==============================================================================

def setup_llm_models(model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    Loads both the Policy LLM and the Reference LLM into memory.
    The Reference LLM is strictly frozen (requires_grad = False).
    """
    print(f"Loading Tokenizer from: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Ensure pad token is set (crucial for batched generation/GRPO later)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Policy LLM (pi_theta) from: {model_name_or_path}...")
    # Load the policy model (this will be optimized by GRPO)
    # Using torch.bfloat16 for memory efficiency without losing precision
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading Reference LLM (pi_ref) from: {model_name_or_path}...")
    # Load the reference model (this remains frozen to compute KL Divergence)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Strictly FREEZE the reference model parameters
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model, tokenizer


# ==============================================================================
# PART D: INTERACTIVE ROLLOUT CONTROLLER
# ==============================================================================

class SearchTagStoppingCriteria(StoppingCriteria):
    """Custom Stopping Criteria to halt text generation exactly when </search> is predicted."""
    def __init__(self, tokenizer, stop_string="</search>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last 10 tokens to check if the stop_string was generated
        tail_tokens = input_ids[0, -10:]
        decoded_tail = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
        return self.stop_string in decoded_tail

class InteractiveRolloutController:
    """Manages the generation -> halt -> search -> resume loop."""
    def __init__(self, policy_model, tokenizer, environment):
        self.model = policy_model
        self.tokenizer = tokenizer
        self.env = environment
        self.stopping_criteria = StoppingCriteriaList([SearchTagStoppingCriteria(tokenizer)])

    def generate_trajectory(self, full_prompt: str, max_turns: int = 5) -> str:
        """Runs the multi-turn generation loop using the Policy LLM."""
        trajectory = full_prompt
        device = self.model.device

        for turn in range(max_turns):
            # Tokenize current trajectory
            inputs = self.tokenizer(trajectory, return_tensors="pt").to(device)
            
            # Generate new tokens
            with torch.no_grad(): # Use no_grad for rollout generation
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,   # Moderate temperature for exploration
                    top_p=0.9
                )
            
            # Extract only the newly generated text
            input_length = inputs.input_ids.shape[1]
            new_text = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
            trajectory += new_text

            # Check if generation halted due to a tool call
            if "</search>" in new_text:
                # Use regex to find the most recent query between <search> and </search>
                search_matches = re.findall(r"<search>(.*?)</search>", trajectory, flags=re.IGNORECASE | re.DOTALL)
                if search_matches:
                    latest_query = search_matches[-1]
                    print(f"  [Turn {turn+1}] Agent triggered search: '{latest_query.strip()}'")
                    
                    # Intersect with external environment
                    snippet = self.env.execute_query(latest_query)
                    formatted_snippet = self.env.format_snippet(snippet)
                    print(f"  [Turn {turn+1}] Environment returned: '{snippet.strip()}'")
                    
                    # Append external knowledge to trajectory and loop again
                    trajectory += formatted_snippet
                    continue 

            # Check if generation concluded with a final answer
            if "</answer>" in new_text:
                print(f"[Turn {turn+1}] Agent provided final answer.")
                break
            
            # If no tags were generated, the model stopped naturally (or hit max tokens)
            break
            
        return trajectory

# ==============================================================================
# PART E: MAIN EXECUTION / TESTING SCRIPT
# ==============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Suppress HF warnings for clean output
    
    print("-" * 60)
    print("POLICY & REFERENCE LLM SETUP (COLD START TEST)")
    print("-" * 60)

    # 1. Setup the Environment (HotpotQA domain)
    qa_env = InteractiveSearchEnvironment(domain="qa")

    # 2. Setup LLMs (can replace with "Qwen/Qwen2.5-3B-Instruct" )
    # The setup_llm_models function correctly freezes the reference model.
    MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct" 
    policy_llm, ref_llm, tokenizer = setup_llm_models(MODEL_PATH)
    
    # Verify the reference model is frozen
    assert not any(p.requires_grad for p in ref_llm.parameters()), "Reference LLM must be frozen!"
    print("Reference LLM successfully frozen.")

    # 3. Setup the Rollout Controller
    controller = InteractiveRolloutController(policy_llm, tokenizer, qa_env)

    # 4. Define a novel multi-hop question to test generalization
    test_question = "In which city was the director of the 2014 sci-fi film Interstellar born?"
    
    # 5. Build the 1-Shot prompt (Using Pi_9 format)
    prompt = build_1shot_prompt(test_question, domain="qa")
    
    print("\n--- Starting Interactive Rollout Generation ---")
    # 6. Execute the trajectory generation
    final_trajectory = controller.generate_trajectory(prompt, max_turns=3)
    
    print("\n--- Final Generated Trajectory ---")
    # We slice out the system prompt from the print statement to only show the model's new behavior
    print(final_trajectory[len(prompt):])
    print("-" * 60)
    print("Completed successfully. The Policy LLM is now capable of real-time environment interaction.")
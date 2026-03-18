"""
training_loop.py

It does:
1. The Policy and Reference LLMs (Qwen).
2. The Interactive Search Environment (Math/AST based for standalone execution).
3. The Outcome-based Verifiable Reward function.
4. The GRPO Loss calculation with proper token-shifting and masking.
5. The PyTorch optimization loop performing the policy update.
"""

import re
import ast
import operator
import string
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ==============================================================================
# 1. CORE COMPONENTS (Integrated from previous steps for standalone execution)
# ==============================================================================

class InteractiveSearchEnvironment:
    """External environment E(q) using AST for safe math calculations (GSM8K)."""
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


class SparseRewardEvaluator:
    """Calculates Sparse Reward R(x,y) per Equation 1."""
    def __init__(self, lambda_penalty=0.1):
        self.lambda_penalty = lambda_penalty

    def compute_reward(self, trajectory: str, ground_truth: str) -> float:
        # Check format constraints (Penalty)
        if "<answer>" not in trajectory or "</answer>" not in trajectory: return -self.lambda_penalty
        if trajectory.count("<search>") != trajectory.count("</search>"): return -self.lambda_penalty
        
        # Extract and compare
        match = re.search(r"<answer>(.*?)</answer>", trajectory, flags=re.IGNORECASE | re.DOTALL)
        if not match: return 0.0
        
        pred = match.group(1).strip().lower().translate(str.maketrans('', '', string.punctuation))
        pred = ' '.join(pred.split())
        gt = ground_truth.strip().lower().translate(str.maketrans('', '', string.punctuation))
        gt = ' '.join(gt.split())
        
        return 1.0 if pred == gt else 0.0


class SearchTagCriteria(StoppingCriteria):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        return "</search>" in self.tokenizer.decode(input_ids[0, -10:], skip_special_tokens=True)

class InteractiveRolloutController:
    """Interleaves generation and environment queries."""
    def __init__(self, model, tokenizer, env):
        self.model, self.tokenizer, self.env = model, tokenizer, env
        self.stop_criteria = StoppingCriteriaList([SearchTagCriteria(tokenizer)])

    def generate(self, prompt: str, max_turns=3) -> str:
        traj = prompt
        for _ in range(max_turns):
            inputs = self.tokenizer(traj, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=64, pad_token_id=self.tokenizer.eos_token_id,
                                          do_sample=True, temperature=0.8, stopping_criteria=self.stop_criteria)
            new_text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            traj += new_text
            
            if "</search>" in new_text:
                queries = re.findall(r"<search>(.*?)</search>", traj, flags=re.DOTALL)
                if queries: traj += self.env.format_snippet(self.env.execute_query(queries[-1]))
            elif "</answer>" in new_text: break
        return traj


# ==============================================================================
# 2. GRPO LOSS & MASKING UTILITIES 
# ==============================================================================

def create_trajectory_mask(input_ids: torch.Tensor, prompt_len: int, tokenizer) -> torch.Tensor:
    """Creates M_{i,t}: 1 for LLM tokens, 0 for prompt, padding, or environment info."""
    mask = torch.ones_like(input_ids, dtype=torch.float32)
    mask[:, :prompt_len] = 0.0 # Mask prompt
    
    for i in range(input_ids.shape[0]):
        tokens = input_ids[i].tolist()
        in_info = False
        for t in range(prompt_len, len(tokens)):
            # Mask out padding tokens
            if tokens[t] == tokenizer.pad_token_id:
                mask[i, t] = 0.0
                continue
            
            tok_str = tokenizer.decode([tokens[t]])
            if "<information>" in tok_str: in_info = True
            if in_info: mask[i, t] = 0.0
            if "</information>" in tok_str: in_info = False
            
    return mask

class GRPOLoss(torch.nn.Module):
    def __init__(self, clip_eps=0.2, beta=0.01, alpha=0.05):
        super().__init__()
        self.clip_eps, self.beta, self.alpha = clip_eps, beta, alpha

    def forward(self, logits_theta, logits_old, logits_ref, actions, advantages, mask):
        # Shift sequences for causal LM: Logits at t predict Action at t+1
        logits_theta = logits_theta[:, :-1, :].contiguous()
        logits_old = logits_old[:, :-1, :].contiguous()
        logits_ref = logits_ref[:, :-1, :].contiguous()
        actions = actions[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()

        # Log probabilities
        log_pi_theta = F.log_softmax(logits_theta, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        log_pi_old = F.log_softmax(logits_old, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        log_pi_ref = F.log_softmax(logits_ref, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # 1. Policy Gradient Loss
        ratio = torch.exp(log_pi_theta - log_pi_old)
        adv = advantages.unsqueeze(1).expand_as(ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        l_pg = -(torch.min(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)

        # 2. KL Loss (Unbiased estimator)
        ratio_kl = torch.exp(log_pi_ref - log_pi_theta)
        l_kl = ((ratio_kl - torch.log(ratio_kl) - 1.0) * mask).sum() / (mask.sum() + 1e-8)

        # 3. Entropy Loss
        probs_theta = F.softmax(logits_theta, dim=-1)
        l_ent = (-torch.sum(probs_theta * F.log_softmax(logits_theta, dim=-1), dim=-1) * mask).sum() / (mask.sum() + 1e-8)

        total_loss = l_pg + (self.beta * l_kl) - (self.alpha * l_ent)
        return total_loss, {"pg": l_pg.item(), "kl": l_kl.item(), "ent": l_ent.item()}


# ==============================================================================
# 3. MAIN TRAINING LOOP 
# ==============================================================================

def train_1_shot_agent():
    print("="*60)
    print("STARTING 1-SHOT SEARCH-RL TRAINING LOOP")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 3B, Replace with "Qwen/Qwen2.5-7B-Instruct" for production.
    model_id = "Qwen/Qwen2.5-7B-Instruct" 
    
    print(f"Loading Models onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Policy model (requires gradients)
    policy_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    policy_model.train()
    
    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    ref_model.eval()
    for param in ref_model.parameters(): param.requires_grad = False

    # Optimizers & Environments
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    env = InteractiveSearchEnvironment()
    controller = InteractiveRolloutController(policy_model, tokenizer, env)
    evaluator = SparseRewardEvaluator()
    grpo_criterion = GRPOLoss(clip_eps=0.2, beta=0.01, alpha=0.05)

    # --------------------------------------------------------------------------
    # The 1-Shot GSM8K Example (\pi_3 blueprint)
    # --------------------------------------------------------------------------
    sys_prompt = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> first. "
        "You can call a search engine by <search> query </search>, and it returns <information>. "
        "Provide final answer in <answer> and </answer>.\n\n"
        "Example:\nQuestion: A store has 5 boxes of shirts. Each box contains 24 shirts. They sell 1/3 on Monday. On Tuesday, they sell the remaining for $15 each. Revenue?\n"
        "<think> Find total shirts. </think>\n<search> 5 * 24 </search>\n<information> 120 </information>\n"
        "<think> 1/3 sold is 40, 80 remain. 80 * 15. </think>\n<search> 80 * 15 </search>\n<information> 1200 </information>\n"
        "<think> Revenue is 1200. </think>\n<answer> 1200 </answer>\n\n"
    )
    
    # The unseen target question we are training the agent to solve using RL
    target_q = "Question: I have 4 baskets, each holding 15 apples. I eat 20% of them. How many apples are left?\n"
    ground_truth = "48"
    
    full_prompt = sys_prompt + target_q
    prompt_len = len(tokenizer.encode(full_prompt))

    # Hyperparameters
    G = 4        # Group size (Number of trajectories per iteration)
    EPOCHS = 3   # Number of RL update steps
    
    print("\nStarting GRPO Rollouts and Optimization...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 1. ROLLOUT PHASE (Generate G trajectories)
        trajectories =[]
        for g in range(G):
            traj = controller.generate(full_prompt, max_turns=3)
            trajectories.append(traj)
            
        # 2. EVALUATION PHASE (Compute Sparse Rewards)
        rewards = torch.tensor([evaluator.compute_reward(t, ground_truth) for t in trajectories], dtype=torch.float32, device=device)
        
        # Calculate Advantages (Standardize group rewards)
        if rewards.std() < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
        print(f"  Rewards: {rewards.tolist()} | Advantages: {[round(a, 2) for a in advantages.tolist()]}")

        # 3. PREPARATION PHASE (Tokenize batch for forward pass)
        inputs = tokenizer(trajectories, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Create M_{i,t} masking indicator
        mask = create_trajectory_mask(input_ids, prompt_len, tokenizer)
        
        # 4. OPTIMIZATION PHASE (Forward passes & Loss)
        # Policy Forward Pass
        logits_theta = policy_model(input_ids, attention_mask=attention_mask).logits
        
        with torch.no_grad():
            # Reference Forward Pass
            logits_ref = ref_model(input_ids, attention_mask=attention_mask).logits
            # In GRPO (and PPO), pi_old is the policy that generated the data. 
            # Since we generate and update immediately, pi_old = policy_model.detach()
            logits_old = logits_theta.detach().clone()
            
        # Compute GRPO Loss
        loss, metrics = grpo_criterion(logits_theta, logits_old, logits_ref, input_ids, advantages, mask)
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f} (PG: {metrics['pg']:.4f}, KL: {metrics['kl']:.4f}, Ent: {metrics['ent']:.4f})")
        
        # Free memory
        del inputs, input_ids, attention_mask, logits_theta, logits_ref, logits_old, loss
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("TRAINING COMPLETE.")
    print("The agent's weights have been autonomously updated based on the interactive sparse rewards!")
    print("="*60)


if __name__ == "__main__":
    train_1_shot_agent()
import numpy as np
import json
from datasets import load_dataset, Dataset
from collections import defaultdict

# ==============================================================================
# 1. DATASET LOADER MODULE 
# ==============================================================================

class RLVRDataLoader:
    """
    Handles loading and stratified downsampling of HotpotQA and GSM8K 
    """
    def __init__(self, seed=42):
        self.seed = seed

    def load_hotpot_qa(self, num_samples=1000):
        """
        Loads HotpotQA 'train-hard' subset and performs stratified sampling 
        to preserve 'Bridge' vs 'Comparison' distribution.
        """
        # Load the distractor (hard) set from HuggingFace
        raw_data = load_dataset("hotpot_qa", "distractor", split="train")
        
        # Identify categories for stratification
        bridge_indices = [i for i, x in enumerate(raw_data) if x['type'] == 'bridge']
        comparison_indices = [i for i, x in enumerate(raw_data) if x['type'] == 'comparison']
        
        # Calculate ratios to maintain original distribution
        total = len(bridge_indices) + len(comparison_indices)
        n_bridge = int((len(bridge_indices) / total) * num_samples)
        n_comp = num_samples - n_bridge
        
        # Randomly sample within strata
        np.random.seed(self.seed)
        sampled_bridge = np.random.choice(bridge_indices, n_bridge, replace=False)
        sampled_comp = np.random.choice(comparison_indices, n_comp, replace=False)
        
        final_indices = np.concatenate([sampled_bridge, sampled_comp])
        return raw_data.select(final_indices)

    def load_gsm8k(self, num_samples=1000):
        """
        Loads GSM8K and extracts numerical answers after ####.
        """
        raw_data = load_dataset("gsm8k", "main", split="train")
        # Sample randomly
        np.random.seed(self.seed)
        indices = np.random.choice(range(len(raw_data)), num_samples, replace=False)
        return raw_data.select(indices)

# ==============================================================================
# 2. HISTORICAL VARIANCE RANKER 
# ==============================================================================

class DataSelector:
    """
    Implements the core Data Selector logic: Ranking examples by 
    Historical Variance Accuracy
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Dictionary to store accuracy per epoch for each example index
        self.history = defaultdict(list)

    def record_accuracy(self, example_idx, accuracy):
        """
        Updates the historical record for an example.
        accuracy: 1.0 (Exact Match) or 0.0
        """
        self.history[example_idx].append(float(accuracy))

    def calculate_variance_ranking(self):
        """
        Computes variance v_i = Var(a_i,1 ... a_i,E) and ranks indices.
        Returns a list of tuples (original_index, variance_score)
        """
        scores = []
        for idx, acc_list in self.history.items():
            # Apply Equation 2: calculate variance of accuracy over epochs
            # High variance indicates the example is at the decision boundary 
            v_i = np.var(acc_list) if len(acc_list) > 0 else 0.0
            scores.append((idx, v_i))
        
        # Sort by variance in descending order (pi_1 has highest variance)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def select_pi_example(self, rank=1):
        """
        Selects the example at rank 'n' (e.g., pi_1, pi_2).
        """
        ranked_scores = self.calculate_variance_ranking()
        # rank-1 because rank is 1-indexed (pi_1)
        target_idx = ranked_scores[rank-1][0]
        return self.dataset[target_idx]

# ==============================================================================
# 3. STANDALONE EXECUTION 
# ==============================================================================

if __name__ == "__main__":
    # Initialize the loader
    loader = RLVRDataLoader(seed=42)
    
    print("Loading and downsampling HotpotQA dataset...")
    # Load 1000 samples with stratified distribution
    dataset = loader.load_hotpot_qa(num_samples=1000)
    
    # Initialize Data Selector
    selector = DataSelector(dataset)
    
    # SIMULATION OF PRELIMINARY TRAINING RUN 
    # To identify pi_1, we simulate accuracy history across 5 epochs.
    print("Simulating historical accuracy tracking over 5 epochs...")
    num_epochs = 5
    for idx in range(len(dataset)):
        # We simulate a "high variance" example at index 42 (oscillating 0 and 1)
        if idx == 42:
            mock_acc = [0, 1, 0, 1, 0]
        # We simulate a "low variance" example at index 10 (always 0)
        elif idx == 10:
            mock_acc = [0, 0, 0, 0, 0]
        # Random behavior for others
        else:
            mock_acc = np.random.choice([0, 1], size=num_epochs).tolist()
            
        for epoch_val in mock_acc:
            selector.record_accuracy(idx, epoch_val)

    # Calculate Ranking
    print("Ranking examples by Historical Variance...")
    ranked_examples = selector.calculate_variance_ranking()
    
    # Select pi_1 (The primary 1-shot example)
    pi_1 = selector.select_pi_example(rank=1)
    
    print("\nSUCCESS: pi_1 Identified.")
    print(f"Top Example Index: {ranked_examples[0][0]}")
    print(f"Variance Score:    {ranked_examples[0][1]:.4f}")
    print("-" * 50)
    print(f"Question for 1-Shot Training:\n{pi_1['question']}")
    print("-" * 50)
    
    # Save the selected pi_1 for the next step (Initialization)
    with open("pi_1_sample.json", "w") as f:
        json.dump(pi_1, f)
    print("Saved selected 1-shot example to 'pi_1_sample.json'.")
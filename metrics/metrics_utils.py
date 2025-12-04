import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

# =============================================================================
# AUTOMATIC SETUP
# =============================================================================
# This block ensures that whoever imports this file has the necessary NLTK data
# without crashing. It fixes the "punkt_tab" error you faced earlier.
def _ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"[metrics_utils] Downloading missing NLTK resource: {res}...")
            nltk.download(res, quiet=True)

# Run setup immediately upon import
_ensure_nltk_resources()

# =============================================================================
# SCORING FUNCTION
# =============================================================================
def calculate_metrics(references, hypotheses):
    """
    Calculates BLEU-1, BLEU-4, and ROUGE-L scores.
    
    Args:
        references (list of str): The ground truth reports (list of strings).
        hypotheses (list of str): The generated reports (list of strings).
        
    Returns:
        dict: A dictionary containing 'BLEU-1', 'BLEU-4', and 'ROUGE-L' scores.
    """
    # Initialize scorers
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    b1_scores = []
    b4_scores = []
    rouge_scores = []
    
    # Iterate through pairs
    for ref, hyp in zip(references, hypotheses):
        # clean and tokenize
        # We use .strip() to remove accidental whitespace lines
        ref_text = str(ref).lower().strip()
        hyp_text = str(hyp).lower().strip()
        
        # Tokenize (split into words)
        ref_tokens = nltk.word_tokenize(ref_text)
        hyp_tokens = nltk.word_tokenize(hyp_text)
        
        # --- BLEU Scores ---
        # Note: NLTK expects reference as a list of lists of tokens [[w1, w2...]]
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            b1_scores.append(0.0)
            b4_scores.append(0.0)
        else:
            # BLEU-1 (Unigram - good for single word accuracy)
            b1_scores.append(sentence_bleu(
                [ref_tokens], 
                hyp_tokens, 
                weights=(1, 0, 0, 0), 
                smoothing_function=smooth
            ))
            
            # BLEU-4 (4-gram - good for fluency)
            b4_scores.append(sentence_bleu(
                [ref_tokens], 
                hyp_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25), 
                smoothing_function=smooth
            ))
        
        # --- ROUGE Score ---
        # ROUGE-L measures longest matching sequence (structure)
        r_score = scorer.score(ref_text, hyp_text)
        rouge_scores.append(r_score['rougeL'].fmeasure)

    # Return averages
    return {
        "BLEU-1": np.mean(b1_scores),
        "BLEU-4": np.mean(b4_scores),
        "ROUGE-L": np.mean(rouge_scores)
    }

# =============================================================================
# SELF-TEST (Runs only if you execute this file directly)
# =============================================================================
if __name__ == "__main__":
    print("Testing metrics utility...")
    
    # Dummy data
    refs = ["the lungs are clear", "cardiomegaly is present"]
    hyps = ["lungs are clear", "cardiomegaly is seen"]
    
    scores = calculate_metrics(refs, hyps)
    
    print("Test Scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print("If you see scores above, the utility is working correctly.")

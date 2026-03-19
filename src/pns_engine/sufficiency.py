import re

def extract_answer_from_trace(trace_text):
    """
    Attempts to extract the final numerical answer from a CoT trace.
    Often, models put the answer at the end, sometimes after 'So The answer is' or '####'.
    """
    # Look for standard GSM8k format (if the model followed it)
    if "####" in trace_text:
        return trace_text.split("####")[-1].strip()
    
    # Fallback: look for the last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', trace_text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def extract_ground_truth(gt_text):
    """Extracts the exact answer from the GSM8k ground truth string."""
    if "####" in gt_text:
        return gt_text.split("####")[-1].strip()
    return gt_text.strip()

def is_trace_sufficient(trace_text, ground_truth_text):
    """
    Returns True if the trace reaches the correct final answer,
    meaning the reasoning path was 'Sufficient' to solve the problem.
    """
    predicted = extract_answer_from_trace(trace_text)
    actual = extract_ground_truth(ground_truth_text)
    
    if predicted is None or actual is None:
        return False
        
    # Simple float comparison to handle things like 72 vs 72.0
    try:
        return abs(float(predicted) - float(actual)) < 1e-5
    except ValueError:
        return predicted.lower() == actual.lower()

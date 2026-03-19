import re
import math
from vllm import SamplingParams

def split_into_steps(trace_text):
    """
    Very basic logic to split a trace into reasoning steps based on newlines
    or common numeration identifiers (e.g., '1.', '2.', etc.).
    """
    # Simply using newline splits as a naive approach.
    # In a real environment, you might use regex: re.split(r'\n(?=\d+\.)', trace_text)
    steps = [s.strip() for s in trace_text.split('\n') if s.strip()]
    return steps

def run_counterfactual_rollouts(llm_engine, problem_statement, previous_steps_text, alternative_step_text, num_rollouts=3):
    """
    Given the problem and the steps up through our new counterfactual,
    roll out 'num_rollouts' full generations to the end of the problem.
    """
    prompt = f"<|im_start|>user\nQuestion: {problem_statement}\nThink step-by-step and provide the final answer.<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{previous_steps_text}\n{alternative_step_text}\n"

    # We want varied completions to see if the model can "recover" from the bad step
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        n=num_rollouts
    )
    
    outputs = llm_engine.generate([prompt], sampling_params, use_tqdm=False)
    
    rollouts = []
    if outputs and len(outputs) > 0:
        for i in range(len(outputs[0].outputs)):
            # The full trace includes the prefix + generated completion
            full_trace_text = f"{previous_steps_text}\n{alternative_step_text}\n{outputs[0].outputs[i].text}"
            rollouts.append(full_trace_text)
            
    return rollouts

def calculate_pns_for_trace(llm_engine, problem, ground_truth, original_trace_text, is_trace_sufficient_fn, generate_counterfactual_fn):
    """
    Calculates the PNS (Probability of Necessity and Sufficiency) score 
    for every step in a single given trace.
    """
    # 1. Check if the original trace even got the answer right
    # If it didn't, Sufficiency (PS) = 0, therefore PNS = 0 across the entire trace.
    if not is_trace_sufficient_fn(original_trace_text, ground_truth):
        return [{"text": step, "pns_score": 0.0} for step in split_into_steps(original_trace_text)]
    
    # PS = 1.0 because the trace was correct.
    ps = 1.0
    
    steps = split_into_steps(original_trace_text)
    scored_steps = []
    
    for i, step in enumerate(steps):
        previous_steps_text = "\n".join(steps[:i])
        
        # 2. Generate the "wrong" alternative step (counterfactual)
        alternative_step = generate_counterfactual_fn(
            llm_engine, 
            problem, 
            previous_steps_text, 
            step
        )
        
        # 3. Roll out the completion from this wrong step 3 times
        cf_rollouts = run_counterfactual_rollouts(
            llm_engine,
            problem,
            previous_steps_text,
            alternative_step,
            num_rollouts=3
        )
        
        # 4. Measure how many of the counterfactual rollouts were ACTUALLY CORRECT
        correct_counterfactuals = sum(
            1 for ro in cf_rollouts if is_trace_sufficient_fn(ro, ground_truth)
        )
        
        # Calculate expected value (Y) of the counterfactual intervention
        expected_y = correct_counterfactuals / len(cf_rollouts) if cf_rollouts else 0.0
        
        # Probability of Necessity (PN) is 1 - E(Y)
        pn = 1.0 - expected_y
        
        # Final PNS = PS * PN
        pns_score = ps * pn
        
        scored_steps.append({
            "text": step,
            "pns_score": round(pns_score, 4)
        })
        
    return scored_steps

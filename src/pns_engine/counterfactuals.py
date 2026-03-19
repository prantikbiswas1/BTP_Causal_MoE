import re

def get_counterfactual_prompt(problem, previous_steps, target_step):
    """
    Creates a prompt that forces the LLM to generate an alternative (counterfactual)
    step that is logically DIFFERENT from the original target step.
    
    Args:
        problem (str): The original math question.
        previous_steps (str): All reasoning steps that occurred before the target step.
        target_step (str): The original reasoning step we want to replace.
        
    Returns:
        str: A formatted prompt for the LLM.
    """
    prompt = f"""<|im_start|>user
You are an expert mathematician simulating a reasoning mistake.
You are given a math problem and the first few steps of a solution.
Then, you are given an "Original Step".

Your task is to write a "Counterfactual Step".
This step MUST contain a logical or arithmetic ERROR that deviates from the Original Step.
It should look like a plausible next step, but it must be mathematically incorrect.

Problem: {problem}

Previous steps leading up to this point:
{previous_steps if previous_steps else "None (This is the first step)"}

Original Step (Do NOT write this):
{target_step}

Write exactly ONE alternative, incorrect step to replace the Original Step:
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def generate_counterfactual_step(llm_engine, problem, previous_steps, target_step, temp=0.7):
    """
    Uses the vLLM engine to generate a counterfactual step.
    (Assumes the llm_engine is a vLLM LLM instance)
    """
    from vllm import SamplingParams
    
    prompt = get_counterfactual_prompt(problem, previous_steps, target_step)
    
    # We only need 1 alternative step, and it shouldn't be very long
    sampling_params = SamplingParams(temperature=temp, max_tokens=100)
    
    # Generate the output (in a real pipeline this would be batched)
    outputs = llm_engine.generate([prompt], sampling_params, use_tqdm=False)
    
    if outputs and len(outputs) > 0:
        return outputs[0].outputs[0].text.strip()
    return ""

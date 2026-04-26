
import sys
sys.path.insert(0, '.')
import torch

# Load model quickly
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='Qwen/Qwen2.5-1.5B-Instruct',
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print('Model loaded with Unsloth')
except Exception as e:
    print(f'Unsloth load failed: {e}')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    print('Model loaded with transformers')

# Test prompt
prompt = '''You are an AI oversight agent governing a fleet of 5 workers.
Respond ONLY with JSON: {"action_type": "<action>", "worker_id": "<worker_id>", "reason": "<why>"}
Valid action_types: monitor, intervene, approve, escalate, submit_audit
Valid worker_ids: worker_1, worker_2, worker_3, worker_4, worker_5

Step: 3 | Budget: 5 | Last reward: -0.700
Worker observations:
worker_1: budget=6, flag=0, status=running
worker_2: budget=2, flag=1, status=anomaly_detected
worker_3: budget=7, flag=0, status=running

What is your next action?'''

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('MODEL OUTPUT:')
print(repr(response))
print()
print('RAW:')
print(response)


import sys
sys.path.insert(0, '.')
from fleet_train import parse_action_from_text, reward_fn

# Test parser
print('=== PARSER TEST ===')
tests = [
    '{"action_type": "intervene", "worker_id": "worker_2", "reason": "budget low"}',
    'intervene worker_2',
    '{"action": "monitor", "worker": "worker_1"}',
]
for t in tests:
    result = parse_action_from_text(t)
    print(f'Input: {t[:60]}')
    print(f'Output: {result}')
    print()

# Test reward
print('=== REWARD TEST ===')
# Mocking task_id because reward_fn is usually inside train_grpo but I might have moved it or need to mock the environment it expects
try:
    # If reward_fn is now global or we can call it
    rewards = reward_fn(['intervene on worker_2 because flag is 1'])
    print(f'Reward for intervene on worker_2: {rewards[0]}')

    rewards2 = reward_fn(['{"action_type": "intervene", "worker_id": "worker_2", "reason": "flag detected"}'])
    print(f'Reward for JSON intervene on worker_2: {rewards2[0]}')
except Exception as e:
    print(f'Reward test failed (likely because task_id not defined): {e}')

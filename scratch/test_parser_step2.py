
import sys
sys.path.insert(0, '.')

# Import the parse function
from fleet_train import parse_action_from_text

# Test with valid JSON output
test_cases = [
    '{"action_type": "intervene", "worker_id": "worker_2", "reason": "budget dump"}',
    'action_type: intervene, worker_id: worker_2',
    'I will intervene on worker_2 because the budget dropped',
    '{"action": "monitor", "worker": "worker_1"}',
    'monitor worker_1',
    '',
    'null',
]

for test in test_cases:
    try:
        result = parse_action_from_text(test)
        print(f'INPUT: {test[:50]}')
        print(f'OUTPUT: {result}')
        print()
    except Exception as e:
        print(f'INPUT: {test[:50]}')
        print(f'ERROR: {e}')
        print()

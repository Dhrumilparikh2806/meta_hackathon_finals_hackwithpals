
import sys
sys.path.insert(0, '.')

# Read fleet_train.py
with open('fleet_train.py', 'r') as f:
    content = f.read()

# Find key functions
import re

# 1. Find parse_action_from_text
match = re.search(r'def parse_action_from_text.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
if match:
    print('=== PARSE_ACTION_FROM_TEXT ===')
    print(match.group()[:800])
    print()

# 2. Find reward_fn
match2 = re.search(r'def reward_fn.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
if match2:
    print('=== REWARD_FN ===')
    print(match2.group()[:800])
    print()

# 3. Find SYSTEM_PROMPT
match3 = re.search(r'SYSTEM_PROMPT\s*=.*?(?=\"\"\"|\Z)', content, re.DOTALL)
if match3:
    print('=== SYSTEM_PROMPT ===')
    print(match3.group()[:400])
    print()

# 4. Find what happens when parse fails
match4 = re.search(r'fallback|default.*action|except.*action', content, re.IGNORECASE)
if match4:
    print('=== FALLBACK/DEFAULT ===')
    print(match4.group())

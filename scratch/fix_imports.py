import os
import re

def update_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Update 'from fleet' and 'import fleet'
    content = content.replace('from fleet.', 'from env.')
    content = content.replace('import fleet.', 'import env.')
    content = content.replace('from fleet import', 'from env import')
    
    # 2. Update Worker 1 import in oversight_env.py
    if 'oversight_env.py' in file_path:
        content = content.replace('import env.environment', 'import env.worker_triage')
        content = content.replace('from env.environment import DataQualityTriageEnv', 'from env.worker_triage import DataQualityTriageEnv')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Target files
targets = [
    'app.py',
    'fleet_train.py',
    'fleet_inference.py',
    'fleet_baseline.py',
]

# Add all files in env/
for f in os.listdir('env'):
    if f.endswith('.py'):
        targets.append(os.path.join('env', f))

# Add all files in tests/
if os.path.exists('tests'):
    for f in os.listdir('tests'):
        if f.endswith('.py'):
            targets.append(os.path.join('tests', f))

for target in targets:
    if os.path.exists(target):
        print(f"Updating imports in {target}...")
        update_imports(target)

print("Imports updated successfully!")

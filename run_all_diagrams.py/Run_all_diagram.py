# run_all_diagrams.py
import subprocess
import os

# Example: run all diagram-generating scripts in the folder
for file in os.listdir('.'):
    if file.endswith('_diagram.py'):
        print(f'Running {file}...')
        subprocess.run(['python', file])
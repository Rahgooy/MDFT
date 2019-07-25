import json
from pathlib import Path
import numpy as np

baseDir = Path('results')


summary = {}
for dir in baseDir.iterdir():
    if dir.is_dir():
        param_name = dir.name
        summary[param_name] = {}
        for s in dir.iterdir():
            if s.is_file() and s.name.endswith(".json"):
                set_name = s.name[:-5]
                with s.open(mode='r') as f:
                    data = json.load(f)
                    summary[param_name][set_name] = np.array([d['mse'] for d in data['results']])

np.set_printoptions(precision=4, suppress=False, linewidth=200)

for p in summary:
    print("*" * 90)
    print(p)
    for s in sorted(summary[p].keys()):
        print(s)
        print(summary[p][s])
        print(f"mean: {summary[p][s].mean():0.5f}")
        print(f"std: {summary[p][s].std():0.5f}")
        print("\n")
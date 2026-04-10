python -c "
content = '''# Contributing

## Setup

\`\`\`bash
git clone https://github.com/svd009/Quantum-Enhanced-Continual-Learning-Fabric-for-Planetary-Climate-Adaptation.git
cd Quantum-Enhanced-Continual-Learning-Fabric-for-Planetary-Climate-Adaptation
pip install -r requirements.txt
\`\`\`

## Branch Strategy

- main        - stable, protected
- dev         - integration branch
- feature/x   - new features
- fix/x       - bug fixes

## Making Changes

1. Branch from main
\`\`\`bash
git checkout -b feature/your-feature
\`\`\`

2. Write tests for new code
3. Make sure all tests pass
\`\`\`bash
pytest tests/ -v
\`\`\`

4. Commit with descriptive messages
\`\`\`bash
git commit -m \"feat: description of what you added\"
\`\`\`

5. Push and open a pull request

## Commit Convention

- feat:     new feature
- fix:      bug fix
- docs:     documentation
- test:     adding tests
- chore:    maintenance
- ci:       CI/CD changes
- refactor: code restructuring

## Running Experiments

\`\`\`bash
# Synthetic data (no API key needed)
python scripts/run_experiment.py --synthetic --n-years 5

# Full ablation
python scripts/run_ablation.py

# Results charts
python notebooks/results_analysis.py
\`\`\`
'''
with open('CONTRIBUTING.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print('done')
"
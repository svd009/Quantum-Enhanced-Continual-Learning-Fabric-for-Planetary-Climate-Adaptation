@"
from setuptools import setup, find_packages

setup(
    name='fedclimate',
    version='0.1.0',
    description='Federated Continual Learning for Climate Forecasting',
    python_requires='>=3.10',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'xarray>=2023.1.0',
        'pyyaml>=6.0',
        'omegaconf>=2.3.0',
        'tqdm>=4.65.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
    ],
)
"@ | Set-Content setup.py
echo "done"
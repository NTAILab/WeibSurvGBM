from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requirements = f.read().splitlines()

setup(
    name='weib_surv_gbm',
    version='1.0.0',
    author="Polytech NTAILab",
    description='Gradient Boosted Parametric Survival Model using XGBoost',
    packages=find_packages(),
    install_requires=install_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",],
    python_requires='>=3.10',
)
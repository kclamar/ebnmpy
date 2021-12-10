from setuptools import setup

requirements = [
    "numpy>=1.21.4",
    "scikit-learn>=1.0.1",
    "scipy>=1.7.3",
]


setup(
    name="ebnmpy",
    version="0.0.1",
    description="Python package for fitting the empirical Bayes normal means (EBNM) model",
    install_requires=requirements,
    extras_require=dict(
        dev=[
            "pre-commit",
            "pytest",
        ],
        report=[
            "jupyter",
            "matplotlib",
            "pandas",
            "seaborn",
        ],
    ),
    python_requires=">=3.8",
    author="Ka Chung Lam",
    url="https://github.com/kclamar/ebnmpy",
)

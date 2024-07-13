from setuptools import setup, find_packages

setup(
    name='ezlmm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy==1.26.4', 'pandas==2.2.2', 'rpy2==3.5.16'],
    authors=[{"name": 'Paradeluxe', "email": "paradeluxe3726@gmail.com"}],
    description='Python interface of Linear Mixed Model analysis in R language (e.g., lmerTest, lme4)',
    license='MIT',
    keywords='Linear Mixed Model',
    url='https://github.com/Paradeluxe/ezLMM',  # 你的项目主页
)


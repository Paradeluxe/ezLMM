from setuptools import setup, find_packages

setup(
    name='ezlmm',
    version='0.4.4',
    packages=find_packages(),
    install_requires=['numpy>=1.26.4', 'pandas>=2.2.2'],
    author='Paradeluxe',
    author_email="paradeluxe3726@gmail.com",
    description='Python interface of Linear Mixed Model analysis in R language',
    license='MIT',
    keywords='Linear Mixed Model',
    url='https://github.com/Paradeluxe/ezLMM',  # 你的项目主页
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",

)


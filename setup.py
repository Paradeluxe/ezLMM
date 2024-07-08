from setuptools import setup, find_packages

setup(
    name='ezlmm',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'cffi==1.16.0',
        'Jinja2==3.1.4',
        'MarkupSafe==2.1.5',
        'numpy==1.26.4',
        'packaging==24.1',
        'pandas==2.2.2',
        'pycparser==2.22',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'rpy2==3.5.16',
        'six==1.16.0',
        'tzdata==2024.1',
        'tzlocal==5.2',
    ],
    # 其他元数据
    author='Paradeluxe',
    author_email='paradeluxe3726.com',
    description='An easy way to use linear mixed model in Python',
    license='MIT',
    keywords='Linear Mixed Model',
    url='http://example.com/MyPackage',  # 你的项目主页
)
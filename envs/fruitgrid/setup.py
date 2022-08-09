from setuptools import setup, find_packages

setup(
    name='fruitgrid',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "gym>=0.17.3",
        "gym-minigrid==1.0.1"
    ],
    url='',
    license='Apache 2.0',
    author='Google Research',
    maintainer=['Manfred Diaz'],
    author_email='diazcabm@mila.quebec',
    description='Multiagent Minigrid from Google Research'
)
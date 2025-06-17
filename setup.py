from setuptools import find_packages,setup
from typing import List

def get_requirements(path:str) -> List[str]:
    '''
    This Function returns the list of requirements
    '''

    requirements=[]

    with open(path) as file_obj:
        requirements=file_obj.readlines()

    requirements = [req.replace('\n','') for req in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements

setup(
    name="Flight Price Predictor",
    author="Vatsalya",
    version='0.0.1',
    author_email='vatsalyabrahmatmaj2@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
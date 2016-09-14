from setuptools import setup, find_packages
from pip.req import parse_requirements

print('Hallo')
install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]
print(reqs)
print(dep_links)
setup(
    name='bb_composer',
    version='0.0.0.dev1',
    description='Composing images from different cam positions',
    long_description='',
    entry_points={
        'console_scripts': [
            'bb_composer = composer.scripts.bb_composer:main'
        ]
    },
    url='https://github.com/gitmirgut/bb_composer',
    author='gitmirgut',
    author_email="gitmirgut@users.noreply.github.com",
    packages=[
        'composer',
        'composer.scripts'],
    install_requires=reqs,
    dependency_links=dep_links,
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ]
)

from setuptools import setup

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

    license='GNU GPLv3',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ]

)

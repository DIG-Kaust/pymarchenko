import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'A bag of Marchenko algorithms implemented on top of PyLops'

# Setup
setup(
    name='pymarchenko',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['geophysics',
              'inverse problems',
              'seismic'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    author='mrava',
    author_email='matteoravasi@gmail.com',
    install_requires=['numpy >= 1.15.0', 'scipy', 'pylops <= 1.18.3'],
    packages=find_packages(exclude=['pytests']),
    use_scm_version=dict(root = '.',
                         relative_to = __file__,
                         write_to = src('pymarchenko/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)

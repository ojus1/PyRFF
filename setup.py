from distutils.core import setup
setup(
    name='py_rff',
    packages=['PyRFF'],
    version='0.2',
    license='MIT',
    description='Random Feature Extraction with Numpy',
    author='Surya Kant Sahu',
    author_email='surya.oju@gmail.com',
    url='https://github.com/ojus1/PyRFF',
    download_url='https://github.com/ojus1/PyRFF/archive/v_02.tar.gz',
    keywords=['Machine Learning', 'Feature Engineering',
              'Artificial Intelligence'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

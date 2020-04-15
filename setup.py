
from skbuild import setup
from setuptools import find_packages

# # python setup.py install --generator "Sublime Text 2 - Unix Makefiles" -- -- -j8
# # python setup.py install  -- -- -j8
package_folder = 'shark'


setup(
    name='shark',
    version='0.0.1',
    description='reinforcement learning project shark',
    author='Aimin Huang',
    author_email='huangepn@gmail.com,huangpen@163.com',
    license='MIT',
    python_requires='>=3.6',
    packages=find_packages(exclude=("test", "test.*", "docs", "docs.*")),  # same as name
    cmake_source_dir="shark",

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='reinforcement learning project pytorch',
#    install_requires=[
#            'gym>=0.15.0',
#            'tqdm',
#            'numpy',
#            'tensorboard',
#            'torch>=1.2.0',
#        ],
)

# print(find_packages())

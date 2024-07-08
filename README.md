In order to run the code from the file 'Code', the installation of the following python packages with the specified minimum versions is necessary:

matplotlib==3.8.0 [https://matplotlib.org/stable/users/release_notes.html]

numpy==1.26.4 [https://numpy.org/install/]

pymor==2023.2.0 [https://pymor.org/]

scipy==1.11.4 [https://scipy.org/install/]

torch==2.3.0+cu121 if CUDA is installed or otherwise torch==2.3.0 [https://pytorch.org/]

The installation of CUDA [https://developer.nvidia.com/cuda-zone] is optional but recommended if available.

Please make sure that ipywidgets is up to date, for example by running: pip install -U ipywidgets

The line of code 'plt.style.use('style.mplstyle')' at the beginning of Code/example.py is commented out to avoid errors that lead to the figures not being plotted. One might try to remove '# ' in front of this line to get figures with the same style as in the thesis.


To replicate the results of this thesis, the code in Code/example.py has to be ran once. Then the following functions return the specified results:

result1(): Figure 4.1, 4.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.11

result2(): FOM-ENOpt results in Table 5.2

result3(): AML-ENOpt results in Table 5.2

result4(): Table 5.3, 5.4 and Figure 5.9

result5(): Figure 5.10

result6(): Figure 5.12

result7(): Figure 5.13

result8(): Figure 5.14, 5.15, 5.16, 5.17, 5.18
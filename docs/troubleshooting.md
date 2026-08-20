# Troubleshooting

## Vipsania reports that no GPU was found

`vipsania annotate` and `vipsania train` print the devices they are going to use when they start.
If they report no GPU although the machine has one, TensorFlow was unable to load its CUDA
libraries and fell back to the CPU, which is far too slow for a genome. Usually another CUDA
installation on the system is being loaded instead of the one that came with TensorFlow. Pointing
the loader at the latter resolves it:

    $ export LD_LIBRARY_PATH=$(python -c "import site, glob, os; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

Inside a conda environment, putting that line into a script in
`$CONDA_PREFIX/etc/conda/activate.d/` makes it permanent. For the same kind of reason, Vipsania
requires `tensorflow<2.20`.

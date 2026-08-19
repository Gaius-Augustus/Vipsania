# Troubleshooting

## Vipsania reports that no GPU was found

`vipsania annotate` and `vipsania train` print the devices they are going to use when they
start. If no GPU is reported
although the machine has one, TensorFlow was unable to load its CUDA libraries and silently fell
back to the CPU. Check it directly:

    $ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

An empty list, together with a message such as `Cannot dlopen some GPU libraries`, means that the
CUDA libraries installed alongside TensorFlow are not on the library search path, or that an older
CUDA installation somewhere on the system is being loaded instead. Rerun with
`TF_CPP_MIN_LOG_LEVEL=0` to see which library failed. Pointing the loader at the libraries that came
with TensorFlow usually resolves it:

    $ export LD_LIBRARY_PATH=$(python -c "import site, glob, os; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

To make this permanent inside a conda environment, put that line into a script in
`$CONDA_PREFIX/etc/conda/activate.d/`.

Vipsania requires `tensorflow<2.20` for the same reason: the 2.21.0 wheel no longer finds the
cuSOLVER library that pip installs next to it, and therefore registers no GPU at all.

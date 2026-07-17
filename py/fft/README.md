# FFT ground truth example

`fft_groundtruth.py` creates a deterministic complex input, uses
`numpy.fft.fft` as the ground-truth FFT API, and cross-checks the result against
an independently implemented direct DFT. Its default 16-point complex input is
also used by `testbench/fft/fft.c` for RVV validation.

## FFT formula

FFT (Fast Fourier Transform) is an efficient algorithm for computing the
Discrete Fourier Transform (DFT). The forward DFT is

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi kn/N},
\qquad k = 0, 1, \ldots, N-1,
$$

where `x[n]` is the time-domain input, `X[k]` is the complex value of frequency
bin `k`, `N` is the transform size, and $j=\sqrt{-1}$.

Using Euler's formula,

$$
e^{-j\theta} = \cos(\theta) - j\sin(\theta),
$$

the DFT can also be written as

$$
X[k] = \sum_{n=0}^{N-1} x[n]
\left[
\cos\left(\frac{2\pi kn}{N}\right)
-j\sin\left(\frac{2\pi kn}{N}\right)
\right].
$$

The inverse transform is

$$
x[n] = \frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}.
$$

FFT and DFT produce the same mathematical result. A direct DFT requires
$O(N^2)$ operations, while a radix-2 FFT requires $O(N\log N)$ operations.
In this example, `numpy.fft.fft` produces the ground truth and `direct_dft()`
implements the forward DFT formula above as an independent cross-check.

Dependency:

```sh
python3 -m pip install numpy
```

Run the validation:

```sh
python3 py/fft/fft_groundtruth.py
```

The command prints `input_real`, `input_imag`, `groundtruth_real`, and
`groundtruth_imag` as C-compatible `float` arrays, followed by a per-frequency
bin table and the direct-DFT cross-check error.

Generate the header consumed directly by the RVV testbench:

```sh
python3 py/fft/fft_groundtruth.py --size 1024 \
    --header testbench/fft/fft_vectors.h
```

Regenerate this header with another power-of-two `--size`, then rebuild the C
testbench to validate a different FFT length.

Save the input and ground-truth arrays for another implementation to consume:

```sh
python3 py/fft/fft_groundtruth.py --output-dir /tmp/fft-groundtruth
```

The generated `fft_input.csv` and `fft_groundtruth.csv` files contain `index`,
`real`, and `imag` columns. The command exits with status 1 if the maximum
absolute error is larger than `--atol` (default `1e-5`).

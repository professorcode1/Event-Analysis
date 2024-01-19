# Benchmark against CoinCalc
Coincalc is the most widely used Open Source library for calculaing ECA. For 25 grid point it took 1 minutes and 12 seconds. In comparison Event-Analysis took ~400 ms. <b>180 times faster</b>. For the entire dataframe with 357 gridpoints it took EA 5.24 seconds. (Coincalc was not benched as it would take 10 days for the same).

The are a couple of reasons behind this
1. The CoinCalc implementation is O(n<sup>2</sup>) while EA implementation is O(n log(n)).
2. The python code is compiled using Numba so it runs at native C speed.
3. Using Numba it is also parallelized. 
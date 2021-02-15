# 时序数据的特征提取

使用 [tsfresh](https://tsfresh.readthedocs.io/)可以方便的提取时序数据的特征。使用tsfresh提供的 `tsfresh.feature_extraction.feature_calculators` 可以方便的计算时序数据的复合特征。这里主要说明一下时序数据的特征信息。

## 1. abs_energy(X)
返回时间序列的绝对能量，该绝对能量是平方值的和:

$$
    E = \sum_{i=1, \dots, n} x^2
$$

```python
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)
```

## 2. absolute_sum_of_changes(x)

返回x中连续变化的绝对值的总和

$$
    \sum_{i=1, \dots, n-1} |x_{i+1} - x_i|
$$

```python

    return np.sum(np.abs(np.diff(x)))
```

## 3. agg_autocorrelation(x, param)

计算聚合函数 $f_{agg}(如方差和均值)$ 处理后的自相关性, $l$ 表示滞后性. 对于滞后 $l$ 的自相关性 $R(l)$ 的定义为:

$$
    R(l) = \frac{1}{(n-l)\sigma^2} \sum_{t=1}^{n-l}(X_t - \mu)(X_{t+l} - \mu)
$$

其中 $X_i$ 是时序数据的值， $n$ 是时序数据的长度。 $\sigma^2$ 和 $\mu$ 是其方差和均值(见 [自相关函数](https://zh.wikipedia.org/wiki/%E8%87%AA%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0))

在一定程度可以衡量数据的周期性质，如果某个 $(l)$ 计算出的值比较大，表示改时序数据具有 $(l)$ 周期性质。

不同的滞后 $(l)$ 的 $R(l)$ 组成一个向量。 这个特征计算器在这个向量上应用一个聚合函数 $f_{agg}$ 其返回结果是:

$$ 
    f_{agg}(R(1), \dots, R(m)) \text{ for } m=max(n, maxlag)
$$

这里的 $maxlag$ 就是第二个需要传入的参数。

**Parameters:**
    - x(numpy.ndarray)- 时序数据
    - param(list) - contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the autocorrelations. Further, n is an int and the maximal number of lags to consider.

**Return:**
    - 该特征的值

**Return type:**
    - float

```python
    def agg_autocorrelation(x, param):
    """
    Calculates the value of an aggregation function :math:`f_{agg}` (e.g. the variance or the mean) over the
    autocorrelation :math:`R(l)` for different lags. The autocorrelation :math:`R(l)` for lag :math:`l` is defined as

    .. math::

        R(l) = \\frac{1}{(n-l)\\sigma^2} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`X_i` are the values of the time series, :math:`n` its length. Finally, :math:`\\sigma^2` and
    :math:`\\mu` are estimators for its variance and mean
    (See `Estimation of the Autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_).

    The :math:`R(l)` for different lags :math:`l` form a vector. This feature calculator applies the aggregation
    function :math:`f_{agg}` to this vector and returns

    .. math::

        f_{agg} \\left( R(1), \\ldots, R(m)\\right) \\quad \\text{for} \\quad m = max(n, maxlag).

    Here :math:`maxlag` is the second parameter passed to this function.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                  (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                  autocorrelations. Further, n is an int and the maximal number of lags to consider.
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, unbiased=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
    return [("f_agg_\"{}\"__maxlag_{}".format(config["f_agg"], config["maxlag"]),
             getattr(np, config["f_agg"])(a[:int(config["maxlag"])])) for config in param]
```
# Event Analysis

### Library for Event Synchronization and Event Coincidence Analysis

This Python library facilitates the calculation of Event Synchronization (ES) and Event Coincidence Analysis (ECA) for event series.

## Understanding ECA and ES
To learn about ECA, read [this](https://arxiv.org/abs/1508.03534) <sup>1</sup>
To learn about ES, read [this](https://aip.scitation.org/doi/10.1063/1.5134012) <sup>2</sup>

## Installation

To install the library, execute the following command:

```
pip install event-analysis
```

### Additional Requirement

To utilize the CUDA method, please install PyCuda separately.

### Usage

Refer to the [Example.ipnby](https://github.com/professorcode1/Event-Analysis/blob/main/Example.ipynb) Jupyter notebook for detailed usage instructions.

```
from EventAnalysis import EventAnalysis

# Initialize the Event Analysis object with your event dataframe
EA = EventAnalysis(event_dataframe)

# Perform Event Synchronization
Q = EA.ES()

# Calculate Event Coincidence Analysis
p_max, p_mean, t_max, t_mean = EA.ECA(time_delta)

# Calculate ECA with p-values
p_max, p_mean, t_max, t_mean, pval_p, pval_t = EA.ECA(time_delta, return_p_values=True)

# Important: Read the documentation for the EA constructor parameter time_normalization_factor before using the library.
```

Please read the notes for the EventAnalysis class before you use this library.

## Documentation

#### EventAnalysis Class
##### Constructor:
```
EventAnalysis(self, event_df, device_Id=None, time_normalization_factor=3600)
```

##### Arguments

+ **event_df**: `pandas.DataFrame`
  + This DataFrame must have the following properties:
    + The index must be a `pandas.DatetimeIndex`.
    + The data type inside the DataFrame must be boolean.
    + Each column represents an event series.
+ **device_Id**: Optional int
  + Specify the ID of the NVIDIA GPU for computation. Defaults to 0 if not provided.
+ **time_normalization_factor**: Optional int
  + The library quantizes time to hours, ignoring minutes and seconds. This behavior can be overridden by changing this argument. Possible values include:
    + `1`: Quantizes to seconds (allows for 69 years of data).
    + `60`: Quantizes to minutes (allows for 4000 years of data).
    + `3600`: (default) Quantizes to hours (allows for 250 centuries of data).
##### Returns:

+ An EventAnalysis object on which you can call the ES and ECA methods.

#### ES Method

```
ES(self, tauMax=np.Inf)
```

##### Arguments

+ **tauMax**: Optional, default is np.Inf
  + Currently not utilized; future functionality will be added. Please open an issue if needed.

##### Return

+ A DataFrame Q of shape N x N, where N is the number of event series. Use Q[event_series_1_name][event_series_2_name] to get the Event Synchronization between two series.

#### ECA Method
```
ECA(self, Delta_T_obj, tau=0, return_p_values=False, pValFillNA=True)
```

##### Arguments

+ **Delta_T_obj**: Required
  + Must be a `datetime.timedelta` object. It is quantized using the same time_normalization_factor as passed to the constructor.
+ **tau**: Optional, default is 0
  + A parameter of the ECA algorithm. Refer to the associated paper for details.
+ **return_p_values**: Optional, default is False
  + If set to True, it calculates the p-values for each combination of event series and returns them.
+ **pValFillNA**: Optional, default is True
  + Replaces NaN p-values with 1.

#### Return

+ `EC_p_max`
+ `EC_p_mean`
+ `EC_t_max`
+ `EC_t_mean`
+ `pval_precursor` (optional): Corresponds to the p-value of `EC_p_max` and `EC_p_mean`.
+ `pval_trigger` (optional): Corresponds to the p-value of `EC_t_max` and `EC_t_mean`.

**Notes**: The paper outlines the conditions for calculating p-values for ECA results:

1) N<sub>a</sub> >> 1 and N<sub>b</sub> >> 1

2) ΔT << T/N<sub>a</sub>

Here, N<sub>a</sub> and N<sub>b</sub> represent the number of events in series A and B, respectively. Δ𝑇 should be sufficiently less than the overall time.

#### ECA_vec

```
ECA_vec(self, Delta_T_objs, taus=0, return_p_values=False, pValFillNA=True)
```

##### Arguments

+ **Delta_T_objs**: Required
  + A list of datetime.timedelta objects.
+ **taus**: Optional, default is 0
  + Can be an integer, list, or numpy array of integers, matching the length of Delta_T_objs. If an integer is provided, all ECAs use that value.
+ **return_p_values**: Same as ECA.
+ **pValFillNA**: Same as ECA.

##### Returns

+ A generator that yields the ECA result for each corresponding pair of `Delta_T_objs[i]` and `taus[i]`.

## CUDA Support

All three methods can be executed on an NVIDIA GPU by appending `_Cuda` to their names: `ES_Cuda`, `ECA_Cuda`, `ECA_vec_Cuda`. Each requires an additional named parameter, block.

+ **ES_Cuda(..., block=None)**
  + Defaults to `(16, 8, 1)`. Any tuple of the form (x, y, 1) where 𝑥 × 𝑦 is a multiple of the GPU's warp size is valid.
+ **ECA_Cuda(..., block=None)**
+ **ECA_vec_Cuda(..., block=None)**
  + Defaults to (32, 1, 1). Similar to ES_Cuda.

**Notes**: The p-values returned by the GPU may differ slightly from those calculated by the CPU due to floating-point precision. Setting epsilon to 10<sup>−2</sup> should minimize these discrepancies.

### References

1. Event coincidence analysis for quantifying statistical interrelationships between event time series - Jonathan F. Donges, Carl-Friedrich Schleussner, Jonatan F. Siegmund, and Reik V. Donner

2. Frederik Wolf, Jurek Bauer, Niklas Boers, and Reik V. Donner , "Event synchrony measures for functional climate network analysis: A case study on South American rainfall dynamics", Chaos 30, 033102 (2020) https://doi.org/10.1063/1.5134012

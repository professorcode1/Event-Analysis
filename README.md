# Event Analysis
### Library to find Event Synchronization and Event Coincidence Analysis
-------------------------------------------------------
The python library to calculate **Event Synchronization** and **Event Coincidence Analysis** for event series. 

To learn about ECA, read [this](https://arxiv.org/abs/1508.03534) <sup>1</sup>

To learn about EA, read [this](https://aip.scitation.org/doi/10.1063/1.5134012) <sup>2</sup>

To install the library 
```
    pip install event-analysis
```
To use the Cuda method install PyCuda separately. 
### Usage 
Look at the [Example.ipnby](https://github.com/professorcode1/Event-Analysis) jupyter notebook to see the detailed usage. 
```
    from EventAnalysis import EventAnalysis
    EA = EventAnalysis(event_dataframe)
    Q = EA.ES() 
    p_max, p_mean, t_max, t_mean = EA.ECA(time_delta)
    p_max, p_mean, t_max, t_mean, pval_p, pval_t = EA.ECA(time_delta, return_p_values = True)
```
Please read the notes for the EventAnalysis class before you use this library. 
## Documentation
-------------------------------------------------------
#### EventAnalysis( self , event_df , device_Id = None)
##### Arguments
1) event_df : pandas dataframe. This dataframe must have
    1) The index must be an array of python datetime objects of ascending order*
    2) The data inside the dataframe must be of type boolean 
    Each column of this Dataframe will be an event series. 
2) device_Id : if you have an multiple Nvidia GPU's then you can use this to specify the ID of the GPU you want the computation to run on. Reverts to 0 if not provided. 
##### Return 
1) EventAnalysis Object : On which you can call the ES, ECA method.

**Notes**
* The index of the dataframe must be a time series (datetime objects in ascending order). This time series must satisfy 2 properties 
    1) The longest time difference must be less than 68 years.  
    2) The difference b/w any two times must be a multiple of 1 hour. This is because internally the library quantises the time series to hours. If this condition is not met, the minutes and seconds will be ignored.

i.e.
```
    (timeseries[-1] - timeseries[0]).years < 68
    for all i,j:
        (timeseries[i] - timeseries[j]).minutes = 0
        (timeseries[i] - timeseries[j]).seconds = 0
```
If you have a use case that conflicts with these properties open an issue. Or you may fix them yourself and open a PR. Contributions are much appretiated!

-------------------------------------------------------
#### ES(self, tauMax = np.Inf)
##### Arguments
1) tauMax (default = np.Inf): currently the code doesn't use this. Functionality to use this will be added in future release.
##### Return 
1) Q : a dataframe of N x N where N is the number of event series (number of columns in the dataframe given to the contructor ). You can do `Q[event_series_1_name][event_series_2_name]` to get those 2 event series Event Synchronization.
-------------------------------------------------------

#### ECA(self, Delta_T_obj, tau = 0, return_p_values = False, pValFillNA = True)
##### Arguments
1) Delta_T_obj : must be a python datetime.timedelta object. Only total number of hours is used. So minutes and everything below will be ignored. 
2) tau (default 0) : a parameter of the ECA algorithm. You can read it in the paper
3) return_p_values(default false) : if set to true will calculate the p-values for each combination of event series as described in paper and return them as well. *
4) pValFillNA (default true) : because of the method used to calculate p-value internally some values may be nan. If this is true it will replace them with 1. 
#### Return
1) EC_p_max 
2) EC_p_mean
3) EC_t_max
4) EC_t_mean
5) pval_precursor[optional] : correspondes to the p-value of the EC_p_max and EC_p_mean 
6) pval_trigger[optional] : correspondes to the p-value of the EC_t_max and EC_t_mean 

All of them are described in the paper.
##### Notes
The paper gives a formula for calculating P-Value of the result of ECA on 2 event series A and B as long as 2 conditions are satisfied
1) N<sub>a</sub> >> 1 and N<sub>b</sub> >> 1
2) Delta_T << T / N<sub>a</sub>

i.e. The number of events on both series must be **sufficiently** greater than 1 and delta_t must be **sufficiently** less than Overall time(`(timeseries[-1] - timeseries[0])`) / Number of events on A . What qualifies as **sufficiently** is for the user to decide.

-------------------------------------------------------
#### ECA_vec(self, Delta_T_objs, taus = 0, return_p_values = False, pValFillNA = True)
##### Arguments 
1) Delta_T_objs : must be a python list of python datetime.timedelta objects 
2) taus (default 0) : must be an int, a list of ints or an numpy array of ints of the same length as Delta_T_objs. If an int is provided all ECA's are called using that one value. 
3) return_p_values : same as ECA
4) pValFillNa : same as ECA
##### Returns 
1) A generator which on the i'th yeild ECA called on Delta_T_objs[i], taus[i].
-------------------------------------------------------
## Cuda
All 3 method can be called on Nvidia GPU by adding `_Cuda` to their name
i.e `ES_Cuda`, `ECA_Cuda`, `ECA_vec_Cuda`. All of them take one additional named parameter `block`.
#### ES_Cuda(..., block = None)
When block is set to none it defaults to the value of (16, 8, 1). During experimentation it was observed this is the optimal value. However any tuple of the form (x, y, 1) where x * y is a multiple of the GPU's Warp size is valid (Nvidia has yet to make a GPU which doesn't have 32 as it warp size).
#### ECA_Cuda(..., block = None), ECA_vec_Cuda(..., block = None)
When block is set to none it defaults to the value of (32, 1, 1). During experimentation it was observed this is the optimal value. However any tuple of the form (x, 1, 1) where x is a multiple of the GPU's Warp size is valid (Nvidia has yet to make a GPU which doesn't have 32 as it warp size).
##### Notes 
The p_values returned by the GPU maybe sligthly different from the once by CPU. Although the methods are identical across both, it is caused by GPU floating points operation returning results sligtly different than CPU. However they wont be much different. Setting Epsilong to 10e-2 should make them allclose.  

-------------------------------------------------------

### References 
1. Event coincidence analysis for quantifying statistical interrelationships between event time series - Jonathan F. Donges, Carl-Friedrich Schleussner, Jonatan F. Siegmund, and Reik V. Donner
2. Frederik Wolf, Jurek Bauer, Niklas Boers, and Reik V. Donner , "Event synchrony measures for functional climate network analysis: A case study on South American rainfall dynamics", Chaos 30, 033102 (2020) https://doi.org/10.1063/1.5134012

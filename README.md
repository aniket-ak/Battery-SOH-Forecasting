GPR
===========

Battery SOH forecasting using GPR. Reference from this [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjqmJ_855bvAhUVgOYKHR4lByAQFjAAegQIBBAD&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS0378775317306250&usg=AOvVaw1P31v8I4zlOJacXNwqw_xP).


Description
===========

Automotive industry has undergone significant changes with innovations like electrification and Autonomous Driving (ADAS). Electrificaiton, in general talks about many aspects but electrifying powertrain is the main focus area. Electrification is supposed to reduce the carbon footprint of vehicles, or at least isolate them away from cities where the major population resides around the world. According to this report from [EV Volumes](https://www.ev-volumes.com/country/total-world-plug-in-vehicle-volumes/) database, the YoY growth of EV sales showed average monotinic increasing trends despite the COVID situation economically.

<img src="https://www.ev-volumes.com/wp-content/uploads/2021/01/WW-A-12-2020.png" width="800" />


The Li-ion battery (LiB) technology is the backbone of electrified powertrain. A strong and reliable LiB technology entails early and robust adaptation of electrified powertrain. To that end, LiB development has many challenges. One among these is the prediction of Age of LiB cells as it is a multiscale and multiphysics problem at core. If the LiB cell capacity history is known, forecasting the State of Health (SoH) is a challenge, because this is governed by many external and user specific aspects. 

In this [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjqmJ_855bvAhUVgOYKHR4lByAQFjAAegQIBBAD&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS0378775317306250&usg=AOvVaw1P31v8I4zlOJacXNwqw_xP), the authors have tried to make use of Gaussian Process Regression (GPR) for this forecasting task. This notebook attempts to recreate the results from the paper. 

This 

# Table of content

> 1. Basic Single Output GP Results
>> 1.1. Kernel Function Selection
>>
>>1.2. Kernel Function Decomposition
>>
>>1.3. Short-term Lookahead Prediction
>>
>>1.4 Remaining Useful Life Prediction
> 2. Encoding exponential degradation via EMFs Results
> 3. Capturing Cell-to-Cell Correlations via Multi-output GPs Results

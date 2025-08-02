# PeFNN
We consider solving complex spatiotemporal dynamical systems governed by partial differential equations (PDEs) using frequency domain-based discrete learning approaches, such as Fourier neural operators. Despite their widespread use for approximating nonlinear PDEs, the majority of these methods neglect fundamental physical laws and lack interpretable nonlinearity. We address these shortcomings by introducing Physicsembedded Fourier Neural Networks (PeFNN) with flexible and explainable error control. PeFNN is designed to enforce momentum conservation and yields interpretable nonlinear expressions by utilizing unique multi-scale momentum-conserving Fourier (MC-Fourier) layers and an element-wise product operation. The MC-Fourier layer is by design translation- and rotationinvariant in the frequency domain, serving as a plug-and-play module that adheres to the laws of momentum conservation. PeFNN sets a new state-of-the-art in solving widely employed spatiotemporal PDEs, particularly for long-term predictions, and generalizes well across spatial resolutions. Further, we establish a flood forecasting benchmark dataset, showcasing the superior performance of PeFNN in complex real-world applications, such as large-scale flood dynamics modeling and forecasting.

# Usage
## Experiments for Spatiotemporal PDEs
After configuring the file path, run the code:

**python experiments_NS_T20.py for the NS experiment from  T=10 to T=20**
   
**python experiments_NS_T30.py for the NS experiment from  T=10 to T=30**

**python experiments_NS_T40.py for the NS experiment from  T=10 to T=40**

**python experiments_SWE.py for the SWE experiment**
   
**python experiments_Flood_T20.py for the flood simulation with T=20**

**python experiments_Flood_T40.py for the flood simulation with T=40**

## Experiments for the super-resolution experiments
Setting super=True, and run:

**python experiments_NS_T30.py for the super-resolution experiment of NS dataset**

**python experiments_SWE_SR.py for the super-resolution experiment of SWE dataset**
   
**python experiments_NS_T30.py for the NS experiment from  T=10 to T=30**

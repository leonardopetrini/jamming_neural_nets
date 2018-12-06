# Jamming transitions in Neural Nets

Study of neural networks around jamming transition at maximum storage capacity

 - `main*.py` files are examples of how to execute the code
 
 - `alice.py` contains   all the useful functions
 - `classes.py` containes the definition of the following classes:
      - **DatasetClass:** define a dataset and make operations on it, in partucular allows to define datasets from network layers' activations
      
      - **ModelClass:** class from which all models inherit (Triangular and Rectangular nets are employed)
      
      - **ResultsClass:** store results for each experiment
      
 - `experiment.py` defines the experiment and finds the jamming transition 

## Conventions

 - `P` denotes the number of samples in a dataset
 - `d` denotes the input dimension
 - `L` denotes the number of layers in a net
 - `N` denotes the total number of parameters in a net
 - `r = P/N` is the order parameter of the jamming transition for NNs 
 - 

:snowflake:

# Palm Oil Price Regression using Backpropagation and Particle Swarm Optimization

Final project Swarm Intelligence Course

## A Brief Repository Structure

```text
prediction-using-backpropagation-pso/
    core/                               # core algorithm module
        __init__.py
        backpropagation_neural_net.py
        particle_swarm_optimization.py

    result/                             # result of hyperparameter test
        ...-.png

    __init__.py
    dataset_minyak_kelapa_sawit.csv     # dataset used in algorithm testing
    dataset_timeseries.csv              # just an example of timeseries dataset
    main.py                             # implementation code of PSOxBackpro
    manualisasi projek akhir.xlsx       # manual version of algorithm to validating
    README.md                           # this file
    test_case.ipynb                     # hyperparam test in jupyter notebook
    test.py                             # just test for main.py
```

## Description

Backpropagation is one of neural network algortihm for supervised learning. The difference betwwen a normal Multi-Layer Perceptron are Backpropagation use Gradient-descent as learning error minimalization passed backward from output back to input layer. As usual Multi-Layer Perceptron algorithm, Backpropagation is using a random initial weight before training progress started. Because using a random initial value, the result of Backpropagation are varying. They can produce a good accuracy value if algorithm find the global optima or low accuracy id algorithm stuck in local optima.

Particle Swarm Optimization (PSO) is one of optimization algorithm that mimics the group of bird movement. A bird in group of bird will try to gather some information of food position. Then they're share a food position information each other. Same as group of bird, PSO using some solution set (known as particle) to run some fitness function. One of the best particle (which is the highest fitness) will be an information to other particle to start finding solution.

In this project, PSO will help to find a best initial weight of Backpropagation. So, this project expect to make PSO can give a result that close to global optima.

## How to Read The Code

The `backpropagation_neural_net.py` and `particle_swarm_optimization.py` files in directory `core/` are the general class of both algorithm. The `main.py` is custom implementation from `core/` to make PSO can optimize the Backpropagation. So, the `main.py` file is combination between both algorithm in `core/` directory.

Using method overriding from parent class in `core/` directory. Here is the following modified method to make PSO can optimize Backpropagation (file: `main.py`):

- initWeight() which using a particle dimension as initial weight of Backpropagation.
- calculate_fitness() which run backpropagation training and get MAPE score as fitness value of PSO
- PSOxBackpro() which using Backpropagation Particle object as particle initialization population

## Test Scenario

```text
PSO Max Iteration                   : [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
PSO Particle Pop Size               : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
PSO c1 & c2 Value Combination       : [(2.5, 0.5), (2, 1), (1.5, 1.5), (1, 2), (0.5, 2.5)]
PSO w Value                         : [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PSO K-Velocity Clamping             : [0.2, 0.4, 0.6, 0.8, 1]
```

## Test Constraint

Backpropagation Param:

```text
Input Layer         : 5
Hidden Layer        : 3
Output Layer        : 1
Alpha               : 0.01
Epoch               : 40
```

As we can see on text above, Backpropagation using 5 input layer and 3 hidden layer. So, the initial weight between input layer and hidden layer is 15. The PSO will try to optimize the initial weight random value before Backpropagation training started. So, according to the test constraint, The dimension of particle of PSO is 15.

## Result

- PSO Max Iteration
  ![Control Variable: pop_size=10, c1=1, c2=1, w=1, k=1](https://github.com/desenfirman/prediction-using-backpropagation-pso/raw/master/result/iter_test.png)
- PSO Particle Pop Size
  ![Control Variable: t_max=25, c1=1, c2=1, w=1, k=1](https://github.com/desenfirman/prediction-using-backpropagation-pso/raw/master/result/par_pop_size_test.png)  
- PSO c1 & c2 Value Combination
  ![Control Variable: t_max=25, pop_size=40, w=1, k=1](https://github.com/desenfirman/prediction-using-backpropagation-pso/raw/master/result/c_value_test.png)  
  
  Note: 0 value is representation for combination of 2.5;0.5 c-value pair. So does next for other value until the value is 4, which is 0.5;2.5 c-value pair.
  
- PSO w Value
  ![Control Variable: t_max=25, pop_size=40, c1=0.4, c2=2.5, k=1](https://github.com/desenfirman/prediction-using-backpropagation-pso/raw/master/result/w_value_test.png)  
- PSO K-Velocity Clamping
  ![Control Variable: t_max=25, pop_size=40, c1=0.4, c2=2.5, w=0.5](https://github.com/desenfirman/prediction-using-backpropagation-pso/raw/master/result/k_value_test.png)
 
Best PSO param from HyperParameter test is:
```
t           =   25
pop_size    =   40
c1 & c2     =   2.5;0.5
w           =   0.4
k           =   1
```

Full testing code is in jupyter notebook `test_case.ipynb`.

## By

```text
1. Ageng Wibowo             155150200111269
2. Dese Narfa Firmansyah    155150201111153
3. Fanny Aulia Dewi         155150201111070
```

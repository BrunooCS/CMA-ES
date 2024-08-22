# CMA-ES for Policy Optimization in Gym Environments

This repository implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for optimizing a policy network in OpenAI Gym environments, specifically targeting the `MountainCarContinuous-v0` enviroment.

<video src="videos/test_policy-episode-0.mp" width="600" height="400" controls></video>

## Overview

The project consists of a Jupyter notebook that demonstrates the following key components:

### 1. Environment Setup

The notebook begins by importing necessary libraries and creating the Gym environment. The `MountainCarContinuous-v0` environment is used for training the policy network. The state and action dimensions are extracted from the environment to initialize the policy network.

### 2. Policy Network Definition

A simple feedforward neural network is defined using PyTorch. The `PolicyNetwork` class consists of:
- An input layer that takes the state dimension.
- A hidden layer with ReLU activation.
- An output layer that produces actions bounded between -1 and 1 using the `tanh` activation function.

### 3. Policy Evaluation

The `evaluate_policy` function evaluates the performance of the policy network given a set of parameters. It runs the policy in the environment, accumulates rewards, and returns the total reward and the number of steps taken.

### 4. CMA-ES Training Loop

The core of the optimization process is implemented in the `train_cma_es` function. This function:
- Initializes the CMA-ES algorithm with the policy network's parameters.
- Runs a training loop for a specified number of generations, generating candidate solutions and evaluating their performance.
- Collects statistics on the best, worst, and mean rewards for each generation.
- Implements a stopping criterion based on achieving a predefined reward threshold.

### 5. Results Visualization

After training, the notebook generates an interactive plot using Plotly to visualize the performance of the policy network over generations. The plot displays the best, worst, and mean rewards, providing insights into the optimization process.

### 6. Policy Testing

The notebook includes a section for testing the trained policy. The best parameters are loaded into the policy network, and the policy is evaluated in the environment. The results, including total reward and steps taken, are printed, and a video of the policy in action can be generated.

## Conclusion

This project demonstrates the application of CMA-ES for optimizing a policy network in continuous action spaces. The notebook serves as a comprehensive guide to understanding the implementation and performance of the policy in the specified Gym environments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

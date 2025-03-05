## Introduction

This repository provides a PyTorch implementation of Forward-Backward Multi-Objective Actor-Critic (**FB-MOAC**) algorithm that can solve *mult-objectuve* sequential decision making problems with **Forward-Backward Markov Decisoin Processes (FB-MDPs)**, see figure below.
It can be used also for standard MDP problems, but the htper-parameters need to be fine-tuuned.

|  An examplery illustration of a FB-MDP.  |
| :-------------------------:|
| <img src="images/FB-MDPv3.png" alt="Alt Text" style="width:400px;"> |

The diagram of FB-MOAC algorithm is shown below.
|  Diagram of FB-MOAC algorithm |  Forward-Backward Multi-Objective Optimization of FB-MOAC  |
| :-------------------------:| :-------------------------:|
|  <img src="images/fwbw-moac-1.png" alt="Alt Text" style="width:400px;">  |  <img src="images/fwbw-moac-2.png" alt="Alt Text" style="width:300px;"> |
| The FB-MOAC algorithm consists **forward evaluation**, **backward evaluation** and **bidirectional learning** steps. During the first two steps, the forward and backward dynamics are evaluated, using forward/backward critics, and the resulting experiences are buffered. By a proper chronological order, the policy distribution is optimized in the bidirectional learning step based on the experiences of both forward and backward dynamics and using a **forward-backward multi-objective learning**. The algorithm is additionally equipped with an add-on **episodic MCS-average** to boost the convergence to Pareto-optimal solutions. | This step first computes the vector-valued gradients of forward and backward objectives, then compute the descent direction q(\.) to ensure that all rewards increase simultaneously,  and finally update the parameters of actor network based on q(.).|


## Usage
- To train the RL agent on a FB-MDP: run `train.py`
- To test a preTrained network : run `test.py`

  ### Experiments:
  Notice that current (standard) RL baselines/experiments are not applicable to forward-backward MDPs. Hence, we go beyond them to assess our algorithm on real-world problems characterized by FB-MDPs.
  For this, we have provided two large-scale experiments falling within FB-MDP frameworks.
  The first experiment is a edge-cashing problem in the context of communication networkings,
  and the second experiment is a computation offloading problem in the domain of cloud computing systems. 
  These experiments are provided in the folder [*experiment*]. Moreover, there is a [Readme] in that folder that explains these experiments and thier hyper-parameters.

  For the edge caching experiment, please uncomment the syntax **from environments.EdgeCaching import NetEnv** and for the computation odffloading experiment uncomment the syntax **from environments.ComputationOffloading import NetEnv**
  in the train.py or test.py.

### Algorithm hyperparameters:
-    **Print_freq**        :                        The frequency based on which the training results should be printed. (after how many episodes).
 -   **Save_model_freq**    :                       The frequencyt based on which  the parameters of model should be saved.
  -  **AverageFrequency**   :                       The frequency based on which  the cumulative rewards should be averagd for the printing and logging purposes.
  -  **N_MCS**             :                        Number of Monte-Carlo Samples for the *episodic MCS-average* add-on.
   - **EpisodeNumber**   :                          Number of training episodes.
  -  **TimeSlots**         :                        Number of time-steps in each episode.
   - **LearningRate**    :                          Learning-Rate of the FB-MOAC algorithm, for *multi-objective actor* and *forward/backward critics*.
   - **SmoothingFactor** :                          The smoothing factor of the episodic MCS-average  add-on.
  -  **DiscountFactor**   :                         Discount-factor related to the cumulative rewards.
  -  **PreferenceCoeff**                            Preference parameter to extract a Pareto-front.

##### Note :
  - For each environment, the hyper-parameters need fine-tuning. FB-MOAC can also be used for forward-only multi-objective MDP problems.


## Results
### (1) Edge-Cahing  Experiment.
please refer to the Readme file inthe environment folder to see a brief explanation for this experiment. 
Full details are given in the paper. 

|Obtained Pareto-set of FB-MOAC for edge-caching experiment
|:-------------------------:|
|  <img src="images/Results/multiobjective_comparison_preferences.png" alt="Alt Text" style="width:800px;"> | 


| Training Comparison of FB-MOAC against PPO and a multi-objective A2C | Comparison of Final Solutions of FB-MOAC against PPO and a multi-objective A2C | 
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/FBMOAC_FA2C_PPO.png" alt="Alt Text" style="width:400px;"> |  <img src="images/Results/test-multicast1.png" alt="Alt Text" style="width:400px;"> | 


### (2) Computation-Offloading  Experiment
please refer to the Readme file inthe environment folder to see a brief explanation for this experiment. 
Full details are given in the paper. 


| Performance of solution of FB-MOAC for computation offloading experiment  | Histogram of solution of FB-MOAC for computation offloading experiment  |
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/performance-offload-fb-moac.png" alt="Alt Text" style="width:400px;"> | <img src="images/Results/offload-fb-moac.png" alt="Alt Text" style="width:400px;"> |


| Comparison of solution of PPO for computation offloading experiment  | Comparison of solution of A2C for computation offloading experiment  |
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/offload-f-ppo.png" alt="Alt Text" style="width:400px;"> |  <img src="images/Results/offload-f-a2c.png" alt="Alt Text" style="width:400px;"> | 


| Comparison of solution of PPO for computation offloading experiment  | Comparison of solution of A2C for computation offloading experiment  |
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/Pareto-Front.html" alt="Alt Text" style="width:400px;"> |



## Dependencies
Trained and Tested on:
```
Python 3.11
PyTorch
NumPy
scipy.special
```
Training Environments 
```
Edge-caching
Computation-offloading
```
Graphs
```
matplotlib
```


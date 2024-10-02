# Reinforcement Learning and Fuzzy Logic Modelling for Personalized Dynamic Treatment
This repository integrates fuzzy logic and reinforcement learning to create personalized and optimal dynamic treatment strategies. The approach is applied to a [validated fuzzy model](https://academic.oup.com/bioinformatics/article/36/7/2181/5637225 "Paper that describes the model.") that simulates cell death processes in oncogenic K-ras cancer cells under progressive glucose depletion. The original code of the fuzzy model is [here](https://github.com/sspola/DynamicFuzzyModels "DFM code").

## Requirements
Before running the code, ensure you have all the necessary dependencies installed. You can install them by running:
```
  pip install -r requirements.txt
```
## Running Experiments
Each experiment is self-contained within its respective directory, focusing on various aspects of treatment strategy optimization:

**Directory Structure**
- *first_experiment*: Focuses on maximizing apoptosis over a 12-hour period.
- *second_experiment*: Optimizes both apoptosis (maximization) and necrosis (minimization) over 12 hours.
- *third_experiment*: Investigates the impact of treatment intervals on cell death processes.
  
To replicate an experiment, navigate to the corresponding directory and execute the relevant Python script. For example, to run the experiment involving Q-functions in the first experiment, use the following commands:
```
cd first_experiment
python Q-functions.py
```
Repeat this process for other experiments by switching to their respective directories and running their scripts.

## Create a new dataset
To create a new dataset, you can modify the data generation process by adjusting the frequencies. Uncomment lines 26/27/28 in *programmed_cell_death.py* according to your interest. After making the necessary adjustments, run the script as follows:
```
  python programmed_cell_death.py
```




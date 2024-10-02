
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from progammed_cell_death import get_fs, set_initial_state, time_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")

fuzzyData = pd.read_excel('../second_experiment/dataset_12h_apo_nec.xlsx', engine='calamine')
fuzzyData = shuffle(fuzzyData, random_state=42)

ALPHA = 0.1
GAMMA = 0.97


def get_history(data, stage_now):
    """
    Returns the history of the patient's.

    Returns:
        list: The history of the patient's treatment.
    """
    PKA = data["PKA"].values
    C1_H = [data["C1_" + str(i)].values for i in range(stage_now + 1)]
    UPR_H = [data["UPR_" + str(i)].values for i in range(stage_now + 1)]
    if (stage_now == 0):
        T_H = []
    else:
        T_H = [data["Treatment_" + str(i)].values for i in range(stage_now)]
    Y_H = [data["Apoptosis_" + str(i)].values for i in range(stage_now)]
    Y2_H = [data["Necrosis_" + str(i)].values for i in range(stage_now)]
    result = [PKA] + C1_H + UPR_H + T_H + Y_H + Y2_H
    return result


actions = [0, 1, 2, 3]
states = ["PKA", "C1", "UPR"]
steps = ["t1", "t2", "t3", "t4", "t5", "t6"]
states_for_PS = ["PKA", "C1", "UPR"]
dyn_states = ["C1", "UPR"]

# Estimate treatment and outcome models at each stage
n_samples = fuzzyData.shape[0]
n_stages = 6

psi = np.zeros((n_samples, n_stages, 3))
AIPW_estimators = [DecisionTreeClassifier(max_depth=6) for _ in range(n_stages)]

treatment_models = [LogisticRegression(multi_class='multinomial', max_iter=250, penalty='l2', l1_ratio=0.1) for _ in
                    range(n_stages)]

outcome_models_apop = [RandomForestRegressor(max_depth=5, n_estimators=300, n_jobs=-1) for _ in range(n_stages)]
outcome_models_necr = [RandomForestRegressor(max_depth=5, n_estimators=300, n_jobs=-1) for _ in range(n_stages)]


PO = np.zeros((n_samples))

PO_apo = np.zeros((n_samples))
PO_necr = np.zeros((n_samples))
for stage in reversed(range(6)):
    H = get_history(fuzzyData, stage)
    Y = fuzzyData["Apoptosis_" + str(stage + 1)].values
    Y2 = -1* fuzzyData["Necrosis_" + str(stage + 1)].values

    Y2 = (Y2 - np.mean(Y2))/np.std(Y2)
    Y = (Y - np.mean(Y)) / np.std(Y)
    A = fuzzyData["Treatment_" + str(stage)].values

    # Treatment model (e.g., logistic regression)
    treatment_model = treatment_models[stage]
    treatment_model.fit(np.column_stack([H]).T, A)
    treatment_models.append(treatment_model)
    propensity = treatment_models[stage].predict_proba(np.column_stack([H]).T)
    #
    #    # Outcome model (e.g., linear regression)
    outcome_model_apop = outcome_models_apop[stage]
    outcome_model_apop.fit(np.column_stack([H + [A]]).T, (Y + PO_apo))
    outcome_models_apop.append(outcome_model_apop)

    outcome_model_necr = outcome_models_necr[stage]
    outcome_model_necr.fit(np.column_stack([H + [A]]).T, (Y2 + PO_necr))
    outcome_models_necr.append(outcome_model_necr)

    print("Outcome model score:", outcome_model_apop.score(np.column_stack([H + [A]]).T, (Y + PO_apo)))
    print("Outcome model score:", outcome_model_necr.score(np.column_stack([H + [A]]).T, (Y2 + PO_necr)))
    print("Treatment model score:", treatment_model.score(np.column_stack([H]).T, A))

    PO1 = outcome_model_apop.predict(np.column_stack([H + [np.ones(n_samples)]]).T)
    PO2 = outcome_model_apop.predict(np.column_stack([H + [np.ones(n_samples) * 2]]).T)
    PO3 = outcome_model_apop.predict(np.column_stack([H + [np.ones(n_samples) * 3]]).T)
    PO_set_apo = [PO1, PO2, PO3]

    PO1 = outcome_model_necr.predict(np.column_stack([H + [np.ones(n_samples)]]).T)
    PO2 = outcome_model_necr.predict(np.column_stack([H + [np.ones(n_samples) * 2]]).T)
    PO3 = outcome_model_necr.predict(np.column_stack([H + [np.ones(n_samples) * 3]]).T)
    PO_set_necr = [PO1, PO2, PO3]

    for a in range(3):
        if (a + 1) in treatment_models[stage].classes_:
            a_index = np.where(treatment_models[stage].classes_ == (a + 1))[0][0]
            psi[:, stage, a] = ((A == (a + 1)) * (Y + Y2 + PO_apo + PO_necr) / propensity[:, a_index] +
                                1 - (1 * (A == (a + 1)) / propensity[:, a_index])) * (PO_set_apo[a] + PO_set_necr[a])
    max_a = np.argmax(psi[:, stage, :], axis=1) + 1

    #param_grid = {'max_depth': [6, 7, 8, 9, 10]}
    #grid_search = GridSearchCV(AIPW_estimators[stage], param_grid, cv=10, n_jobs=-1, verbose=3)
    #grid_search.fit(np.column_stack([H]).T, max_a)
    #AIPW_estimators[stage] = grid_search.best_estimator_

    AIPW_estimators[stage].fit(np.column_stack([H]).T, max_a)
    tree_estimates = AIPW_estimators[stage].predict(np.column_stack([H]).T)
    max_PO_apo = [PO_set_apo[tree_estimates[i] - 1][i] for i in range(n_samples)]
    max_PO_necr = [PO_set_necr[tree_estimates[i] - 1][i] for i in range(n_samples)]


    PO_apo = [GAMMA * + max_PO_apo[i] for i in range(n_samples)]
    PO_necr = [GAMMA * max_PO_necr[i] for i in range(n_samples)]
    print("Stage: ", stage)




def predict(stage, H):
    """
    Predicts the treatment and outcome at a given stage.

    Args:
        stage (int): The stage at which to make the prediction.
        H (list): The history of the patient's treatment.

    Returns:
        tuple: The predicted treatment and outcome.
    """
    A = AIPW_estimators[stage].predict(np.column_stack([H]).T)

    return A



## Test the agent ##
def set_action(action, FS):
    """
    Sets the variables in the fuzzy system (FS) based on the given action.

    Args:
        action (int): The action to be taken. Possible values are:
                      0 - No action
                      1 - Set UPR to 1.0
                      2 - Set C1 to 0.0
                      3 - Set UPR to 0.75 and C1 to 0.25
        FS (object): The fuzzy system object where the variables will be set.

    Returns:
        int: The action that was set.
    """
    if action == 1:
        FS.set_variable("C1", 0.0)
        return 1
    if action == 2:
        FS.set_variable("UPR", 1.0)
        return 2
    if action == 3:
        FS.set_variable("UPR", 0.75)
        FS.set_variable("C1", 0.25)
        return 3

    return 0


def run_test_planner(nsteps=3):
    steps = 100
    steps_to_save = [2, 17, 33, 50, 67, 83, 99]
    necrosis = np.zeros((4, 7))
    apoptosis = np.zeros((4, 7))
    treatment_print = np.zeros((7, 4))
    individual_final_apop = np.zeros((4, nsteps))
    individual_final_necr = np.zeros((4, nsteps))
    res = Parallel(n_jobs=-1)(delayed(test_planner)(k, steps, steps_to_save) for k in range(0, nsteps))

    for i in range(nsteps):
        np.add(apoptosis, res[i][0], out=apoptosis)
        np.add(necrosis, res[i][1], out=necrosis)
        individual_final_apop[:, i] = res[i][2]
        individual_final_necr[:, i] = res[i][3]
        np.add(treatment_print, res[i][4], out=treatment_print)
    print("Done")

    print("Apoptosis")
    print("DTR mean: ", np.mean(individual_final_apop[0]), "C1 mean: ", np.mean(individual_final_apop[1]), "UPR mean: ",
          np.mean(individual_final_apop[2]), "Combined mean: ", np.mean(individual_final_apop[3]))
    print("DTR std: ", np.std(individual_final_apop[0]), "C1 std: ", np.std(individual_final_apop[1]), "UPR std: ",
          np.std(individual_final_apop[2]), "Combined std: ", np.std(individual_final_apop[3]))

    print("Necrosis")
    print("DTR mean: ", np.mean(individual_final_necr[0]), "C1 mean: ", np.mean(individual_final_necr[1]), "UPR mean: ",
          np.mean(individual_final_necr[2]), "Combined mean: ", np.mean(individual_final_necr[3]))
    print("DTR std: ", np.std(individual_final_necr[0]), "C1 std: ", np.std(individual_final_necr[1]), "UPR std: ",
          np.std(individual_final_necr[2]), "Combined std: ", np.std(individual_final_necr[3]))


    time_h = ["0h", "12h", "24h", "36h", "48h", "60h", "72h"]
    apo_pd = pd.DataFrame(apoptosis, columns=time_h)
    necr_pd = pd.DataFrame(necrosis, columns=time_h)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Apoptosis")
    ax.plot(apo_pd.columns, apo_pd.iloc[0], linestyle='-', marker='o', linewidth=1., color='royalblue')
    ax.plot([0, 1, 2, 3, 4, 5, 6], apoptosis[1, 0:7], linestyle='--', marker='o', linewidth=1., color='indianred')
    ax.plot([0, 1, 2, 3, 4, 5, 6], apoptosis[2, 0:7], linestyle='--', marker='o', linewidth=1., color='orange')
    ax.plot([0, 1, 2, 3, 4, 5, 6], apoptosis[3, 0:7], linestyle='--', marker='o', linewidth=1., color='seagreen')
    plt.legend(["DTR", "C1  inhibition", "UPR Activation", "Combined"], loc="lower right")
    plt.show()
    fig2 = plt.figure(dpi=300)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlim(0, 7)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Necrosis")
    ax2.plot(necr_pd.columns, necr_pd.iloc[0], linestyle='-', marker='o', linewidth=1., color='royalblue')
    ax2.plot([0, 1, 2, 3, 4, 5, 6], necrosis[1, 0:7], linestyle='--', marker='o', linewidth=1., color='indianred')
    ax2.plot([0, 1, 2, 3, 4, 5, 6], necrosis[2, 0:7], linestyle='--', marker='o', linewidth=1., color='orange')
    ax2.plot([0, 1, 2, 3, 4, 5, 6], necrosis[3, 0:7], linestyle='--', marker='o', linewidth=1., color='seagreen')
    plt.legend(["DTR", "C1  inhibition", "UPR Activation", "Combined"], loc="lower right")
    plt.show()
    print(necrosis)
    print(apoptosis)
    print(treatment_print)


def test_planner(patient_i, steps=100, steps_to_save=None):
    """
    Simulates the evolution of a fuzzy system over a series of steps, applying different actions
    and recording the resulting states. Compares the effect of the proposed treatment with another strategy
    for the same set of patients.

    Args:
        nsteps (int): Number of steps to simulate. Default is 3.

    Returns:
        None
    """
    i = patient_i
    necrosis = np.zeros((4, 7))
    apoptosis = np.zeros((4, 7))
    treatment_print = np.zeros((7, 4))
    individual_final_apop = np.zeros((4))
    individual_final_necr = np.zeros((4))
    count = 0
    index = 0
    fs = get_fs()
    set_initial_state(fs)
    # Save the initial state variables
    variables_at_start = deepcopy(fs._variables)
    dynamics = deepcopy(fs._variables)  # Initialize dynamics dictionary to store variable changes
    for var in dynamics.keys():
        dynamics[var] = [dynamics[var]]

    action = 0
    list_patients_i = [dynamics["PKA"][0], dynamics["RasGTP"][0]]
    key_patient_i = ["PKA", "RasGTP"]
    df_patient_test = pd.DataFrame([list_patients_i], columns=key_patient_i)
    for T in np.linspace(0, 1, steps):

        # compute the actual state
        set_action(action, fs)
        new_values = fs.Sugeno_inference()
        fs._variables.update(new_values)

        fs.set_variable("Glucose", time_function(T))
        new_values['Glucose'] = fs._variables['Glucose']
        for var in new_values.keys():
            dynamics[var].append(new_values[var])

        if count in steps_to_save:
            # Save the values of Necrosis and Apoptosis at the current step
            necrosis[0][index] += fs._variables["Necrosis"]
            apoptosis[0][index] += fs._variables["Apoptosis"]

            # Create a dataframe with the patient's data at the current step

            new_values_in_df = []
            new_columns = []
            for key, value in dynamics.items():
                if key != "PKA" and key != "RasGTP":
                    new_values_in_df.append(value[count])
                    new_columns.append(key + "_" + str(index))

            df_patient_tmp = pd.DataFrame([new_values_in_df], columns=new_columns)
            df_patient_test = df_patient_test.combine_first(df_patient_tmp)

            # Select the action with the highest Q-value given the values of the state variables
            if index < 6:
                # Select the action with the highest Q-value
                action = predict(index, get_history(df_patient_test, index))[0]
                df_patient_test["Treatment_" + str(index)] = action
                treatment_print[index][action] += 1
                index += 1

        count += 1

    print("Patient", i)
    print("==== Variable at %d ===" % count)
    print("Apt :  ", fs._variables["Apoptosis"])
    print("Necr:  ", fs._variables["Necrosis"])
    reward = fs._variables["Apoptosis"] - fs._variables["Necrosis"]
    print("Reward:", reward)
    individual_final_apop[0] += fs._variables["Apoptosis"]
    individual_final_necr[0] += fs._variables["Necrosis"]

    # Reset variables to compute a comparison with another strategy for the same set of patients
    fs._variables = deepcopy(variables_at_start)
    count = 0
    index = 0
    action = 0
    for T in np.linspace(0, 1, steps):
        set_action(action, fs)
        new_values = fs.Sugeno_inference()
        fs._variables.update(new_values)
        fs.set_variable("Glucose", time_function(T))
        new_values['Glucose'] = fs._variables['Glucose']
        for var in new_values.keys():
            dynamics[var].append(new_values[var])

        if count in steps_to_save:
            necrosis[1][index] += fs._variables["Necrosis"]
            apoptosis[1][index] += fs._variables["Apoptosis"]
            action = 1
            index += 1

        count += 1
    individual_final_apop[1] += fs._variables["Apoptosis"]
    individual_final_necr[1] += fs._variables["Necrosis"]

    # Reset variables to compute a comparison with another strategy for the same set of patients
    fs._variables = deepcopy(variables_at_start)
    count = 0
    index = 0
    action = 0
    for T in np.linspace(0, 1, steps):
        set_action(action, fs)
        new_values = fs.Sugeno_inference()
        fs._variables.update(new_values)
        fs.set_variable("Glucose", time_function(T))
        new_values['Glucose'] = fs._variables['Glucose']
        for var in new_values.keys():
            dynamics[var].append(new_values[var])

        if count in steps_to_save:
            necrosis[2][index] += fs._variables["Necrosis"]
            apoptosis[2][index] += fs._variables["Apoptosis"]
            action = 2
            index += 1

        count += 1
    individual_final_apop[2] += fs._variables["Apoptosis"]
    individual_final_necr[2] += fs._variables["Necrosis"]

    fs._variables = deepcopy(variables_at_start)
    count = 0
    index = 0
    action = 0
    for T in np.linspace(0, 1, steps):
        set_action(action, fs)
        new_values = fs.Sugeno_inference()
        fs._variables.update(new_values)
        fs.set_variable("Glucose", time_function(T))
        new_values['Glucose'] = fs._variables['Glucose']
        for var in new_values.keys():
            dynamics[var].append(new_values[var])

        if count in steps_to_save:
            necrosis[3][index] += fs._variables["Necrosis"]
            apoptosis[3][index] += fs._variables["Apoptosis"]
            action = 3
            index += 1

        count += 1
    individual_final_apop[3] += fs._variables["Apoptosis"]
    individual_final_necr[3] += fs._variables["Necrosis"]

    return apoptosis, necrosis, individual_final_apop, individual_final_necr, treatment_print


df_columns = fuzzyData.columns
run_test_planner(nsteps=50)

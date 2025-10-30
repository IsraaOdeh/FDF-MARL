
import os
import functools
import numpy as np
import pandas as pd

from copy import copy

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
from sklearn.model_selection import train_test_split
# from pettingzoo import ParallelEnv

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import contextlib


NUM_AGENTS = 10
DIM_AGENT_STATE = 2

norm_cols = [
    'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'age_<=45', 'age_>45',
    'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked',
    'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
    'workclass_State-gov', 'workclass_Without-pay',
    'education_10th', 'education_11th', 'education_12th', 'education_1st-4th',
    'education_5th-6th', 'education_7th-8th', 'education_9th',
    'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
    'education_Doctorate', 'education_HS-grad', 'education_Masters',
    'education_Preschool', 'education_Prof-school', 'education_Some-college',
    'marital-status_Divorced', 'marital-status_Married-AF-spouse',
    'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent',
    'marital-status_Never-married', 'marital-status_Separated',
    'marital-status_Widowed',
    'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair',
    'occupation_Exec-managerial', 'occupation_Farming-fishing',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Other-service', 'occupation_Priv-house-serv',
    'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales',
    'occupation_Tech-support', 'occupation_Transport-moving',
    'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative',
    'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
    'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other',
    'race_White',
    'sex_Female', 'sex_Male',
    'native-country_Cambodia', 'native-country_Canada', 'native-country_China',
    'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic',
    'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
    'native-country_France', 'native-country_Germany', 'native-country_Greece',
    'native-country_Guatemala', 'native-country_Haiti', 'native-country_Honduras',
    'native-country_Hong', 'native-country_Hungary', 'native-country_India',
    'native-country_Iran', 'native-country_Ireland', 'native-country_Italy',
    'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos',
    'native-country_Mexico', 'native-country_Nicaragua',
    'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru',
    'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal',
    'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South',
    'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago',
    'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia',
    'salary_<=50K', 'salary_>50K'
]

scaler = StandardScaler()
label_encoder = LabelEncoder()

def distribute_data(data, n):
    return np.array_split(data,n)


dataframe = pd.read_csv("adults_income_preprocessed.csv")
X_frame, Y_frame = dataframe.iloc[:,:-2], dataframe.iloc[:,-2:]
X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(X_frame, Y_frame, test_size=0.0000001, random_state=42)
# X_train_init = X_frame
# y_train_init = Y_frame

# X_test_init = []
# y_test_init = []

def PyTorchMLP(x_shape):
    x_shape, y_shape = X_train_init.shape[1], 2
    mean_shape = (x_shape + y_shape) // 2
    # Define the MLP model
    mlp_model = Sequential()
    mlp_model.add(Dense(x_shape, input_shape=(x_shape,), activation='relu'))
    mlp_model.add(Dense(mean_shape, activation='relu'))
    mlp_model.add(Dense(y_shape, activation='softmax'))

    # Compile the model
    mlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return mlp_model

es = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)

class ENVIRONMENT():

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


    def initial_obs(self):
        obs = []
        for a in self.agents:
            obs.append(sum([-abs(v) for v in self.agentsFairness[a].values()]))
            obs.append(self.agentsF1[a])
        return obs


    def initial_state(self):
        state = [self.fairness, self.accuracy] * self.n_agents
        return state

    def __init__(self, test=False):
        # EzPickle.__init__(self)
        self.fairness = 0
        self.accuracy = 0
        self.f1 = 0

        self.n_agents = NUM_AGENTS
        self.timestep = 0

        self.groups = [1,2,3,4,5,6,7]
        self.groups = [1,2,3,4,5,6] #for testing
        
        self.X, self.y, self.X_test, self.y_test = X_train_init, y_train_init, X_test_init, y_test_init
    
        # self.y = self.y.to_numpy()
        # self.y_test = self.y_test.to_numpy()

        if test:
            test_data = pd.DataFrame(np.concatenate((self.X_test, self.y_test), axis=1), columns=norm_cols)
            self.clientsData = np.array_split(test_data, self.n_agents)
        else:
            training_data = pd.DataFrame(np.concatenate((self.X, self.y), axis=1), columns=norm_cols)
            self.clientsData = distribute_data(training_data, self.n_agents)

        self.possible_agents = ["Server", "Client 1", "Client 2", "Client 3", "Client 4", "Client 5", "Client 6", "Client 7", "Client 8", "Client 9"]
        self.agents = self.possible_agents

        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        
        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.reset()
        
        self.agentsFairness = {a: {idx:0 for idx in self.groups} for a in self.agents}
        self.agentsF1 = {a: 0 for a in self.agents}

        self.observation_spaces = {i:Box(low=-0, high=100, shape=(2,), dtype=float) for i in self.possible_agents}
        self.action_spaces = {i:Discrete(2) for i in self.possible_agents}

        self.agentsRewardsPerEpisode = []

        # self.device = torch.device("cpu" if torch.mps.is_available() else "cpu")
        self._seed(42)
    

    def reset(self, test=False, seed=None, options=None):

        self.agents = copy(self.agents)

        self.fairness = 0
        self.accuracy = 0
        self.f1  = 0
        self.groups = [1,2,3,4,5,6,7]
        self.groups = [1,2,3,4,5,6] # for testing
        
        training_data = pd.DataFrame(np.concatenate((self.X, self.y), axis=1), columns=norm_cols)

        if test:
            test_data = pd.DataFrame(np.concatenate((self.X_test, self.y_test), axis=1), columns=norm_cols)
            self.clientsData = np.array_split(test_data, self.n_agents)
        else:
            self.clientsData = distribute_data(training_data, self.n_agents)

        self.timestep = 0

        self.models = {a: PyTorchMLP(self.X.shape[1]) for a in self.agents}

        self.agentsFairness = {a: {idx:0 for idx in self.groups} for a in self.agents}
        self.agentsF1 = {a: 0 for a in self.agents}
        # infos = {a: {} for a in self.agents}

        # self.terminations = {a: False for a in self.agents}
        # self.truncations = {a: False for a in self.agents}
        self.rewards = {a: 0 for a in self.agents}

        self.reward = 0

        # Train clients' models using their data
        for i, agent in enumerate(self.agents):
            if agent == 'Server':
                pass
            else:
                X = self.clientsData[i].iloc[:, :-2].astype('float32').values
                y = self.clientsData[i].iloc[:, -2:].astype('float32').values

                self.train(agent, X, y)

        # train server once to init weights
        self.train("Server", X, y, 1)

        # Get global observations [fairness, f1]
        observations = {agent: np.array([self.fairness, self.f1]) for agent in self.agents}
        obs = []
        for a in self.agents:
            obs.append(sum([-abs(v) for v in self.agentsFairness[a].values()]))
            obs.append(self.agentsF1[a])

        # self._agent_selector.reinit(self.agents)
        # self.agent_selection = self._agent_selector.reset()

        # self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.state = observations


        return obs, self.infos


    def step(self, actions):

        fairnessPerStep = []
        f1PerStep = []
        rewardsPerStep = []

        groups = self.groups

        participants = [a for a in self.agents if actions[a] == 1]

        agentsPrevFairness = {a: {idx:0 for idx in groups} for a in self.agents}
        agentsPrevF1 = {a: 0 for a in self.agents}

        DI = {a: 0 for a in self.agents}

        if(participants!=[]):
            self.aggregate_models(participants)

        for i, agent in enumerate(participants):

            if agent=='Server':
                pass
            else:
                # Save current agent model metrics
                agentsPrevFairness[agent] = self.agentsFairness[agent]
                agentsPrevF1[agent] = self.agentsF1[agent]

                X = self.clientsData[i].iloc[:, :-2].astype('float32').values
                y = self.clientsData[i].iloc[:, -2:].astype('float32').values

                self.train(agent, X, y, 50)

                with torch.no_grad():
                    predicted = self.models[agent](X).numpy()

                y_true_labels = np.argmax(y, axis=1)
                y_pred_labels = np.argmax(predicted, axis=1).round()

                # Evaluate current agent model
                f1 = f1_score(y_true_labels, y_pred_labels)
                fairnessVector, dindex = self.evaluate_fairness(X, self.groups, predicted.round(), y)

                # Store new metrics
                self.agentsFairness[agent] = fairnessVector
                DI[agent] = dindex
                self.agentsF1[agent] = f1

        # Evaluate Server model after aggregation: TEST on Server Data
        X_test = self.clientsData[0].iloc[:, :-2].astype('float32').values
        y_test = self.clientsData[0].iloc[:, -2:].astype('float32').values

        with torch.no_grad():
            predicted = self.models['Server'](X_test).numpy()

        y_true_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(predicted, axis=1).round()

        # Evaluate current agent model
        self.agentsF1["Server"] = f1_score(y_true_labels, y_pred_labels)

        self.agentsFairness["Server"], DI["Server"] = self.evaluate_fairness(X_test, self.groups, predicted.round(), y_test)
        # End of evaluation

        self.fairness = sum(-abs(i) for i in self.agentsFairness["Server"].values()) ## CHECK IF SUM SHOULD CHANGE
        self.f1 = self.agentsF1["Server"]

        # Compute Reward for all agents
        self.reward_agents(participants, agentsPrevF1, self.agentsF1, agentsPrevFairness, self.agentsFairness)

        self.agentsRewardsPerEpisode.append(self.rewards)
        self.reward = sum(self.rewards[a] for a in self.agents)
        #######################################

        fairnessPerStep.append(self.fairness)
        f1PerStep.append(self.f1)
        rewardsPerStep.append(self.reward)

        self.timestep += 1

        observations = {agent: np.array([self.fairness, self.f1]) for agent in self.agents}
        r = np.average([self.rewards[v] for v in self.rewards if v!="Server"])
        equalizedRewards = {a:r for a in self.agents}

        infos = {"fairness":fairnessPerStep,
                "F1": f1PerStep,
                "rewards": rewardsPerStep,
                "agentsFairness": self.agentsFairness,
                "agentsF1": self.agentsF1,
                "agentsRewards":self.rewards, 
                "DI": DI}
        # info={}

        obs = []
        for a in self.agents:
            obs.append(sum([-abs(v) for v in self.agentsFairness[a].values()]))
            obs.append(self.agentsF1[a])

        return obs, self.rewards, infos, #infos
        # return observations, equalizedRewards, self.terminations, self.truncations, infos


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-0, high=100, shape=(2,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)


    def aggregate_models(self, participants, test=False, testModel=None):

        if not participants:
            participants = ['Server']

        # Get model weights from each participant (list of weight arrays)
        participant_weights = [self.models[name].get_weights() for name in participants]

        # Average weights layer-wise
        avg_weights = []
        for layer_weights in zip(*participant_weights):
            layer_avg = np.mean(np.array(layer_weights), axis=0)
            avg_weights.append(layer_avg)

        # Apply averaged weights
        if not test:
            self.models["Server"].set_weights(avg_weights)
        else:
            testModel.set_weights(avg_weights)


    def evaluate_fairness(self, X, groups, predicted, y):
        fairnessVector = {}
        weightedFairnessVector = {}

        attribute_val_dict = {
        "1" : ["sex_Female", True],
        "2" : ["age_>45", True],
        "3" : ["race_Black", True],
        "4" : ["race_White", True],
        "5" : ["race_Asian-Pac-Islander", True],
        "6" : ["race_Amer-Indian-Eskimo", True],
        }

        Xx = pd.DataFrame(X, columns=norm_cols[:-2])

        for group in groups:
            key, val = attribute_val_dict[f"{group}"]
            
            if not isinstance(predicted, np.ndarray):
                predicted = predicted.numpy()
                
            if not isinstance(y, np.ndarray):
                y = y.numpy()

            
            group1indices = np.where(Xx[key].astype(int) == val)[0]
            group2indices = np.where(Xx[key].astype(int) != val)[0]

            group1_true = y[group1indices]
            group1_pred = predicted[group1indices]

            group2_true = y[group2indices]
            group2_pred = predicted[group2indices]
            
            y_true_labels_1 = np.argmax(group1_true, axis=1)
            y_pred_labels_1 = np.argmax(group1_pred, axis=1).round()

            y_true_labels_2 = np.argmax(group2_true, axis=1)
            y_pred_labels_2 = np.argmax(group2_pred, axis=1).round()

            group1F1 = f1_score(y_true_labels_1, y_pred_labels_1)
            group2F1 = f1_score(y_true_labels_2, y_pred_labels_2)

            weightedF1 = (len(group1indices)*group1F1 + len(group2indices)*group2F1) / (len(group1indices) + len(group2indices))

            weightedFairnessVector[group] = max(abs(group2F1-weightedF1), (group1F1-weightedF1))
            fairnessVector[group] = group2F1-group1F1


        # print(fairnessVector)
        return weightedFairnessVector, fairnessVector

    def flatten_mlp_parameters(self, model):
        weights = model.get_weights()  # list of weight arrays
        flat_params = np.concatenate([w.flatten() for w in weights])
        return flat_params

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def render(self):
        pass

    def close(self):
        pass

    def train(self, agent, X, y, epochs=100, batch_size=128):
        model = self.models[agent]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')


        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=500,
                callbacks=[es],
                batch_size=batch_size,
                verbose=0
            )

    def train_test(self, model, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')


        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=500,
                callbacks=[es],
                batch_size=128,
                verbose=0
            )

    def reward_agents(self, participants, prevF1, currF1, prevFairness, currFairness):
        # LOO: Leave-One-Out contains mean f1 and DI for all agents except the index agent in [agent]
        # LOO_f1 = {a:0 for a in self.agents}
        # LOO_fairness = {a:0 for a in self.agents}
        divergence = {a:0 for a in self.agents}

        # X_test = self.clientsData[0].iloc[:, :-2].astype('float32').values
        # y_test = self.clientsData[0].iloc[:, -2:].astype('float32').values

        # testModel = PyTorchMLP(X_test.shape[1])

        # testModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

        # self.train_test(testModel,X_test, y_test)

        for agent in self.agents:
            if agent == 'Server':
                pass
            else:
                # LOO_participants = [a for a in participants if a!=agent]
                # self.aggregate_models(LOO_participants, test=True, testModel=testModel)

                # with torch.no_grad():
                #     predictedLOO = testModel(X_test)

                # y_true_labels = np.argmax(y_test, axis=1)
                # y_pred_labels = np.argmax(predictedLOO, axis=1).round()

                # LOO_f1[agent] = f1_score(y_true_labels, y_pred_labels)
                # LOO_fairness[agent] = self.evaluate_fairness(X_test, self.groups, predictedLOO, y_test).values()

                f1_improvement = currF1[agent] - prevF1[agent]
                fairness_improvement = sum([(abs(a)-abs(b)) for a, b in zip(currFairness[agent].values(), prevFairness[agent].values())])
                
                # if agent in participants: ## if action = 1 compute the effect of their presence
                #     delta_f1 = self.f1 - LOO_f1[agent] # if positive, the client is benefecial, else not.
                #     delta_fairness = sum([(abs(a)-abs(b)) for a,b in zip(self.agentsFairness["Server"].values(), LOO_fairness[agent])])
                # else: ## Opposite the equations if action = 0, compute the effect of their absence
                #     delta_f1 = LOO_f1[agent] - self.f1 # if positive, the client is benefecial, else not.
                #     delta_fairness = sum([(abs(b)-abs(a)) for a,b in zip(self.agentsFairness["Server"].values(), LOO_fairness[agent])])

                # # Flatten parameters for each model
                # params1 = self.flatten_mlp_parameters(self.models['Server'])
                # params2 = self.flatten_mlp_parameters(self.models[agent])

                # # Compute cosine similarity between the two sets of parameters
                # divergence[agent] = 1 - self.cosine_similarity(params1, params2)

                baseReward = f1_improvement + fairness_improvement # Improvement on local client machines
                # self.rewards[agent] = (baseReward + delta_f1 + delta_fairness - 0.1*divergence[agent]) # additional reward for improvement with respect to the server machine
                self.rewards[agent] = baseReward #- 0.1*divergence[agent]) # additional reward for improvement with respect to the server machine

                # print(f"f1 local: {f1_improvement}")
                # print(f"fairness local: {fairness_improvement}")

                # print(f"f1 with: {self.f1}")
                # print(f"f1 without: {LOO_f1[agent]}")

                # print(f"fairness with: {self.fairness}")
                # print(f"fairness without: {sum(LOO_fairness[agent])}")

                # print(f"f1 delta: {delta_f1}")
                # print(f"fairness delta: {delta_fairness}")
                # print(f"divergence: {divergence[agent]}")
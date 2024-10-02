#########################################################################################################
# Model originally published in:
# Marco S. Nobile, Giuseppina Votta, Roberta Palorini, Simone Spolaor, Humberto De Vitto, Paolo Cazzaniga,
# Francesca Ricciardiello, Giancarlo Mauri, Lilia Alberghina, Ferdinando Chiaradonna, Daniela Besozzi.
# Fuzzy modeling and global optimization to predict novel therapeutic targets in cancer cells.
# Bioinformatics, 2020, 36.7: 2181–2188.
#########################################################################################################
# Python version, defined by means of the Simpful library, published in:
# Simone Spolaor, Martijn Scheve, Murat Firat, Paolo Cazzaniga, Daniela Besozzi, Marco S. Nobile.
# Screening for combination cancer therapieswith dynamic fuzzy modeling andmulti-objective optimization.
# Frontiers in Genetics, 2021, 12: 449.
#########################################################################################################


import random
from copy import deepcopy
import pandas as pd

import numpy as np
from simpful import *
from joblib import Parallel, delayed
from simpful import FuzzySystem

steps = 100

#steps_to_save = [2, 34, 67, 99] # 24h
#steps_to_save = [2, 9, 17, 25, 33, 42, 50, 58, 67, 75, 83, 90, 99] #6h
steps_to_save = [2, 17, 33, 50, 67, 83, 99] #12h


FS = FuzzySystem()

# Define linguistic variables
GLUCOSE_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
GLUCOSE_2 = FuzzySet(points=[[0, 0], [0.3, 1], [0.7, 1], [1, 0]], term="Medium")
GLUCOSE_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Glucose", LinguisticVariable([GLUCOSE_1, GLUCOSE_2, GLUCOSE_3], concept="Glucose"))

GLYCOLYSIS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
GLYCOLYSIS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
GLYCOLYSIS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Glycolysis",
                           LinguisticVariable([GLYCOLYSIS_1, GLYCOLYSIS_2, GLYCOLYSIS_3], concept="Glycolysis"))

C1_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="LessFunctional")
C1_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="MediumFunctional")
C1_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="MoreFunctional")
FS.add_linguistic_variable("C1", LinguisticVariable([C1_1, C1_2, C1_3], concept="C1"))

DELTAPSI_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
DELTAPSI_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("DeltaPsi", LinguisticVariable([DELTAPSI_1, DELTAPSI_2], concept="DeltaPsi"))

ROS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
ROS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
ROS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("ROS", LinguisticVariable([ROS_1, ROS_2, ROS_3], concept="ROS"))

PKA_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
PKA_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("PKA", LinguisticVariable([PKA_1, PKA_2], concept="PKA"))

ATP_1 = FuzzySet(points=[[0, 1], [0.1, 0]], term="VeryLow")
ATP_2 = FuzzySet(points=[[0, 0], [0.1, 1], [0.4, 1], [0.5, 0]], term="Low")
ATP_3 = FuzzySet(points=[[0.4, 0], [0.5, 1], [1, 0]], term="Medium")
ATP_4 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("ATP", LinguisticVariable([ATP_1, ATP_2, ATP_3, ATP_4], concept="ATP"))

CA2_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
CA2_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
CA2_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("CA2", LinguisticVariable([CA2_1, CA2_2, CA2_3], concept="CA2"))

HBP_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
HBP_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
HBP_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("HBP", LinguisticVariable([HBP_1, HBP_2, HBP_3], concept="HBP"))

NGLYCOS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
NGLYCOS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
NGLYCOS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("NGlycos", LinguisticVariable([NGLYCOS_1, NGLYCOS_2, NGLYCOS_3], concept="NGlycos"))

UPR_1 = FuzzySet(points=[[0, 1], [0.2, 1], [0.6, 0]], term="Low")
UPR_2 = FuzzySet(points=[[0.2, 0], [0.6, 1], [1, 0]], term="Medium")
UPR_3 = FuzzySet(points=[[0.6, 0], [1, 1]], term="High")
FS.add_linguistic_variable("UPR", LinguisticVariable([UPR_1, UPR_2, UPR_3], concept="UPR"))

CHOP_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
CHOP_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
CHOP_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("CHOP", LinguisticVariable([CHOP_1, CHOP_2, CHOP_3], concept="CHOP"))

BCL2_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
BCL2_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
BCL2_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Bcl2", LinguisticVariable([BCL2_1, BCL2_2, BCL2_3], concept="Bcl2"))

JNK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
JNK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("JNK", LinguisticVariable([JNK_1, JNK_2], concept="JNK"))

AUTOPHAGY_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
AUTOPHAGY_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
AUTOPHAGY_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Autophagy",
                           LinguisticVariable([AUTOPHAGY_1, AUTOPHAGY_2, AUTOPHAGY_3], concept="Autophagy"))

DAPK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
DAPK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("DAPK", LinguisticVariable([DAPK_1, DAPK_2], concept="DAPK"))

BCN1_1 = FuzzySet(points=[[0, 1], [0.33, 0]], term="Low")
BCN1_2 = FuzzySet(points=[[0, 0], [0.33, 1], [0.66, 0]], term="Medium")
BCN1_3 = FuzzySet(points=[[0.33, 0], [0.66, 1], [1.0, 0]], term="High")
BCN1_4 = FuzzySet(points=[[0.66, 0], [1, 1]], term="VeryHigh")
FS.add_linguistic_variable("BCN1", LinguisticVariable([BCN1_1, BCN1_2, BCN1_3, BCN1_4], concept="BCN1"))

CASPASE3_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
CASPASE3_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
CASPASE3_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Caspase3", LinguisticVariable([CASPASE3_1, CASPASE3_2, CASPASE3_3], concept="Caspase3"))

ATTACH_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
ATTACH_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Attach", LinguisticVariable([DAPK_1, DAPK_2], concept="Attach"))

SRC_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
SRC_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Src", LinguisticVariable([SRC_1, SRC_2], concept="Src"))

RASGTP_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Off")
RASGTP_2 = FuzzySet(points=[[0, 0], [1, 1]], term="On")
FS.add_linguistic_variable("RasGTP", LinguisticVariable([RASGTP_1, RASGTP_2], concept="RasGTP"))

NECROSIS_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
NECROSIS_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Necrosis", LinguisticVariable([NECROSIS_1, NECROSIS_2], concept="Necrosis"))

APOPTOSIS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
APOPTOSIS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
APOPTOSIS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Apoptosis",
                           LinguisticVariable([APOPTOSIS_1, APOPTOSIS_2, APOPTOSIS_3], concept="Apoptosis"))

SURVIVAL_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
SURVIVAL_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("Survival", LinguisticVariable([SURVIVAL_1, SURVIVAL_2], concept="Survival"))

ERK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
ERK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
FS.add_linguistic_variable("ERK", LinguisticVariable([ERK_1, ERK_2], concept="ERK"))

# Define output crisp values
FS.set_crisp_output_value("Low", 0)
FS.set_crisp_output_value("Medium", 0.5)
FS.set_crisp_output_value("High", 1)

FS.set_crisp_output_value("VeryLow_ATP", 0)
FS.set_crisp_output_value("Low_ATP", 0.1)

FS.set_crisp_output_value("Medium_BCN1", 0.33)
FS.set_crisp_output_value("High_BCN1", 0.66)
FS.set_crisp_output_value("VeryHigh_BCN1", 1)

FS.set_crisp_output_value("LessFunctional", 0)
FS.set_crisp_output_value("MediumFunctional", 0.5)
FS.set_crisp_output_value("MoreFunctional", 1)

# Define fuzzy rules
RULES = []

# Glycolysis_rules
RULES.append("IF (Glucose IS Low) THEN (Glycolysis IS Low)")
RULES.append("IF (Glucose IS Medium) THEN (Glycolysis IS Medium)")
RULES.append("IF (Glucose IS High) THEN (Glycolysis IS High)")

# C1_rules
RULES.append("IF (PKA IS Low) THEN (C1 IS MediumFunctional)")
RULES.append("IF (Glycolysis IS High) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
RULES.append("IF (Glycolysis IS Medium) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
RULES.append("IF (Glycolysis IS High) AND (PKA IS Low) THEN (C1 IS MediumFunctional)")
RULES.append("IF (Glycolysis IS Medium) AND (PKA IS Low) THEN (C1 IS MediumFunctional)")
RULES.append("IF (Glycolysis IS Low) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
RULES.append("IF (Glycolysis IS Low) AND (PKA IS Low) THEN (C1 IS LessFunctional)")

# DeltaPsi_rules
RULES.append("IF (Glycolysis IS Medium) THEN (DeltaPsi IS High)")
RULES.append("IF (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
RULES.append("IF (C1 IS MoreFunctional) THEN (DeltaPsi IS High)")
RULES.append("IF (C1 IS MediumFunctional) THEN (DeltaPsi IS High)")
RULES.append("IF (CA2 IS Low) THEN (DeltaPsi IS High)")
RULES.append("IF (Bcl2 IS High) THEN (DeltaPsi IS High)")
RULES.append("IF (CA2 IS High) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
RULES.append("IF (CA2 IS High) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
RULES.append("IF (CA2 IS High) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
RULES.append("IF (Bcl2 IS Medium) THEN (DeltaPsi IS Low)")
RULES.append("IF (Glycolysis IS High) THEN (DeltaPsi IS High)")

# ROS_rules
RULES.append("IF (C1 IS LessFunctional) THEN (ROS IS High)")
RULES.append("IF (C1 IS MediumFunctional) THEN (ROS IS Medium)")
RULES.append("IF (C1 IS MoreFunctional) THEN (ROS IS Low)")
RULES.append("IF (DeltaPsi IS Low) THEN (ROS IS High)")
RULES.append("IF (DeltaPsi IS High) THEN (ROS IS Medium)")

# ATP_rules
RULES.append("IF (DeltaPsi IS High) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS High) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS Medium) THEN (ATP IS Medium)")
RULES.append("IF (Glycolysis IS Low) THEN (ATP IS Low_ATP)")
RULES.append("IF (C1 IS MoreFunctional) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS High) AND (DeltaPsi IS Low) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS High) AND (C1 IS LessFunctional) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS High) AND (C1 IS MediumFunctional) THEN (ATP IS High)")
RULES.append("IF (Glycolysis IS Medium) AND (DeltaPsi IS Low) THEN (ATP IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND (C1 IS LessFunctional) THEN (ATP IS Medium)")
RULES.append("IF (Glycolysis IS Low) AND (DeltaPsi IS Low) THEN (ATP IS Medium)")
RULES.append("IF (Glycolysis IS Low) AND (C1 IS LessFunctional) THEN (ATP IS VeryLow_ATP)")
RULES.append("IF (Glycolysis IS Low) AND (C1 IS MediumFunctional) THEN (ATP IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND (C1 IS MediumFunctional) THEN (ATP IS Medium)")

# CA2_rules
RULES.append("IF (UPR IS Low) THEN (CA2 IS Low)")
RULES.append("IF (UPR IS Medium) THEN (CA2 IS Medium)")
RULES.append("IF (UPR IS High) THEN (CA2 IS High)")

# HBP_rules
RULES.append("IF (Glucose IS High) AND (Autophagy IS Low) THEN (HBP IS High)")
RULES.append("IF (Glucose IS High) AND (Autophagy IS Medium) THEN (HBP IS High)")
RULES.append("IF (Glucose IS High) AND (Autophagy IS High) THEN (HBP IS High)")
RULES.append("IF (Glucose IS Medium) AND (Autophagy IS Low) THEN (HBP IS Medium)")
RULES.append("IF (Glucose IS Medium) AND (Autophagy IS Medium) THEN (HBP IS Medium)")
RULES.append("IF (Glucose IS Medium) AND (Autophagy IS High) THEN (HBP IS High)")
RULES.append("IF (Glucose IS Low) AND (Autophagy IS Low) THEN (HBP IS Low)")
RULES.append("IF (Glucose IS Low) AND (Autophagy IS Medium) THEN (HBP IS Medium)")
RULES.append("IF (Glucose IS Low) AND (Autophagy IS High) THEN (HBP IS High)")

# NGlycos_rules
RULES.append("IF (HBP IS High) THEN (NGlycos IS High)")
RULES.append("IF (HBP IS Medium) THEN (NGlycos IS Medium)")
RULES.append("IF (HBP IS Low) THEN (NGlycos IS Low)")
RULES.append("IF (ATP IS High) THEN (NGlycos IS High)")
RULES.append("IF (ATP IS Medium) THEN (NGlycos IS High)")
RULES.append("IF (ATP IS Low) THEN (NGlycos IS Medium)")
RULES.append("IF (ATP IS VeryLow) THEN (NGlycos IS Low)")

# UPR_rules
RULES.append("IF (NGlycos IS High) THEN (UPR IS Low)")
RULES.append("IF (NGlycos IS Medium) THEN (UPR IS Medium)")
RULES.append("IF (NGlycos IS Low) THEN (UPR IS High)")
RULES.append("IF (ATP IS High) THEN (UPR IS Low)")
RULES.append("IF (ATP IS Medium) THEN (UPR IS Low)")
RULES.append("IF (ATP IS Low) THEN (UPR IS Medium)")
RULES.append("IF (ATP IS VeryLow) THEN (UPR IS High)")

# CHOP_rules
RULES.append("IF (UPR IS Low) THEN (CHOP IS Low)")
RULES.append("IF (UPR IS Medium) THEN (CHOP IS Medium)")
RULES.append("IF (UPR IS High) THEN (CHOP IS High)")

# JNK_rules
RULES.append("IF (UPR IS High) THEN (JNK IS High)")
RULES.append("IF (UPR IS Medium) THEN (JNK IS High)")
RULES.append("IF (UPR IS Low) THEN (JNK IS Low)")

# Autophagy_rules
RULES.append("IF (CA2 IS High) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS High) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS VeryHigh) THEN (Autophagy IS High)")
RULES.append("IF (ROS IS High) THEN (Autophagy IS High)")
RULES.append("IF (Glycolysis IS Low) THEN (Autophagy IS High)")
RULES.append("IF (CA2 IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (ATP IS VeryLow) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (ATP IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (ATP IS High) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (Glycolysis IS High) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (CA2 IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (BCN1 IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (ATP IS VeryLow) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (ATP IS Medium) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (ATP IS High) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (Glycolysis IS High) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (CA2 IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (Glycolysis IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (CA2 IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
RULES.append("IF (BCN1 IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
RULES.append("IF (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (ROS IS Medium) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (ROS IS Medium) AND (PKA IS High) THEN (Autophagy IS Medium)")
RULES.append("IF (ROS IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
RULES.append("IF (ROS IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
RULES.append("IF (PKA IS High) AND (ATP IS Low) THEN (Autophagy IS High)")
RULES.append("IF (PKA IS Low) AND (ATP IS Low) THEN (Autophagy IS Medium)")
RULES.append("IF (PKA IS Low) AND (CA2 IS High) THEN (Autophagy IS High)")
RULES.append("IF (PKA IS Low) AND (BCN1 IS High) THEN (Autophagy IS High)")
RULES.append("IF (PKA IS Low) AND (Glycolysis IS Low) THEN (Autophagy IS High)")

# DAPK_rules
RULES.append("IF (Src IS High) THEN (DAPK IS Low)")
RULES.append("IF (CA2 IS Low) THEN (DAPK IS Low)")
RULES.append("IF (ERK IS High) THEN (DAPK IS Low)")
RULES.append("IF (Src IS Low) THEN (DAPK IS High)")
RULES.append("IF (CA2 IS High) THEN (DAPK IS High)")
RULES.append("IF (CA2 IS Medium) THEN (DAPK IS High)")
RULES.append("IF (ERK IS Low) THEN (DAPK IS High)")

# Bcl2_rules
RULES.append("IF (CHOP IS High) THEN (Bcl2 IS Low)")
RULES.append("IF (CHOP IS Medium) THEN (Bcl2 IS Medium)")
RULES.append("IF (CHOP IS Low) THEN (Bcl2 IS High)")
RULES.append("IF (CHOP IS Medium) AND (JNK IS High) THEN (Bcl2 IS Low)")
RULES.append("IF (CHOP IS Low) AND (JNK IS High) THEN (Bcl2 IS Low)")
RULES.append("IF (CHOP IS Medium) AND (JNK IS Low) THEN (Bcl2 IS Medium)")
RULES.append("IF (BCN1 IS High) THEN (Bcl2 IS Low)")
RULES.append("IF (ATP IS High) THEN (Bcl2 IS High)")
RULES.append("IF (ATP IS Medium) THEN (Bcl2 IS High)")
RULES.append("IF (ATP IS Low) THEN (Bcl2 IS Low)")
RULES.append("IF (ATP IS VeryLow) THEN (Bcl2 IS Low)")
RULES.append("IF (JNK IS Low) THEN (Bcl2 IS High)")
RULES.append("IF (BCN1 IS Low) THEN (Bcl2 IS High)")
RULES.append("IF (CHOP IS High) AND (JNK IS High) THEN (Bcl2 IS Low)")
RULES.append("IF (BCN1 IS Medium) THEN (Bcl2 IS Medium)")
RULES.append("IF (BCN1 IS VeryHigh) THEN (Bcl2 IS Low)")
RULES.append("IF (PKA IS High) THEN (Bcl2 IS Medium)")
RULES.append("IF (PKA IS Low) THEN (Bcl2 IS High)")

# BCN1_rules
RULES.append("IF (Bcl2 IS Low) THEN (BCN1 IS High_BCN1)")
RULES.append("IF (Bcl2 IS Medium) THEN (BCN1 IS Medium_BCN1)")
RULES.append("IF (Bcl2 IS High) THEN (BCN1 IS Low)")
RULES.append("IF (DAPK IS High) THEN (BCN1 IS VeryHigh_BCN1)")
RULES.append("IF (DAPK IS Low) THEN (BCN1 IS Low)")

# Caspase3_rules
RULES.append("IF (ATP IS Low) THEN (Caspase3 IS High)")
RULES.append("IF (ATP IS High) THEN (Caspase3 IS Low)")
RULES.append("IF (ATP IS Medium) THEN (Caspase3 IS Low)")
RULES.append("IF (ROS IS Low) THEN (Caspase3 IS Low)")
RULES.append("IF (ROS IS Medium) THEN (Caspase3 IS Low)")
RULES.append("IF (CA2 IS Medium) THEN (Caspase3 IS Low)")
RULES.append("IF (CA2 IS Low) THEN (Caspase3 IS Low)")
RULES.append("IF (ERK IS High) THEN (Caspase3 IS Low)")
RULES.append("IF (Bcl2 IS High) THEN (Caspase3 IS Low)")
RULES.append("IF (DeltaPsi IS High) THEN (Caspase3 IS Low)")
RULES.append("IF (ROS IS High) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (CA2 IS High) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (DeltaPsi IS Low) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (ROS IS High) AND (ATP IS High) THEN (Caspase3 IS High)")
RULES.append("IF (ROS IS High) AND (ATP IS Medium) THEN (Caspase3 IS High)")
RULES.append("IF (ROS IS High) AND (ATP IS Low) THEN (Caspase3 IS High)")
RULES.append("IF (CA2 IS High) AND (ATP IS High) THEN (Caspase3 IS High)")
RULES.append("IF (CA2 IS High) AND (ATP IS Medium) THEN (Caspase3 IS High)")
RULES.append("IF (CA2 IS High) AND (ATP IS Low) THEN (Caspase3 IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS High) THEN (Caspase3 IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS Medium) THEN (Caspase3 IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS Low) THEN (Caspase3 IS High)")
RULES.append("IF (DeltaPsi IS Low) AND (ATP IS High) THEN (Caspase3 IS High)")
RULES.append("IF (DeltaPsi IS Low) AND (ATP IS Medium) THEN (Caspase3 IS High)")
RULES.append("IF (DeltaPsi IS Low) AND (ATP IS Low) THEN (Caspase3 IS High)")
RULES.append("IF (Bcl2 IS Medium) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (Bcl2 IS Medium) AND (ATP IS High) THEN (Caspase3 IS Medium)")
RULES.append("IF (Bcl2 IS Medium) AND (ATP IS Medium) THEN (Caspase3 IS Medium)")
RULES.append("IF (Bcl2 IS Medium) AND (ATP IS Low) THEN (Caspase3 IS Medium)")
RULES.append("IF (ATP IS VeryLow) THEN (Caspase3 IS Low)")
RULES.append("IF (ERK IS Low) THEN (Caspase3 IS Medium)")

# Attach_rules
RULES.append("IF (NGlycos IS High) THEN (Attach IS High)")
RULES.append("IF (NGlycos IS Low) THEN (Attach IS Low)")
RULES.append("IF (NGlycos IS Medium) THEN (Attach IS Low)")

# Src_rules
RULES.append("IF (Attach IS Low) THEN (Src IS Low)")
RULES.append("IF (Attach IS High) THEN (Src IS High)")
RULES.append("IF (PKA IS High) THEN (Src IS High)")
RULES.append("IF (PKA IS Low) THEN (Src IS Low)")
RULES.append("IF (PKA IS High) AND (Attach IS Low) THEN (Src IS High)")

# ERK_rules
RULES.append("IF (Src IS High) THEN (ERK IS High)")
RULES.append("IF (Src IS Low) AND (RasGTP IS On) THEN (ERK IS High)")
RULES.append("IF (DAPK IS High) AND (RasGTP IS On) THEN (ERK IS High)")
RULES.append("IF (Src IS Low) AND (RasGTP IS Off) THEN (ERK IS Low)")
RULES.append("IF (DAPK IS High) AND (RasGTP IS Off) THEN (ERK IS Low)")
RULES.append("IF (RasGTP IS On) THEN (ERK IS High)")
RULES.append("IF (DAPK IS Low) THEN (ERK IS High)")
RULES.append("IF (RasGTP IS Off) AND (Src IS High) THEN (ERK IS High)")
RULES.append("IF (RasGTP IS Off) AND (DAPK IS Low) THEN (ERK IS High)")

# Necrosis_rules
RULES.append("IF (Bcl2 IS High) THEN (Necrosis IS Low)")
RULES.append("IF (Bcl2 IS Medium) THEN (Necrosis IS Low)")
RULES.append("IF (ATP IS VeryLow) THEN (Necrosis IS High)")
RULES.append("IF (ATP IS Low) THEN (Necrosis IS Low)")
RULES.append("IF (ATP IS Medium) THEN (Necrosis IS Low)")
RULES.append("IF (ROS IS Medium) THEN (Necrosis IS Low)")
RULES.append("IF (ROS IS Low) THEN (Necrosis IS Low)")
RULES.append("IF (ROS IS High) AND (ATP IS High) THEN (Necrosis IS Low)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS High) THEN (Necrosis IS Low)")
RULES.append("IF (ROS IS High) AND (ATP IS VeryLow) THEN (Necrosis IS High)")
RULES.append("IF (ROS IS High) AND (ATP IS Low) THEN (Necrosis IS High)")
RULES.append("IF (ROS IS High) AND (ATP IS Medium) THEN (Necrosis IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS VeryLow) THEN (Necrosis IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS Low) THEN (Necrosis IS High)")
RULES.append("IF (Bcl2 IS Low) AND (ATP IS Medium) THEN (Necrosis IS High)")
RULES.append("IF (ATP IS High) THEN (Necrosis IS Low)")

# Apoptosis_rules
RULES.append("IF (Caspase3 IS Low) THEN (Apoptosis IS Low)")
RULES.append("IF (Caspase3 IS Medium) THEN (Apoptosis IS Medium)")
RULES.append("IF (Caspase3 IS High) THEN (Apoptosis IS High)")

# Survival_rules
RULES.append("IF (DeltaPsi IS Low) THEN (Survival IS Low)")
RULES.append("IF (DeltaPsi IS High) THEN (Survival IS High)")
RULES.append("IF (Caspase3 IS High) THEN (Survival IS Low)")
RULES.append("IF (Caspase3 IS Low) THEN (Survival IS High)")
RULES.append("IF (Caspase3 IS Medium) THEN (Survival IS Low)")
RULES.append("IF (ATP IS VeryLow) THEN (Survival IS Low)")
RULES.append("IF (ATP IS Low) THEN (Survival IS Low)")
RULES.append("IF (ATP IS Medium) THEN (Survival IS High)")
RULES.append("IF (ATP IS High) THEN (Survival IS High)")
RULES.append("IF (Autophagy IS Medium) THEN (Survival IS High)")
RULES.append("IF (Autophagy IS High) THEN (Survival IS High)")
RULES.append("IF (Autophagy IS Low) THEN (Survival IS Low)")

FS.add_rules(RULES)


# Set initial state of the model
def set_initial_state(FS):
    FS.set_variable("Glucose", 1.0)

    random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("Glycolysis", 1.0)

    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("C1", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("DeltaPsi", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("ROS", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.4, 0.6, 1], [0.25, 0.25, 0.25, 0.25], k=1)[0]
    FS.set_variable("ATP", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("CA2", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("HBP", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("NGlycos", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("UPR", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("CHOP", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("Bcl2", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("JNK", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("Autophagy", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("DAPK", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.33, 0.66, 1], [0.25, 0.25, 0.25, 0.25], k=1)[0]
    FS.set_variable("BCN1", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("Caspase3", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("Attach", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("Src", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("Necrosis", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 0.5, 1], [0.33, 0.33, 0.34], k=1)[0]
    FS.set_variable("Apoptosis", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("Survival", random_variable)
    random_variable = random.random()
    #random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("ERK", random_variable)
    # PKA low state

    random_variable = random.uniform(0.51, 1)
    FS.set_variable("RasGTP", random_variable)

    random_variable = random.choices([0, 1], [0.5, 0.5], k=1)[0]
    FS.set_variable("PKA", random_variable)
    random_variable = random.random()
    FS.set_variable("PKA", random_variable)

# Define update function for Glucose variable
def time_function(curtime):
    if curtime < 0.075:
        return 1
    elif curtime > 0.7:
        return 0
    else:
        return 1. / (7 * curtime ** 0.75) - 0.185


def get_fs():
    return FS


def choice_treatment(FS):
    """
    Retrieve the action based on some rules

    Args:
        action (int): The action to be taken.
        FS (object): The fuzzy system object where the variables will be set.

    Returns:
        int: The action that was set.
    """
    if (FS._variables["PKA"] > 0.5):
        if (FS._variables["UPR"] < 0.5):
            return random.choices([1, 2, 3], [0.05, 0.9, 0.05], k=1)[0]
        else:
            return random.choices([1, 2, 3], [0.05, 0.05, 0.9], k=1)[0]
    else:
        if (FS._variables["UPR"] < 0.4):
            if (FS._variables["C1"] > 0.5):
                return random.choices([1, 2, 3], [0.05, 0.05, 0.9], k=1)[0]
            else:
                return random.choices([1, 2, 3], [0.05, 0.9, 0.05], k=1)[0]
        else:
            if (FS._variables["C1"] > 0.5):
                return random.choices([1, 2, 3], [0.9, 0.05, 0.05], k=1)[0]
            else:
                return random.choices([1, 2, 3], [0.3, 0.1, 0.6], k=1)[0]

def set_treatment(FS, drug):
    if drug == 1:
        FS.set_variable("C1", 0.0)
    elif drug == 2:
        FS.set_variable("UPR", 1.0)
    elif drug == 3:
        FS.set_variable("UPR", 0.75)
        FS.set_variable("C1", 0.25)


def simulate(index, FS):
    set_initial_state(FS)
    dynamics = deepcopy(FS._variables)
    for var in dynamics.keys():
        dynamics[var] = [dynamics[var]]

    treatments = []
    drug = 0
    step_index = 0
    step_total = 0
    for T in np.linspace(0, 1, steps):
        if step_total in steps_to_save:
            if step_index == 0:
                drug = choice_treatment(FS)
                step_index += 1
            else:
                drug = choice_treatment(FS)
                step_index += 1
        set_treatment(FS, drug)
        treatments.append(drug)
        new_values = FS.Sugeno_inference()
        FS._variables.update(new_values)
        FS.set_variable("Glucose", time_function(T))
        new_values['Glucose'] = FS._variables['Glucose']
        step_total += 1

        for var in new_values.keys():
            dynamics[var].append(new_values[var])

    #plot 50 patients
    #if index < 50:
    #    survival = dynamics["Survival"]
    #    necrosis = dynamics["Necrosis"]
    #    apoptosis = dynamics["Apoptosis"]
    #    plt.plot(range(steps + 1), survival)
    #    plt.plot(range(steps + 1), necrosis)
    #    plt.plot(range(steps + 1), apoptosis)
    #    plt.ylim(0, 1.05)
    #    plt.xlabel("Time")
    #    plt.ylabel("Level")
    #    plt.legend(["Survival", "Necrosis", "Apoptosis"], loc="lower right", framealpha=1.0)
    #    plt.show()

    list_patients_i = [FS._variables["PKA"], FS._variables["RasGTP"]]

    for key, value in dynamics.items():
        if (key != "PKA" and key != "RasGTP"):
            for j in steps_to_save:
                list_patients_i.append(value[j])

    for j in steps_to_save:
        list_patients_i.append(treatments[j])

    return list_patients_i


def generate_columns_names():
    keys_patients = ["PKA", "RasGTP"]
    set_initial_state(FS)
    dynamics = deepcopy(FS._variables)
    for key, _ in dynamics.items():
        for j in range(len(steps_to_save)):
            if key != "PKA" and key != "RasGTP":
                keys_patients.append(key + "_" + str(j))

    for j in range(len(steps_to_save)):
        keys_patients.append("Treatment_" + str(j))
    return keys_patients


# Perform Sugeno inference and save results
if __name__ == "__main__":
    element_run = Parallel(n_jobs=-1)(delayed(simulate)(k, FS) for k in range(1, 30000))
    list_patients = element_run
    keys_patients = generate_columns_names()
    df_patients = pd.DataFrame(list_patients, columns=keys_patients)
    df_patients.to_excel("dataset_15.xlsx")

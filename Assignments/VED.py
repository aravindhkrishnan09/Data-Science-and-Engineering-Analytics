#import csv into df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os

# Set the working directory
os.chdir("G:\\DIY Guru\\Data-Science-and-Engineering-Analytics\\VED\\Data")
# Read the CSV file into a DataFrame
df_static = pd.read_excel("VED_Static_Data_ICE&HEV_PHEV&EV.xlsx")
df_dynamic = pd.read_csv("VED_Dynamic_Data_Part1&2_Sampled.csv")

# Display the first few rows of the DataFrame
print(df_static.head())
print(df_dynamic.head())


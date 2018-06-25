# We Move from the original list of one-hot encoded training attributes to a modified list of attributes with the conditional dependancy on the protected attribute removed.

import csv
import numpy as np
import math

# Converts a file of comma separated values into a numerical array.
def array_preprocess(filename):
    file=open(filename,"r")
    original_data=file.read().split('\n')
    resulting_array=[]
    for row in original_data[:-1]:
        value_storer=[]
        string_vals=row.split(',')
        for s in string_vals:
            num=float(s)
            value_storer.append(num)
        resulting_array.append(value_storer)
    return resulting_array

# Same thing but now produces single list rather than array (i.e. 1D array).
def list_preprocess(filename):
    file=open(filename,"r")
    original_data=file.read().split('\n')
    resulting_array=[]
    for s in original_data:
        resulting_array.append(float(s))
    return resulting_array

# To compute the mean of the continuous attributes, we assume a continuous internal distribution and therefore the expectation can be given as the probability for it to lie in the range times the middle value of the range.

def expectation(mids,probabilities):
    assert (len(mids)==len(probabilities))
    expectation=0
    for bin,mid in mids:
        expectation = expectation+mid*probabilities[bin]
    return expectation

def calculate_middles(interval_list):
    middles = []
    for interval_number, threshold in enumerate(interval_list[:-1]):
        mid=(threshold+interval_list[interval_number+1])/2
        middles.append(mid)
    return middles


# One-hot encoded test data:
original_test_oh=array_preprocess("germancredit.ts.data")

# List of the bin boundaries initially used to categorise continuous variables.
intervals=array_preprocess("germancredit.intervals.data")

# Log likelihoods for all DEPENDENT attributes calculated by the SPN.
log_likelihoods_male=list_preprocess("Inference_Male_Log_Likelihoods")
log_likelihoods_female=list_preprocess("Inference_Female_Log_Likelihoods")

# Convert log likelihoods to probabilities.
probabilities_male=[math.exp(i) for i in log_likelihoods_male]
probabilities_female=[math.exp(i) for i in log_likelihoods_female]


print(intervals)

all_mids=[]
for continuous_data in intervals:
    all_mids.append(calculate_middles(continuous_data))

print(all_mids)

four_discrete=[1.0,2.0,3.0,4.0]
two_discrete=[1.0,4.0]

discrete_encoder=[0,0,4,4,0,4,2]

for attribute,list in enumerate(all_mids):
    if discrete_encoder[attribute]==4:
        all_mids[attribute]=four_discrete
    elif discrete_encoder[attribute]==2:
        all_mids[attribute]=two_discrete

print(all_mids)


# Reimport original data

# Data types:
# 2 - encodes binary categorical - STRING
# 1 - encodes multiple categorical - NUMBER
# 0 - encodes numeric

encoding_list=[1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,2,2,2]

# Ranges of all the attributes with high statistical dependency on the protected (including the protected)
unique_range=[[44, 49], [53, 54], [57, 61], [61, 65], [65, 75], [78, 81], [85, 89], [89, 91], [91, 92]]

# The only continuous protected attributes which happen to exhibit high statistical dependency are AGE (65-74), DEPENDENTS (89-90) and PERCENTAGE OF INCOME (57-61).



# We now isolate this attribute and compute the expectation:

# Target array should be a combination of ONE HOT encoded categoricals and real values continuous attribute.


# The following is a list of all variables which exhibit dependency on the SPN - this can be modified to an import statement later.

# For Categorical Variables, E(x_j|a) = P(x_j=1|a).


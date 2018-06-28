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
    for bin,mid in enumerate(mids):
        expectation = expectation+mid*probabilities[bin]
    return expectation

def calculate_middles(interval_list):
    middles = []
    for interval_number, threshold in enumerate(interval_list[:-1]):
        mid=(threshold+interval_list[interval_number+1])/2
        middles.append(mid)
    return middles

# List of the bin boundaries initially used to categorise continuous variables.
intervals=array_preprocess("germancredit.intervals.data")
print(intervals)

# Log likelihoods for all DEPENDENT attributes calculated by the SPN.
log_likelihoods_male=list_preprocess("Inference_Male_Log_Likelihoods")
log_likelihoods_female=list_preprocess("Inference_Female_Log_Likelihoods")

# Convert log likelihoods to probabilities.
probabilities_male=[math.exp(i) for i in log_likelihoods_male]
probabilities_female=[math.exp(i) for i in log_likelihoods_female]
print(len(probabilities_male))

## IMPORTANT: GIVEN THE NATURE OF THE SPN, IT MAY TURN OUT TO BE THE CASE THAT IT'S NECESSARY TO RENORMALISE THE PROBABILITIES. FOR THE MOMENT, I'M GOING TO KEEP EVERYTHING AS IS AND THEN ADJUST LATER IF NECESSARY.

# For every list containing numeric data, we wish to calculate the middles of each.
all_mids=[]
for numeric_data in intervals:
    all_mids.append(calculate_middles(numeric_data))


# Subsequently for all discrete numeric data, we replace the intervals ascribed to them with their TRUE values.
four_discrete=[1.0,2.0,3.0,4.0]
two_discrete=[1.0,2.0]

# 0 = Continuous
# 4 = [1,2,3,4] - possible values
# 2 = [1,2] - possible values
discrete_encoder=[0,0,4,4,0,4,2]

# Replace the calculated list of 10 mids with the true values where necessary.
for attribute,list in enumerate(all_mids):
    if discrete_encoder[attribute]==4:
        all_mids[attribute]=four_discrete
    elif discrete_encoder[attribute]==2:
        all_mids[attribute]=two_discrete

print("MIDS")
print(all_mids)

# Data types:
# 2 - encodes binary categorical - STRING
# 1 - encodes multiple categorical - NUMBER
# 0 - encodes numeric

encoding_list=[1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,2,2,2]

# Specifies number of distinct classes in each attribute. (i.e. number of entries in probability vector which each separate attribute occupies).
list_of_prob_lengths=[4,10,5,10,10,5,5,4,2,3,4,4,10,3,3,4,4,2,2,2,2]

# Specifies whether or not each attribute exhibits strong statistical dependency on the final outcome according to SPN. 0 = no, 1 = yes.

dependency_list=[0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0]
print("length of dependency list")
print(len(dependency_list))

# Establish two lists which mark the boundaries in the two columns where attributes begin and end (note one is list of all probabilities with length 98, the other is the list of only the probabilities of those which are heavily dependent on the protected attribute).

prob_cumulative=[0]
cumulative_dependent=[0]
# First attribute "starts" at base 0
for i,j in enumerate(list_of_prob_lengths):
    # Next attribute starts at (last_attribute_start + length of last attribute).
    prob_cumulative.append(prob_cumulative[i]+j)
    
    # Only add to cumulative_dependent if the attribute is one of the ones with significant dependency on the protected.
    
    if dependency_list[i]==1:
        cumulative_dependent.append(cumulative_dependent[-1]+j)

print("prob_cumulative")
print(prob_cumulative)
print(cumulative_dependent)

# I want to Isolate the probabilities and the corresponding mids so I can compute expectations.

all_cont_male_expectations=[]
all_cont_female_expectations=[]

all_categoric_male_probs=[]
all_categoric_female_probs=[]

continuous_attribute_tracker=0
dependent_attribute_tracker=0

# Loop over all attributes

for attribute,position in enumerate(prob_cumulative[:-1]):
    
    # Select only the attributes which are BOTH dependent on the protected AND numeric.
    
    if encoding_list[attribute]==0 and dependency_list[attribute]==1:
        
        # Separate out the section of the probabilities which are heavily dependent on the male attribute.
        
        min_index=cumulative_dependent[dependent_attribute_tracker]
        max_index=cumulative_dependent[dependent_attribute_tracker+1]
        pm = probabilities_male[min_index:max_index]
        print("pm length is " + str(len(pm)))
        print(pm)
        pf = probabilities_female[min_index:max_index]
        
        # Separate out the relevant section of mids (i.e. average value in each section).
        this_mid=all_mids[continuous_attribute_tracker]
        print("mid len is " + str(len(this_mid)))
        print(this_mid)
        
        # Compute the expectation for males and females and append to list.
        all_cont_male_expectations.append(expectation(this_mid,pm))
        all_cont_female_expectations.append(expectation(this_mid,pf))
        
        # Update continous and protected attribute trackers
        continuous_attribute_tracker=continuous_attribute_tracker+1
        dependent_attribute_tracker=dependent_attribute_tracker+1
    
    elif encoding_list[attribute]==0 and dependency_list[attribute]==0:
        
        # Update continous tracker ONLY.
        continuous_attribute_tracker=continuous_attribute_tracker+1

    elif dependency_list[attribute]==1 and encoding_list[attribute]==1:
        # this implies it is a dependent categorical, and we need to extract the probabilities that they are positive.
        
        min_index=cumulative_dependent[dependent_attribute_tracker]
        max_index=cumulative_dependent[dependent_attribute_tracker+1]
        pm = probabilities_male[min_index:max_index]
        pf = probabilities_female[min_index:max_index]
        
        all_categoric_male_probs.append(pm)
        all_categoric_female_probs.append(pf)
        # should now have list of categorical probabilities as well
        
        # Somehow need to include binaries as well.
        
        # Update dependency tracker ONLY.
        dependent_attribute_tracker=dependent_attribute_tracker+1
    
print (all_cont_male_expectations)
print (all_cont_female_expectations)

# Start work again from here.

# Step 1 - import "Unfair Data".

raw_train_data=array_preprocess("unfair_training_data.csv")
raw_valid_data=array_preprocess("unfair_validation_data.csv")
raw_test_data=array_preprocess("unfair_testing_data.csv")


# Step 2 - subtract off numeric expectations.

list_of_learning_lengths=[4,1,5,10,1,5,5,1,1,3,1,4,1,3,3,1,4,1,1,1,1]

cumulative_learning=[0]
for l in list_of_learning_lengths:
    cumulative_learning.append(cumulative_learning[-1]+l)
print(cumulative_learning)

male_subtract_numeric=[0 for i in range(sum(list_of_learning_lengths))]
female_subtract_numeric=[0 for i in range(sum(list_of_learning_lengths))]

numeric_dependent_tracker=0

# Pseudocode:
# if encode is numeric and dependent is yes:
    # change element cumulative_learning[this - 1] to expectations[numeric_dependent_tracker]
# else don't bother.

for attribute, indicator in enumerate(dependency_list):
    if encoding_list[attribute]==0 and indicator==1:
        male_subtract_numeric[cumulative_learning[attribute]]=all_cont_male_expectations[numeric_dependent_tracker]
        female_subtract_numeric[cumulative_learning[attribute]]=all_cont_female_expectations[numeric_dependent_tracker]
        numeric_dependent_tracker=numeric_dependent_tracker+1

print(female_subtract_numeric)
print(male_subtract_numeric)



# Step 3 - now calculate Expectation for each categorical attribute.

male_subtract_categorical=[0 for i in range(sum(list_of_learning_lengths))]
female_subtract_categorical=[0 for i in range(sum(list_of_learning_lengths))]

# Think Carefully about how to do this - set up separate "Categorical Expectations" calculator.

for attribute, position in enumerate(prob_cumulative):

    if encoding_list[attribute]==1 and dependency_list[attribute]==1:
        #if this is the case then we have a dependent categorical variable.
        # we now need to fish out the section of the probabilities vector which corresponds to the categorical.
        min_index=cumulative_dependent[dependent_attribute_tracker]
        max_index=cumulative_dependent[dependent_attribute_tracker+1]

for attribute, indicator in enumerate(dependency_list):
    if encoding_list[attribute]==1 and indicator==1:
        male_subtract_numeric[cumulative_learning[attribute]:cumulative_learning[attribute+1]] = probabilities_male[]





# Step 4 - subtract off categorical attribute.






## Post processsing:

# Import data and turn it into the "one-hot categorical, discrete stay constant" format.
# Adjust the discrete data by subtracting off the expectations.
# Adjust the categorical data EITHER by adjusting using a MIXTURE MODEL, or alternatively by simply subtracting the raw conditional mean of each presented attribute. In fact, why not try both??
# Remember to do this ONLY for the conditionally dependent attributes.
# Also, remember to divide the dataset into the same categories as before. (Train,Val,Test).
# Have a look at how the different models perform on the validation set.


# NOTE TO SELF: In introduction - make the distinction between preprocessing and intraprocessing.


# The only continuous protected attributes which happen to exhibit high statistical dependency are AGE (65-74), DEPENDENTS (89-90) and PERCENTAGE OF INCOME (57-61).




# We now isolate this attribute and compute the expectation:

# Target array should be a combination of ONE HOT encoded categoricals and real values continuous attribute.


# The following is a list of all variables which exhibit dependency on the SPN - this can be modified to an import statement later.

# For Categorical Variables, E(x_j|a) = P(x_j=1|a).


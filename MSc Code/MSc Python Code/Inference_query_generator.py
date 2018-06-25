# Require individual inference queries of the form P(q|e):


# Number of columns occupied by each attribute
list_of_lengths=[4,10,5,10,10,5,5,4,1,3,4,4,10,3,3,4,4,2,1,1,1]

# Get column indices for where each attribute begins and ends
cumulative=[0]
for i,j in enumerate(list_of_lengths):
    cumulative.append(cumulative[i]+j)
print(cumulative)

total_number = sum(list_of_lengths) # 94
template=["*" for j in range(total_number)] # * indicates loop over variable

queries=[]
evidence_male=[]
evidence_female=[]

""" For every attribute component, set up a blank template, and then specify the configuration of the query we want to make for ONE ATTRIBUTE (e.g. [1,0,0,0,*,*,*,*,...])"""

for inference_instance in range(total_number):
    # Template
    temporary=["*" for j in range(total_number)]
    for position,threshold in enumerate(cumulative):
        # Only perform procedure if the inference_instance is in the range of the attribute we're currently considering
        if (inference_instance<threshold) and (inference_instance>=cumulative[position-1]):
            
            # Procedure for binary variables (must do [*,*,*,0,*] and [*,*,*,1,*]).
            if (threshold-cumulative[position-1])==1:
                # Set up two queries, one with a 0 in the relevant position and the other with a one.
                temporary_1=["*" for j in range(total_number)]
                temporary_2=["*" for j in range(total_number)]
                temporary_1[inference_instance]=0
                temporary_2[inference_instance]=1
                
                # Append to list of queries
                queries.append(temporary_1)
                queries.append(temporary_2)
                
                # Now allocate an evidence query (1 or 0 in protected attribute column
                temporary_evidence_1=["*" for j in range(total_number)]
                temporary_evidence_F1=["*" for j in range(total_number)]
                temporary_evidence_1[53]=0
                temporary_evidence_F1[53]=1
                
                # Append to evidence
                evidence_male.append(temporary_evidence_1)
                evidence_male.append(temporary_evidence_1)
                evidence_female.append(temporary_evidence_F1)
                evidence_female.append(temporary_evidence_F1)
        
            # Procedure for one-hot encoded variables (e.g. [*,*,0,0,1,0,*,*]
            else:
                temp_range=range(cumulative[position-1],threshold)
                
                # Specify all zeros in range of attribute.
                for index in temp_range:
                    temporary[index]=0
                
                # Set one specific column of these zeros to be 1
                temporary[inference_instance]=1
                
                # Append to queries
                queries.append(temporary)
                
                # Generate corresponding evidence input and append
                temporary_evidence_1=["*" for j in range(total_number)]
                temporary_evidence_F1=["*" for j in range(total_number)]
                temporary_evidence_1[53]=0
                temporary_evidence_F1[53]=1
                evidence_male.append(temporary_evidence_1)
                evidence_female.append(temporary_evidence_F1)

#print(queries)



import csv

csvfile="single_attribute_query.q"
male_evidencefile="protected_attribute_evidence_male.ev"
female_evidencefile="protected_attribute_evidence_female.ev"

# Write queries and evidence to CSV files for input to SPN Inference task.
with open(csvfile,"w") as output:
    writer=csv.writer(output,lineterminator='\n')
    writer.writerows(queries)

with open(male_evidencefile,"w") as output:
    writer=csv.writer(output,lineterminator='\n')
    writer.writerows(evidence_male)

with open(female_evidencefile,"w") as output:
    writer=csv.writer(output,lineterminator='\n')
    writer.writerows(evidence_female)

# Pseudocode for inference.

# Ascertain set {v_i,...,v_j} \in s such that v_i || protected attribute a.

# Write separate python script to clean out all statistically INDEPENDENT variables, so we just train a vanilla SPN with data subdivision in first step on just the dependent variables.

# Write to array P(q|ev) for all q.

# Calculate (again using python script) \mathbb{E}(q|ev) for all continuous variables.

# Back out P(q|ev) for all categoricals.

# Train (on all variables, including the modified ones) a model for predicting some attribute on the basis of the existing set.

# I've had another idea: what if you TRAIN on unadjusted variables, but then perform TESTING on adjusted counterparts??
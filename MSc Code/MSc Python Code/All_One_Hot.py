# MSc Project

# Data pre-processing code: binning and one-hot augmentation.

# Steps: 1. Download Data Set, 2. Sort it out, bin it, one-hot-encode-it, and knit it back together again. 3. Ensure it reads out a CSV file containing nicely arranges ones and zeros for the SPN to read.

# Bin algorithm

# Read Data
import csv

textfile=open("German_credit_space_sep.txt","r")
German_Raw=textfile.read().split('\n')
print("German_Raw")
print(German_Raw[1])

german_array=[]
for j in German_Raw:
    k=j.split(' ')
    german_array.append(k)
print("german_array")
print(german_array[1])

# We now have an array as required.

def bins(data_col, num_of_bins):
    
    # determine maximum and minimum values in data column

    maximum = max(data_col)
    minimum = min(data_col)
    
    # split into bins, marked by interval boundaries
    interval = (maximum-minimum)/num_of_bins
    interval_list = [minimum+(l*interval) for l in range((num_of_bins+1))]

    new_data_col=[]
    
    # allocate each data point to a bin.
    for data_point in data_col:
        j = 0
        int_marker=interval_list[j+1]
        while (data_point>int_marker):
            j=j+1
            int_marker=interval_list[j+1]
        new_val=j
        # produce new column of data containing the bin number for each data point.
        new_data_col.append(new_val)
    return new_data_col

#data_col_raw=range(0,18)
#out=bins(data_col_raw,6)
#test_list=[-37,42,32,12,11,4,27,6,-7,3,3]
#out_2=bins(test_list,3)

# One_hot_algorithm

def one_hot(data_col):
    # create set of all the different class labels - might be strings
    value_set = set(data_col)
    # put these in a list
    list_of_values=list(value_set)
    
    #### print to see if some instances are mislabeled
    
    # determine how many separate classes there are
    num_classes=len(list_of_values)
    # create empty list
    one_hot_list=[]
    # create an array - each entry in the list is itself a list, and each such list is a one-hot encoder of the classes
    for i in range(num_classes):
        temp_one_hot= [0 for j in range(num_classes)]
        temp_one_hot[i]=1
        one_hot_list.append(temp_one_hot)
    # Hash map between the list of values and an integer
    one_hot_encoder_lookup=dict(zip(list_of_values,range(num_classes)))
    classifier_list=[]
    
    for value in data_col:
        # integer number from hashmap used to pull a particular list from the list of one-hot encodings
        numeric_encoder=one_hot_encoder_lookup.get(value)
        one_hot_encoder=one_hot_list[numeric_encoder]
        classifier_list.append(one_hot_encoder)

    return classifier_list

### Binary encoder (one-hot for two variables)

def binary_encoder(data_col):

    value_set = set(data_col)

    list_of_values=list(value_set)

    class_id_lookup=dict(zip(list_of_values,[0,1]))
    
    classifier_list=[]

    for value in data_col:
        classifier_list.append([class_id_lookup.get(value)])
    #print(classifier_list)
    return classifier_list



################

# Data types:
# 1 - encodes categorical
# 0 - encodes numeric

encoding_list=[1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,2,2,2]

print(len(encoding_list))

processed_data=[]
german_array=german_array[:-1]

for i in range(len(german_array[1])):
    data_category_list=[row[i] for row in german_array]
    
    if (i==8): # attribute 9 - male/female
        # sort out the problem involving allocating men to one category and women to another.
        new_list=[]
        for entry in data_category_list:
            if (entry == 'A91' or entry == 'A93' or entry == 'A94'):
                new_list.append([0]) # Male
            else:
                new_list.append([1]) # Female
        processed_data.append(new_list)
    
    # Bin numerical data if numeric
    elif (encoding_list[i] == 0):
        new_list=[]
        for entry in data_category_list:
            new_list.append(int(entry)) # does this work????
        discrete_numeric_list=bins(new_list,10)
        one_hot_bins=one_hot(discrete_numeric_list)
        processed_data.append(one_hot_bins)
    
    # One-hot encode if categorical with more than two classes
    elif (encoding_list[i] == 1):
        one_hot_category_list=one_hot(data_category_list)
        processed_data.append(one_hot_category_list)

    # Binary encode if categorical with two classes.
    else:
        binary_category_list=binary_encoder(data_category_list)
        processed_data.append(binary_category_list)

# processed_data should now be an array consisting of 20 lists, each of length 1000, the contents of which are ALL one-hot encoded(regardsless of whether they started kufe as categorical or numerical).

"""In theory, the only thing remaining to do is test this, return it to the original format (1000 of 20 rather than 20 of 1000), and save it as a csv file. Then we can consider next steps involving how to give it to the LearnSPN algortihm, and from there how to perform inference. I think it's worth waiting until I have Andreas's help."""

# Turn processed_data into 1000*20 (not 20*1000).

reshaped_processed_data=[]

for data_point_number in range(len(processed_data[1])):
    reshaped_processed_data.append([processed_data[i][data_point_number] for i in range(len(encoding_list))])

print(reshaped_processed_data[0:4])

reshaped_processed_data # this is the final thing - fully binned and ALL one-hot encoded.

listless_rows_reshaped_processed_data=[]
# Remove lists internally.
for row_number, row in enumerate(reshaped_processed_data):
    listless_rows_reshaped_processed_data.append([])
    for list_element in row:
        for binary_variable in list_element:
            listless_rows_reshaped_processed_data[row_number].append(binary_variable)

print(listless_rows_reshaped_processed_data[0]) # Yep. this works
print(len(listless_rows_reshaped_processed_data[0]))
print(len(listless_rows_reshaped_processed_data))

# Split into training, testing and validation.

german_credit_train=listless_rows_reshaped_processed_data[0:699] #700
german_credit_valid=listless_rows_reshaped_processed_data[700:849] #150
german_credit_test=listless_rows_reshaped_processed_data[850:999] #150

# Now just need to write this to csv file.

csvfile="one_hot_german_credit.csv"

testfile="germancredit.test.data"
validfile="germancredit.valid.data"
trainfile="germancredit.ts.data"

with open(testfile,"w") as output:
    writer = csv.writer(output,lineterminator='\n')
    writer.writerows(german_credit_test)

with open(validfile,"w") as output:
    writer = csv.writer(output,lineterminator='\n')
    writer.writerows(german_credit_valid)

with open(trainfile,"w") as output:
    writer = csv.writer(output,lineterminator='\n')
    writer.writerows(german_credit_train)


# We now need to just form a list rather than a list of lists.

# Binary variable at top

###############

list_2=["yellow","red","yellow","red","red","green","yellow","red","red","green","red"]

out_3=one_hot(list_2)
#print(out_3)

"""

# Idea: within the confines of the German Data set, we have:
# 1. A female/male divide
# 2. A native/foreign divide.

# Let's consider how protecting attributes which turn out to be good indicators of the thing we're trying to predict.

# Also, let's suppose x_j is binary. x_j = 1 or 0. Then x_j'=x_j-P(x_j=1|a)???
# Try it and see if it works.

# Several models are starting to emerge: train on adjusted continous only, train on both adjusted continous and adjusted binary.
# Train on the basis of some sort of adjusted one-hot encoded representation. e.g. P(x_j|a)

# Another idea. suppose we have a binary attribute x and another PROTECTED binary attribute a. Now suppose we adjust: x_j' = x_j-P(x_j=1|a). Question - could a complicated neural net learn the difference and learn to re-factor in the protected variable - i.e if more complicated models can improve prediction performance, then can they also predict the protected and use it to train on??

# YET ANOTHER IDEA - AND I THINK THIS IS A GOOD ONE. In view of the fact that we seek to learn only an APPROXIMATION of the true underlying probability distribution, assign a very high penalty to clustering in the first step. This should allow us to separate out a large number of variables in the first instance.

"""


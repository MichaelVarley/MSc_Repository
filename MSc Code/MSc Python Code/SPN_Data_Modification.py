# MSc Project

# Data pre-processing code: binning and one-hot augmentation.

# Steps: 1. Download Data Set, 2. Sort it out, bin it, one-hot-encode-it, and knit it back together again. 3. Ensure it reads out a CSV file containing nicely arranges ones and zeros for the SPN to read.

# Bin algorithm

# Read Data
import csv

textfile=open("German_credit_space_sep.txt","r")
German_Raw=textfile.read().split('\n') # List of long strings containing all text.
print("German_Raw")
print(German_Raw[1])

german_array=[]
for j in German_Raw:
    k=j.split(' ')
    german_array.append(k)
print("German_Array")
print(german_array[1])

# List of subdivided lists, such that each "row" corresponds to a particular individual, and each column to an attribute

# We now have an array as required.



""" SUBROUTINES: Next section contains subroutines for binning, one-hot encoding and binary encoding of variables"""
#############

# BINNING DATA
def bins(data_col, num_of_bins):
    # determine maximum and minimum values in data column
    maximum = max(data_col)
    minimum = min(data_col)
    
    # split into bins, marked by interval boundaries
    interval = (maximum-minimum)/num_of_bins
    interval_list = [minimum+(l*interval) for l in range((num_of_bins+1))]
    # [min,min+interval,min+2*interval,...,max]
    print("intervals")
    print(interval_list)
    
    
    # Write interval list to file.

    new_data_col=[]
    
    # allocate each data point to a bin.
    for data_point in data_col:
        j = 0
        int_marker=interval_list[1]
        while (data_point>int_marker):
            j=j+1
            int_marker=interval_list[j+1]
        new_val=j
        # produce new column of data containing the bin number for each data point.
        new_data_col.append(new_val)
    return new_data_col, interval_list




# CREATES LIST OF ALL UNIQUE INSTANCES IN ANOTHER LIST
def unique(l):
    u=[]
    for i in l:
        if i not in u:
            u.append(i)
    return u
# returns ordered list - this could be made more efficient.




# SORTING ALGORITHM - CAN BE USED TO SORT BY NUMBER OR ALPHABETICALLY
def bubble_sort(string_list):
    for i in range(len(string_list)):
        for j in range(i+1,len(string_list)):
            if string_list[i]>string_list[j]:
                string_list[i],string_list[j]=string_list[j],string_list[i]
    return string_list
# Double check this algo.




# ONE HOT ENCODING ALGORITHM.
def one_hot(data_col):
    
    # create set of all the different class labels - might be strings; put these in a list
    list_of_values=bubble_sort(unique(data_col))
    
    print(list_of_values)
    
    #### CAN print to see if some instances are mislabeled
    
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

### BINARY ENCODER (one-hot for two variables)
def binary_encoder(data_col):
    
    list_of_values=bubble_sort(unique(data_col))
    
    class_id_lookup=dict(zip(list_of_values,[0,1]))
    
    classifier_list=[]
    
    for value in data_col:
        classifier_list.append([class_id_lookup.get(value)])
    #print(classifier_list)
    return classifier_list


################

""" END OF SUBROUTINES"""

""" TEST SECTION: Checks that all subroutines are behaving as they're supposed to """

print("TEST RESULTS")

test_string=['this','is','a','sentence']
test_numbers=[44,21,3,132,64,2,33,-4]
print(bubble_sort(test_string))
print(bubble_sort(test_numbers))

test_list=[0,100,22,54,33,11,2,4,76,70,80,90,10,10.001,65,12,44,18,92,63,32,28]
print(bins(test_list,10))

test_unique=[4,4,2,1,4,3,77,2,11,6]
print(unique(test_unique))

print("END OF TEST RESULTS")


############

""" END OF TESTS """



""" PERFORM ONE-HOT ENCODING ETC. ON MAIN DATA SET """

# Data types:
# 2 - encodes binary categorical - STRING
# 1 - encodes multiple categorical - NUMBER
# 0 - encodes numeric

encoding_list=[1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,2,2,2]
# This needs to be changed depending on the data set.
# In the case of the German Credit data set, the first attribute is categorical, the second numerical, the third categorical etc.
# This could be automated.


print("Total number of attributes according to the encoding list is: " +str(len(encoding_list)))
print("Actual number of attributes is: " + str(len(german_array[1])))

processed_data=[]
german_array=german_array[:-1] # remove redundant final row.
all_interval_lists=[]

for i in range(len(german_array[1])):
    data_category_list=[row[i] for row in german_array] # Extract columns
    
    if (i==8): # attribute 9 - male/female
        
        # sort out the problem involving allocating men to one category and women to another.
        new_list=[]
        for entry in data_category_list:
            if (entry == 'A91' or entry == 'A93' or entry == 'A94'):
                new_list.append([0]) # Male
            else:
                new_list.append([1]) # Female

        processed_data.append(new_list) # Append the new binary encoded list.
    
    # Bin numerical data if numeric and then one hot encode
    elif (encoding_list[i] == 0):
        new_list=[]
        for entry in data_category_list:
            new_list.append(int(entry)) # does this work????
        discrete_numeric_list, current_intervals=bins(new_list,10)
        all_interval_lists.append(current_intervals)
        one_hot_bins=one_hot(discrete_numeric_list)
        processed_data.append(one_hot_bins)
        # Append list of one-hot encoded attributes [[0,1,0],[0,0,1],[1,0,0],...]

    # One-hot encode if categorical with more than two classes
    elif (encoding_list[i] == 1):
        one_hot_category_list=one_hot(data_category_list)
        processed_data.append(one_hot_category_list)

    # Binary encode if categorical with two classes.
    else:
        binary_category_list=binary_encoder(data_category_list)
        processed_data.append(binary_category_list)

""" The array 'processed_data' is now an array comprising 20 lists.' Each list is of length 1000. 
    Every element of each list is a one-hot encoding (regardless of whether they started life as categorical or numerical)."""

"""In theory, the only thing remaining to do is test this, return it to the original format (1000 of 20 rather than 20 of 1000), and save it as a csv file. Then we can consider next steps involving how to give it to the LearnSPN algortihm, and from there how to perform inference. I think it's worth waiting until I have Andreas's help."""




"""RESHAPE DATA BACK INTO ORIGINAL FORMAT"""

# Turn processed_data into 1000*20 (not 20*1000).

# Create blank array
reshaped_processed_data=[]

# for 1:1000:
for data_point_number in range(len(processed_data[1])):
    # for 1:num_attributes
    reshaped_processed_data.append([processed_data[i][data_point_number] for i in range(len(encoding_list))])

print(reshaped_processed_data[0:1])

reshaped_processed_data # this is the final thing - fully binned and ALL one-hot encoded.





""" REMOVE INTERNAL LISTS SO IT CAN BE READ BY SPN """

listless_rows_reshaped_processed_data=[]
# Remove lists internally.
for row_number, row in enumerate(reshaped_processed_data):
    listless_rows_reshaped_processed_data.append([])
    for list_element in row:
        for binary_variable in list_element:
            listless_rows_reshaped_processed_data[row_number].append(binary_variable)

print("First row: "+ str(listless_rows_reshaped_processed_data[0])) # Yep. this works
print("Number of attributes as seen by SPN is: "+ str(len(listless_rows_reshaped_processed_data[0])))
print("Number of Data Points: " + str(len(listless_rows_reshaped_processed_data)))





"""SPLIT INTO TRAINING, VALIDATION AND TESTING SETS."""

german_credit_train=listless_rows_reshaped_processed_data[0:699] #700
german_credit_valid=listless_rows_reshaped_processed_data[700:849] #150
german_credit_test=listless_rows_reshaped_processed_data[850:999] #150



"""WRITE TO CSV FILE"""

csvfile="one_hot_german_credit.csv"

intervalfile="germancredit.intervals.data"
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

with open(intervalfile,"w") as output:
    writer = csv.writer(output,lineterminator='\n')
    writer.writerows(all_interval_lists)





""" VARIOUS IDEAS

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


# MSc Project

# This imports the relevant data and now processes it differently.

# As before categorical attributes are one-hot encoded, but now the numerical attributes are left as they are.

# Step 1: Read Data
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


# Import all required functions from the original file.

from SPN_Data_Modification import bins, unique, one_hot, bubble_sort, binary_encoder

""" PERFORM ONE-HOT ENCODING ON CATEGORICAL VARIABLES AND CONVERT NUMERICS """

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

processed_data_with_numeric=[]
german_array=german_array[:-1] # remove redundant final row.

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

        processed_data_with_numeric.append(new_list) # Append the new binary encoded list.
    
    # Bin numerical data if numeric and then one hot encode
    elif (encoding_list[i] == 0):
        new_list=[]
        for entry in data_category_list:
            new_list.append([int(entry)]) # does this work????
        processed_data_with_numeric.append(new_list)
        # Append list of one-hot encoded attributes [[0,1,0],[0,0,1],[1,0,0],...]

    # One-hot encode if categorical with more than two classes
    elif (encoding_list[i] == 1):
        one_hot_category_list=one_hot(data_category_list)
        processed_data_with_numeric.append(one_hot_category_list)

    # Binary encode if categorical with two classes.
    else:
        binary_category_list=binary_encoder(data_category_list)
        processed_data_with_numeric.append(binary_category_list)

""" The array 'processed_data_with_numeric' is now an array comprising 20 lists.' Each list is of length 1000.
    Every element of each list is EITHER a one-hot encoding (if categorical) OR list comprising SINGLE number if numeric or binary categorical."""

"""In theory, the only thing remaining to do is test this, return it to the original format (1000 of 20 rather than 20 of 1000), and save it as a csv file."""

"""RESHAPE DATA BACK INTO ORIGINAL FORMAT"""

# Turn processed_data into 1000*20 (not 20*1000).

# Create blank array
reshaped_numeric_data=[]

# for 1:1000:
for data_point_number in range(len(processed_data_with_numeric[1])):
    # for 1:num_attributes
    reshaped_numeric_data.append([processed_data_with_numeric[i][data_point_number] for i in range(len(encoding_list))])

print("reshaped_numeric_data first two rows")

print(reshaped_numeric_data[0:1])

reshaped_numeric_data # this is the final thing - fully binned and ALL one-hot encoded.


""" REMOVE INTERNAL LISTS SO IT CAN BE READ BY SPN """

listless_rows_reshaped_numeric_data=[]
# Remove lists internally.
for row_number, row in enumerate(reshaped_numeric_data):
    listless_rows_reshaped_numeric_data.append([])
    for list_element in row:
        for binary_variable in list_element:
            listless_rows_reshaped_numeric_data[row_number].append(binary_variable)

print("First row: "+ str(listless_rows_reshaped_numeric_data[0])) # Yep. this works
print("Number of attributes as seen by SPN is: "+ str(len(listless_rows_reshaped_numeric_data[0])))
print("Number of Data Points: " + str(len(listless_rows_reshaped_numeric_data)))

# Start work again from here - now just need to subdivide the data and save it as "Unfair Training" or something.

# Then import to other script and remove expectations etc.

unfair_training_file = "unfair_training_data.csv"
unfair_validation_file = "unfair_validation_data.csv"
unfair_testing_file = "unfair_testing_data.csv"

unfair_train_array=listless_rows_reshaped_numeric_data[0:699]
unfair_valid_array=listless_rows_reshaped_numeric_data[700:849]
unfair_test_array=listless_rows_reshaped_numeric_data[850:999]

def write_to_external(file_id, array):
    with open(file_id,"w") as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerows(array)

write_to_external(unfair_training_file, unfair_train_array)
write_to_external(unfair_validation_file, unfair_valid_array)
write_to_external(unfair_testing_file, unfair_test_array)







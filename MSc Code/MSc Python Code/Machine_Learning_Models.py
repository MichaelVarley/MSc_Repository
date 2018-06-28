import csv
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# import data

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


train_array=array_preprocess("unfair_training_data.csv")
valid_array=array_preprocess("unfair_validation_data.csv")
test_array=array_preprocess("unfair_testing_data.csv")

fair_train_array=array_preprocess("fair_train_data.csv")
fair_valid_array=array_preprocess("fair_valid_data.csv")
fair_test_array=array_preprocess("fair_test_data.csv")

# Isolate attributes and labels.

protected_attribute_position=32

gender_list_train=[row[protected_attribute_position] for row in train_array]
gender_list_valid=[row[protected_attribute_position] for row in valid_array]
gender_list_test=[row[protected_attribute_position] for row in test_array]


# Don't train explicitly on the protected (remove column 8/9)
x_train = [row[:-1] for row in train_array]
x_fair_train=[row[:-1] for row in fair_train_array]
ftu_x_train=[row[0:32]+row[33:-1] for row in train_array]
y_train=[row[-1] for row in train_array]

x_valid = [row[:-1] for row in valid_array]
x_fair_valid=[row[:-1] for row in fair_valid_array]
ftu_x_valid=[row[0:32]+row[33:-1] for row in valid_array]
y_valid=[row[-1] for row in valid_array]

x_test = [row[:-1] for row in test_array]
x_fair_test=[row[:-1] for row in fair_test_array]
ftu_x_test=[row[0:32]+row[33:-1] for row in test_array]
y_test=[row[-1] for row in test_array]

# Train logistic regression model. (FTU = Fairness Through Unawareness)
unfair_log_reg_model = LogisticRegression()
unfair_log_reg_model.fit(x_train,y_train)
unfair_valid_predictions = unfair_log_reg_model.predict(x_valid)

ftu_log_reg_model=LogisticRegression()
ftu_log_reg_model.fit(ftu_x_train,y_train)
ftu_valid_predictions = ftu_log_reg_model.predict(ftu_x_valid)

fair_log_reg_model = LogisticRegression()
fair_log_reg_model.fit(x_fair_train,y_train)
fair_valid_predictions = fair_log_reg_model.predict(x_fair_valid)


# Compute function to calculate unfairness

def demographic_disparity_calculator(predictions,genders):
    n_female=sum(genders)
    n_male=len(genders)-sum(genders)
    male_prob=0
    female_prob=0
    for point, probability in enumerate(predictions):
        if genders[point] == 0:
            male_prob=male_prob+probability
        elif genders[point] == 1:
            female_prob=female_prob+probability
    demo_disparity=female_prob/n_female - male_prob/n_male
    return demo_disparity

# Could also add in an equality of opportunity calculator.

ftu_disparity= demographic_disparity_calculator(ftu_valid_predictions,gender_list_valid)
unfair_disparity= demographic_disparity_calculator(unfair_valid_predictions, gender_list_valid)
fair_disparity = demographic_disparity_calculator(fair_valid_predictions, gender_list_valid)

print(ftu_disparity)
print(unfair_disparity)
print(fair_disparity)



# def compute_fairness():
# sum (probability|male)/(number of males) - sum(probability|female)/(number_of_females) for everything in valid_array



# Things we still need to do:

# Compute function to calculate the unfairness.

#gnb = GaussianNB()
#gnb.fit(train_data,train_targets)
"""
def train_logistic_regression(train_data,train_targets):
    log_reg_model = LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
    log_reg_model.fit(train_data,train_targets)
    return log_reg_model
"""
#log_reg_model.predict()

#def train_GNB(train_data,train_targets):



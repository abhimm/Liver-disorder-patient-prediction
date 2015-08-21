__author__ = 'Abhinav'
import random
import math
import copy

def get_data(filename):
    f = open(filename, "rb")
    data = f.readlines()
    feature_list = list()
    result_list = list()
    for i in range(6):
        feature_list.append([])


    for record in data:
        temp = record.strip().split(",")
        result = int(temp[6])
        for i in range(6):
           feature_list[i].append(float(temp[i]))

        if result == 1:
            result_list.append(1)

        else:

            result_list.append(-1)
    return feature_list, result_list

def main():
    num_iter = 10

    train_file_name = "bupa.data"
    # get feature and result for test and train data
    train_feature_list, train_result_list = get_data(train_file_name)

    # build feature threshold list
    feature_threshold_list = list()
    for i in range(6):
        feature_threshold_list.append([])

    feature_list = list()
    for i in range(6):
        feature_list = copy.deepcopy(train_feature_list[i])
        feature_list.sort()
        previous = feature_list[0]
        for j in range(1,len(feature_list)):
            if previous == feature_list[j]:
                continue
            feature_threshold_list[i].append((float(previous) + float(feature_list[j]))/2.0)
            previous = feature_list[j]

    result = adaboost(train_feature_list, train_result_list, feature_threshold_list, num_iter)
    out_file = open("./part2_training_result.out", "wb")
    for i in range(num_iter):
        out_file.write("Boosting iteration: %d"%(i+1) +"\n")
        out_file.write("Training error: %f"%result[i][0] +"\n")
        out_file.write("Optimal feature: %d"%result[i][1] +"\n")
        out_file.write("Optimal threshold: %f"%result[i][2] +"\n")
        out_file.write("Class label: %f"%result[i][3] +"\n")
        out_file.write("--------------------------\n")
    out_file.close()


def adaboost(train_feature_list, train_result_list, train_feature_threshold_list, num_iter):
    return train_adaboost(train_feature_list, train_result_list, train_feature_threshold_list, num_iter)

def train_adaboost(train_feature_list, train_result_list, train_feature_threshold_list, num_iter):
    # no of samples
    n = len(train_result_list)
    D = list()
    alpha = list()
    #initialize weights for samples
    for i in range(n):
        D.append(1.0/n)

    Z = dict()
    for i in range(num_iter):
        Z[i] = 0
    training_error = 1
    for i in range(num_iter):
        alpha.append(0.0)
    iter_result = dict()
    for i in range(num_iter):

        optimal_feature, optimal_threshold, optimal_error, optimal_result, optimal_class_predicted = get_best_decision_stump(train_feature_list, train_result_list, train_feature_threshold_list, D)
        #print optimal_feature, optimal_threshold, optimal_error

        if optimal_error == 0.0:
            alpha[i] = float("inf")
        elif optimal_error == 1.0:
            alpha[i] = float("-inf")
        else:
            alpha[i] = 0.5*(math.log((1-optimal_error)/optimal_error))


        for j in range(len(D)):
            Z[i] += D[j]*math.exp(-1*optimal_result[j]*alpha[i])

        for j in range(len(D)):
            D[j] = D[j]*math.exp(-1*optimal_result[j]*alpha[i])/Z[i]
        #print "Iteration: %d"%(i+1), "Error: %f"%optimal_error, "Feature: %d"%optimal_feature, "Threshold: %f"%optimal_threshold

        training_error *= Z[i]

        iter_result[i] = (0.5- 0.5*optimal_error, optimal_feature, optimal_threshold, optimal_class_predicted)
    return iter_result


def get_best_decision_stump(train_feature_list, train_result_list, train_feature_threshold_list, D):
    optimal_error = float("inf")
    optimal_feature = - 1
    optimal_threshold = 0.0
    optimal_result = list()
    optimal_class_predicted = 0
    for i in range(6):
       feature_threshold_list = train_feature_threshold_list[i]
       best_local_error = float("inf")
       best_local_threshold = 0.0
       best_local_result = list()
       best_local_class_predicted = 0
       for threshold in feature_threshold_list:
           local_error = 0
           local_result = list()
           local_class_predicted = 0

           local_error_left = 0
           local_error_right = 0
           local_result_right = list()
           local_result_left = list()
           class_predicted_left = 0
           class_predicted_right = 0
           for j in range(len(train_feature_list[i])):

               value = train_feature_list[i][j]

               if float(value) < threshold:
                   class_predicted_left = -1
                   class_predicted_right = 1
               else:
                   class_predicted_left = 1
                   class_predicted_right = -1

               if not class_predicted_left ==  train_result_list[j]:
                   local_error_left += D[j]
                   local_result_left.append(-1)
               else:
                   local_result_left.append(1)

               if not class_predicted_right ==  train_result_list[j]:
                   local_error_right += D[j]
                   local_result_right.append(-1)
               else:
                   local_result_right.append(1)

           if local_error_left < local_error_right:
               local_error = local_error_left
               local_result = local_result_left
               local_class_predicted = 1
           else:
               local_error = local_error_right
               local_result = local_result_right
               local_class_predicted = -1

           if local_error < best_local_error:
               best_local_error = local_error
               best_local_threshold = threshold
               best_local_result = local_result
               best_local_class_predicted = local_class_predicted
       #print "for feature: %d"%i, "best local error: %f"%best_local_error , "optimal error: %f"%optimal_error
       if optimal_error > best_local_error:
           optimal_error = best_local_error
           optimal_feature = i
           optimal_threshold = best_local_threshold
           optimal_result = best_local_result
           optimal_class_predicted = best_local_class_predicted
    return optimal_feature, optimal_threshold, optimal_error, optimal_result, optimal_class_predicted



if __name__ == '__main__':
    main()
__author__ = 'Abhinav'
import random
import math
import copy

def gen_data_set(num_sets, training_data_percentage):
    bupa_file = open("./bupa.data", "rb")
    bupa_data = bupa_file.read().split('\n')
    train_data_size = int(training_data_percentage*len(bupa_data))

    for i in range(num_sets):
        random.shuffle(bupa_data)
        train_data = bupa_data[:train_data_size][:]
        test_data = bupa_data[train_data_size:][:]

        train_data_file_name = "train_data_%d"%i + ".data"
        train_data_file = open(train_data_file_name, "wb")
        for j in range(len(train_data)):
            if train_data[j] == "":
                continue
            train_data_file.write(train_data[j]+"\n")
        train_data_file.close()

        test_data_file_name = "test_data_%d"%i + ".data"
        test_data_file = open(test_data_file_name, "wb")
        for j in range(len(test_data)):
            if test_data[j] == "":
                continue
            test_data_file.write(test_data[j]+"\n")
        test_data_file.close()

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
    num_sets = 50
    split_percentage = 0.9
    num_iter = 100
    # generate the 50 data sets
    gen_data_set(num_sets, split_percentage)
    set_train_result = dict()
    set_test_result = dict()
    for i in range(num_sets):
        train_file_name = "train_data_%d"%i + ".data"
        test_file_name = "test_data_%d"%i + ".data"

        # get feature and result for test and train data
        train_feature_list, train_result_list = get_data(train_file_name)
        test_feature_list, test_result_list = get_data(test_file_name)

        # build feature threshold list
        feature_threshold_list = list()
        for j in range(6):
            feature_threshold_list.append([])
        feature_list = list()

        for j in range(6):
            feature_list = copy.deepcopy(train_feature_list[j])
            feature_list.sort()
            previous = feature_list[0]
            for k in range(1,len(feature_list)):
                if previous == feature_list[k]:
                    continue
                feature_threshold_list[j].append((float(previous) + float(feature_list[k]))/2.0)
                previous = feature_list[k]

        print "Running Adaboost for training and test data set index: %d"%i
        set_train_result[i], set_test_result[i] = adaboost(train_feature_list, train_result_list, feature_threshold_list, num_iter
                                       , test_feature_list, test_result_list)



    avg_result = dict()
    for i in range(num_iter):
        avg_result[i] = 0.0

    for i in range(num_iter):
        for j in range(num_sets):
            temp_set_result = set_train_result[j]

            avg_result[i] = avg_result[i] + temp_set_result[i]

    for j in range(num_iter):
        avg_result[j] /= num_sets

    out_train_file = open("./out_train.data", "wb")
    for j in range(num_iter):
        out_train_file.write(str(j+1) + "," +str(avg_result[j])+"\n")
    out_train_file.close()

    avg_test_result = dict()
    for i in range(num_iter):
        avg_test_result[i] = 0.0

    for i in range(num_iter):
        for j in range(num_sets):
            temp_set_result = set_test_result[j]
            avg_test_result[i] += temp_set_result[i]

    for j in range(num_iter):
        avg_test_result[j] /= num_sets

    out_test_file = open("./out_test.data", "wb")
    for j in range(num_iter):
        out_test_file.write(str(j+1) + "," +str(avg_test_result[j])+"\n")
    out_test_file.close()

def adaboost(train_feature_list, train_result_list, train_feature_threshold_list, num_iter, test_feature_list,
             test_result_list):
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
    train_result = dict()
    test_error_result = dict()
    for i in range(num_iter):

        optimal_feature, optimal_threshold, optimal_error, optimal_result, optimal_side = get_best_decision_stump(train_feature_list, train_result_list, train_feature_threshold_list, D)
        #print optimal_feature, optimal_threshold, optimal_error

        if optimal_error == 0.0:
            alpha[i] = float("inf")
        elif optimal_error == 1.0:
            alpha[i] = float("-inf")
        else:
            alpha[i] = 0.5*(math.log((1-optimal_error)/optimal_error))


        for j in range(len(D)):
            Z[i] += D[j]*math.exp(-1*optimal_result[j]*alpha[i])
        #print D
        for j in range(len(D)):
            D[j] = D[j]*math.exp(-1*optimal_result[j]*alpha[i])/Z[i]
        #print "Iteration: %d"%(i+1), "Error: %f"%optimal_error, "Feature: %d"%optimal_feature, "Threshold: %f"%optimal_threshold
        training_error *= Z[i]
        iter_result[i] = 0.5 - 0.5*optimal_error
        train_result[i] = (optimal_feature, optimal_threshold, optimal_side, alpha[i])
        test_error_result[i] = test_adaboost(test_feature_list, test_result_list, train_result, i)
    return iter_result, test_error_result

def test_adaboost(test_feature_list, test_result_list, train_result, num_iteration):
    mis_classified_instance = 0.0

    for i in range(len(test_result_list)):
        h_value = 0
        for j in range(num_iteration+1):
            feature_value = test_feature_list[train_result[j][0]][i]
            if feature_value < train_result[j][1]:
                if train_result[j][2] == "left":
                    h_value += -1*train_result[j][3]
                else:
                    h_value += train_result[j][3]
            else:
                if train_result[j][2] == "left":
                    h_value += train_result[j][3]
                else:
                    h_value += -1*train_result[j][3]

        if h_value < 0 and test_result_list[i] == 1:
            mis_classified_instance += 1
        elif h_value > 0 and test_result_list[i] == -1:
            mis_classified_instance +=1
    return mis_classified_instance/len(test_result_list)

def get_best_decision_stump(train_feature_list, train_result_list, train_feature_threshold_list, D):
    optimal_error = float("inf")
    optimal_feature = - 1
    optimal_threshold = 0.0
    optimal_result = list()
    optimal_side = ""
    for i in range(6):
       feature_threshold_list = train_feature_threshold_list[i]
       best_local_error = float("inf")
       best_local_threshold = 0.0
       best_local_result = list()
       best_side = ""
       for threshold in feature_threshold_list:
           local_error = 0
           local_result = list()

           local_error_left = 0
           local_error_right = 0
           local_result_right = list()
           local_result_left = list()
           local_side = ""
           for j in range(len(train_feature_list[i])):
               class_predicted_left = 0
               class_predicted_right = 0
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
               local_side = "left"
           else:
               local_error = local_error_right
               local_result = local_result_right
               local_side = "right"

           if local_error < best_local_error:
               best_local_error = local_error
               best_local_threshold = threshold
               best_local_result = local_result
               best_side = local_side
       #print "for feature: %d"%i, "best local error: %f"%best_local_error , "optimal error: %f"%optimal_error
       if optimal_error > best_local_error:
           optimal_error = best_local_error
           optimal_feature = i
           optimal_threshold = best_local_threshold
           optimal_result = best_local_result
           optimal_side = best_side
    return optimal_feature, optimal_threshold, optimal_error, optimal_result, optimal_side



if __name__ == '__main__':
    main()
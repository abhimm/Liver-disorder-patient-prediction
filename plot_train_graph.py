__author__ = 'Abhinav'
__author__ = 'Abhinav'
import matplotlib.pyplot as pyplot

train_file = open("./out_train.data", "rb")
test_file = open("./out_test.data", "rb")
train_data = train_file.readlines()
test_data = test_file.readlines()

plotX_train = list()
plotX_test = list()
plot_test = list()
plot_train = list()
for record in train_data:
    temp = record.strip().split(",")
    plotX_train.append(int(temp[0]))
    plot_train.append(float(temp[1]))


for record in train_data:
    temp = record.strip().split(",")
    plotX_test.append(int(temp[0]))
    plot_test.append(float(temp[1]))


pyplot.axis([0, 105, 0.2, 0.5])
pyplot.title("Test and Training for Adaboost")
pyplot.xlabel("Iteration")
pyplot.ylabel("Error")
pyplot.plot(plotX_train, plot_train, label='$Train Error$')
pyplot.savefig("train_error_graph.png")
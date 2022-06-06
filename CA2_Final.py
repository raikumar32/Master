# Required Library to implement this code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading data from the file
d1 = pd.read_csv('./animals', delimiter=' ',header=None)
d2 = pd.read_csv('./countries',  delimiter=' ', header=None,)
d3 = pd.read_csv('./fruits',  delimiter=' ', header=None,)
d4 = pd.read_csv('./veggies',  delimiter=' ', header=None,)
# creating a new index and Delete column to be used as the new index.
d1 = d1.set_index(0, drop=True)
d2 = d2.set_index(0, drop=True)
d3 = d3.set_index(0, drop=True)
d4 = d4.set_index(0, drop=True)
# Combining all four DataFrame into a single DataFrame and shuffling it.
dataframe = pd.concat([d1, d2, d3, d4])
# A random 100% sample of the DataFrame with no replacement and random state as 45
merged_dataset = dataframe.sample(frac=1,random_state=45)

# Converting the DataFrame into a numpy array
numpy_array = merged_dataset.to_numpy()

# Q1
"""
Implement the k-means clustering algorithm to cluster the instances into k clusters.
"""


def euclidean_distance(datapoint1, datapoint2):
    """
    This function takes two data points
    :param datapoint1: first Data point
    :param datapoint2: second Data point
    :return: Euclidean Distance
    """
    # using linear algebra norm function to find out the euclidean distance
    eucl_distance = np.sqrt(np.sum(np.square(datapoint1-datapoint2)))
    return eucl_distance


def gauss_normalize(merged_dataset):
    """
    This function is used to normalize the DataSet
    :param merged_dataset: Given DataSet
    :return: Gausssian Normalize DataSet
    """
    # creating a new Dataset by copy command
    dataset_normalized = merged_dataset.copy()
    # using for loop to find out the mean and median of the DataSet
    for i in range(len(dataset_normalized.T)):
        feature_mean = dataset_normalized[:, i].mean(axis=0)
        feature_std = dataset_normalized[:, i].std(axis=0)

        # Gaussian normalisation of features
        dataset_normalized[:, i] = (dataset_normalized[:, i] - feature_mean) / feature_std

    return dataset_normalized


# Using random initialization for selecting the first k centroids
def rand_init(k, merged_dataset):
    """
    Randomly selects k object to act as the initial centroids
    :param k: number of clusters
    :param merged_dataset: DataSet from file
    :return: centroids
    """
    # variable to hold the total number of rows in the DataSet
    obj = len(merged_dataset)
    # Initialize the random number generator
    np.random.seed(45)
    # Randomly selecting the centroids
    cent_ind = np.random.choice(obj, k, replace=False)
    # A list to hold the values of centroids
    cent = []
    for i in cent_ind:
        cent.append(merged_dataset[i])
    # using stack function to stack all the centroids
    cent = np.stack(cent, axis=0)

    return cent


def mean_assign_object(centroid_array, merged_dataset):
    """
    This function assign different objects to different clusters
    :param centroid_array: centroids of the DataSet
    :param merged_dataset: Given DataSet
    :return: distance Matrix
    """
    obj = len(merged_dataset)
    condition = True
    # finding Euclidean distance between centroids and a Data Set point
    for item in centroid_array:
        distance_between_two_mean = []
        for x in range(obj):
            distance_between_two_mean.append(euclidean_distance(item, merged_dataset[x]))

        distance_between_two_mean = np.array(distance_between_two_mean).reshape(len(merged_dataset), 1)

        if condition == True:
            array = distance_between_two_mean
            condition = False

        else:
            array = np.concatenate((array, distance_between_two_mean), axis=1)
    array = np.argmin(array, axis=1)
    return array


def mean_optimizer_function(k, cluster_array, merged_dataset, centroid_array):
    """
    This function optimizes the values of centroids
    :param k: number of clusters
    :param cluster_array: clusters of DataSet
    :param merged_dataset: Given DataSet
    :param centroid_array : centroids of the DataSet
    :return: New optimized clusters
    """
    # Variable list to hold the value optimized centroids
    centroids_data_points = []
    # Iterating through different value of k to find out optimized centroids
    for k in range(k):
        clust_data_points = merged_dataset[np.where(cluster_array == k)]
        for i in range(len(clust_data_points)):
            centroids_data_point = np.mean(clust_data_points, axis=0)

        centroids_data_points.append(np.array(centroids_data_point))
    # using stack function of numpy to create a new list of centroids
    centroids_data_points = np.stack(centroids_data_points, axis=0)

    return centroids_data_points


def mean_objective_function(k, merged_dataset, centroid_array, cluster_array):
    """
    This function is for finding objective function
    :param k: number of clusters
    :param merged_dataset: Given DataSet
    :param centroid_array: centroids of DataSet
    :param clusters: Clusters of DataSet
    :return: Objective function
    """
    objective_function = 0
    # iterating through k values to find the objective function for each k values
    for x in range(k):
        distance_between_two_median = merged_dataset[np.where(cluster_array == x)]
        for item in range(len(distance_between_two_median)):
            objective_function = objective_function + (euclidean_distance(distance_between_two_median[item], centroid_array[x])) ** 2

    return objective_function


def mean(k, merged_dataset, epoch=64, norm=False):
    """
    This function calculates the k_mean
    :param k: Number of clusters
    :param merged_dataset: Given DataSet
    :param epoch: Number of epoch
    :param normalize_data: Whether input data is normalized or not
    :return: centroids, best_objective and  cluster_array
    """
    centroid_array = rand_init(k, merged_dataset)
    # checking whether normalisation is required or not
    if norm == True:
        merged_dataset = gauss_normalize(merged_dataset)
    # floating point representation of (positive) infinity
    obj = np.Infinity
    for item in range(epoch):
        cluster_array = mean_assign_object(centroid_array, merged_dataset)
        x = mean_objective_function(k, merged_dataset, centroid_array, cluster_array)

        if (x < obj):
            obj = x
            centroid_array = mean_optimizer_function(k, cluster_array, merged_dataset, centroid_array)
        else:
            break
    return centroid_array, obj, cluster_array


# ========================================================================================================
# b_cubed_precision , b_cubed_recall and b_cubed_f_score function are common for both K_Mean and K_Median

def b_cubed_recall(metrix, numpy_array, merged_dataset, d2, d1, d4, d3):
    """
    This function calculate the recall
    :param metrix: cluster array
    :param numpy_array: Numpy array of given dataset
    :param merged_dataset: given DataFrame
    :param d2: countries Dataset
    :param d1: animals Dataset
    :param d4: veggies Dataset
    :param d3: fruits Dataset
    :return: B_CUBED Recall
    """
    record_list = [d2, d1, d4, d3]
    # creating a list to store recall value as a list
    recall = []
    for item in range(len(numpy_array)):
        cluster_label = metrix[item]
        obj_indx = merged_dataset.index[item]

        # Checking which class the object belongs to
        if obj_indx in list(d2.index):
            label = 0
        if obj_indx in list(d1.index):
            label = 1
        if obj_indx in list(d4.index):
            label = 2
        if obj_indx in list(d3.index):
            label = 3
        # defined a variable x to store the count
        x = 0
        for pointer in list(merged_dataset.iloc[np.where(metrix == cluster_label)].index):
            if pointer in list(record_list[label].index):
                x += 1

        recall.append(x / len(list(record_list[label].index)))
    # summing up all the recall and dividing it by total number of objects
    b_cubed_recall = np.sum(recall) / len(merged_dataset)

    return b_cubed_recall, recall


def b_cubed_precision(metrix, numpy_array, merged_dataset, d2, d1, d4, d3):
    """
    This function calculates the B-CUBED Precision
    :param metrix: cluster array
    :param numpy_array: Numpy array of given dataset
    :param merged_dataset: given DataFrame
    :param d2: countries Dataset
    :param d1: animals Dataset
    :param d4: veggies Dataset
    :param d3: fruits Dataset
    :return: B-CUBED Precision
    """
    # Creating a list
    record_list = [d2, d1, d4, d3]
    # creating a list to store precision value as a list
    precision = []
    for item in range(len(numpy_array)):
        cluster_label = metrix[item]
        obj_indx = merged_dataset.index[item]

        # Checking which class the object belongs to
        if obj_indx in list(d2.index):
            label = 0
        if obj_indx in list(d1.index):
            label = 1
        if obj_indx in list(d4.index):
            label = 2
        if obj_indx in list(d3.index):
            label = 3
        # defined a variable x to store the count
        x = 0
        # calculating the precision of each object
        for pointer in list(merged_dataset.iloc[np.where(metrix == cluster_label)].index):
            if pointer in list(record_list[label].index):
                x += 1

        precision.append(x / len(list(merged_dataset.iloc[np.where(metrix == cluster_label)].index)))
    # calculating overall precision
    b_cubed_precision = np.sum(precision) / len(merged_dataset)

    return b_cubed_precision, precision


def b_cubed_f_score(merged_dataset, b_cubed_precision, b_cubed_recall):
    """
    This function calculates b_cubed f_score based b_cubed precision list and b_cubed recall list
    :param merged_dataset: given DataSet
    :param b_cubed_precision: B_CUBED precision list
    :param b_cubed_recall: B_cubed recall list
    :return: B_CUBED f_score
    """
    # creating a list to store f_score value as a list
    f_score = []
    # iterating through the b_cubed_precision list
    for item in range(len(b_cubed_precision)):
        f_score.append((2 * b_cubed_precision[item] * b_cubed_recall[item]) / (b_cubed_precision[item] + b_cubed_recall[item]))
    # summing up all the f_score and dividing it by total number of objects
    b_cubed_f_score = np.sum(f_score) / len(merged_dataset)

    return b_cubed_f_score


# ==================================================================================================

# Q3
"""
Run the k-means clustering algorithm you implemented in part (1) to cluster the given instances. Vary the value of k 
from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of clusters. Plot k in the horizontal 
axis and the B-CUBED precision, recall and F-score in the vertical axis in the same plot.
"""
K = list(np.arange(1, 10))
precision_list = []
recall_list = []
f_score_list = []
print('\nB-CUBED Precision, Recall, F-Score for K-means using un-normalized data for k = 1 to 9')
for k in range(1, 10):
    kmean, obj, matrix = mean(k, numpy_array, norm=False)
    # calculating the b_cubed recall and creating list for the plot
    recall_value, recall = b_cubed_recall(matrix, numpy_array, merged_dataset, d2, d1, d4, d3)
    recall_list.append(recall_value)
    # calculating the b_cubed precision and creating list for the plot
    precision_value, precision = b_cubed_precision(matrix, numpy_array, merged_dataset, d2, d1, d4, d3)
    precision_list.append(precision_value)
    # calculating the b_cubed f_score and creating list for the plot
    b_cubed_precision_list = precision
    b_cubed_recall_list = recall
    f_score = b_cubed_f_score(merged_dataset, b_cubed_precision_list, b_cubed_recall_list)
    f_score_list.append(f_score)

    # Printing the B_ CUBED Metrics  for the corresponding k values
    print('\nK-value = ', k)
    print(f'Precision = {precision_value}')
    print(f'Recall = {recall_value}')
    print(f'F-Score = {f_score}')

# Plotting the B-CUBED metrics vs the k values
plt.plot(K, precision_list, c='b')
plt.plot(K, recall_list, c='g')
plt.plot(K, f_score_list, c='r')
plt.legend(labels=['Precision', 'Recall', 'F-score'])
plt.xlabel('K-Values')
plt.ylabel('B-Cubed Metrics')
plt.title('K-Means algorithm for un-normalized data')
plt.show()

# Q4
"""
Now re-run the k-means clustering algorithm you implemented in part (1) but normalise each object (vector) to unit l2 
length before clustering. Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for 
each set of clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in the vertical 
axis in the same plot.
"""

K = list(np.arange(1, 10))
precision_list = []
recall_list = []
f_score_list = []
print('\nB-CUBED Precision, Recall, F-Score for K-means using Normalized data for k = 1 to 9')
for k in range(1, 10):
    kmean, obj, matrix = mean(k, numpy_array, norm=True)
    # calculating the b_cubed recall and creating list for the plot
    recall_value, recall = b_cubed_recall(matrix, numpy_array, merged_dataset, d2, d1, d4, d3)
    recall_list.append(recall_value)
    # calculating the b_cubed precision and creating list for the plot
    precision_value, precision = b_cubed_precision(matrix, numpy_array, merged_dataset, d2, d1, d4, d3)
    precision_list.append(precision_value)
    # calculating the b_cubed f_score and creating list for the plot
    b_cubed_precision_list = precision
    b_cubed_recall_list = recall
    f_score = b_cubed_f_score(merged_dataset, b_cubed_precision_list, b_cubed_recall_list)
    f_score_list.append(f_score)

    # Printing the B_ CUBED Metrics  for the corresponding k values
    print('\nK-value = ', k)
    print(f'Precision = {precision_value}')
    print(f'Recall = {recall_value}')
    print(f'F-Score = {f_score}')

# Plotting the B-CUBED Precision, Recall, F-score vs the k values
plt.plot(K, precision_list, c='b')
plt.plot(K, recall_list, c='g')
plt.plot(K, f_score_list, c='r')
plt.legend(labels=['Precision', 'Recall', 'F-score'])
plt.xlabel('K-Values')
plt.ylabel('B-Cubed Metrics')
plt.title('K-Means algorithm for Normalized data')
plt.show()

# Q2
"""
Implement the k-medians clustering algorithm to cluster the instances into k clusters.
"""


def manhattan_distance(datapoint1, datapoint2):
    """
    This function calculates the Manhattan distance
    :param datapoint1: first Data point
    :param datapoint2: second Data point
    :return: Manhattan distance
    """
    man_distance = 0
    for d1, d2 in zip(datapoint1, datapoint2):
        difference = d2 - d1
        absolute_difference = abs(difference)
        man_distance += absolute_difference

    return man_distance


def medians_assign_object(median, merged_dataset):
    """
    This function is used to assign the median
    :param median: median
    :param merged_dataset: given DataSet
    :return: distance_matrix
    """
    number_of_object = len(merged_dataset)
    condition = True

    for item in median:
        distance_between_two_median = []
        for x in range(number_of_object):
            distance_between_two_median.append(manhattan_distance(item, merged_dataset[x]))

        distance_between_two_median = np.array(distance_between_two_median).reshape(len(merged_dataset), 1)

        if condition == True:
            array = distance_between_two_median
            condition = False

        else:
            array = np.concatenate((array, distance_between_two_median), axis=1)
    # Returns the indices of the minimum values along axis =1 and storing in a variable
    array =  np.argmin(array, axis=1)
    return array


def optimizer_function(k, cluster_array, merged_dataset, medians):
    """
    This function optimizes the median values for all objects
    :param k: number of clusters
    :param clusters: cluster
    :param merged_dataset: Given DataSet
    :param medians: Median
    :return: optimized median value
    """
    centroids_data_points = []
    for value in range(k):
        data_points_in_clusters = merged_dataset[np.where(cluster_array == value)]
        for i in range(len(data_points_in_clusters)):
            data_points_in_cluster = np.median(data_points_in_clusters, axis=0)

        centroids_data_points.append(np.array(data_points_in_cluster))
    centroids_data_points = np.stack(centroids_data_points, axis=0)

    return centroids_data_points


def obj_funct_median(k, merged_dataset, m_matrix, c_matrix):
    """
    This is function is used to calculate the objective function
    :param k: number of clusters
    :param merged_dataset: Given DataSet
    :param m_matrix: Median
    :param c_matrix: clusters
    :return: objective function
    """
    obj = 0
    for k in range(k):
        cluster_data_points = merged_dataset[np.where(c_matrix == k)]
        for i in range(len(cluster_data_points)):
            obj = obj + (manhattan_distance(cluster_data_points[i], m_matrix[k]))

    return obj


def median(k, merged_dataset, epoch=64, norm=False):
    """
    This function calculates the K_median
    :param k: number of clusters
    :param merged_dataset: Given DataSet
    :param epoch: Number of epoch
    :param norm: Whether Data is normalized or not
    :return:
    """
    centroid_array = rand_init(k, merged_dataset)
    # checking whether normalisation is required or not
    if norm == True:
        merged_dataset = gauss_normalize(merged_dataset)
    # floating point representation of (positive) infinity
    obj = np.Infinity
    for iter in range(epoch):
        cluster_array = medians_assign_object(centroid_array, merged_dataset)
        initial_score = obj_funct_median(k, merged_dataset, centroid_array, cluster_array)
        if (initial_score < obj):
            obj = initial_score
            centroid_array = optimizer_function(k, cluster_array, merged_dataset, centroid_array)
        else:
            break
    return centroid_array, obj, cluster_array


# Q5
"""
Run the k-medians clustering algorithm you implemented in part (2) over the unnormalised objects. Vary the value of k 
from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of clusters. Plot k in the horizontal 
axis and the B-CUBED precision, recall and F-score in the vertical axis in the same plot.

"""

K = list(np.arange(1, 10))
precision_list = []
recall_list = []
f_score_list = []
print('\nB-CUBED Precision, Recall, F-Score for K-medians using un-normalized data for k = 1 to 9')
for i in range(1, 10):
    medians, obj, array = median(i, numpy_array, norm=False)
    # calculating the b_cubed recall and creating list for the plot
    recall_value, recall = b_cubed_recall(array, numpy_array, merged_dataset, d2, d1, d4, d3)
    recall_list.append(recall_value)
    # calculating the b_cubed precision and creating list for the plot
    precision_value, precision = b_cubed_precision(array, numpy_array, merged_dataset, d2, d2, d4, d3)
    precision_list.append(precision_value)
    # calculating the b_cubed f_score and creating list for the plot
    b_cubed_precision_list = precision
    b_cubed_recall_list = recall
    f_score = b_cubed_f_score(merged_dataset, b_cubed_precision_list, b_cubed_recall_list)
    f_score_list.append(f_score)

    # Printing the B_ CUBED Metrics  for the corresponding k values
    print('\nK-value = ', i)
    print(f'Precision = {precision_value}')
    print(f'Recall = {recall_value}')
    print(f'F-Score = {f_score}')

plt.plot(K, precision_list, c='b')
plt.plot(K, recall_list, c='g')
plt.plot(K, f_score_list, c='r')
plt.legend(labels=['Precision', 'Recall', 'F-score'])
plt.xlabel('K-Values')
plt.ylabel('B-Cubed Metrics')
plt.title('K-Medians algorithm using un-normalized data')
plt.show()

# Q6
"""
Now re-run the k-medians clustering algorithm you implemented in part (2) but normalise each object (vector) to unit 
l2 length before clustering. Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for
each set of clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in the vertical 
axis in the same plot.

"""
K = list(np.arange(1, 10))
precision_list = []
recall_list = []
f_score_list = []
print('\nB-CUBED Precision, Recall, F-Score for K-medians using Normalized data for k = 1 to 9')
for x in range(1, 10):
    medians, obj, array = median(x, numpy_array, norm=True)
    # calculating the b_cubed precision and creating list for the plot
    recall_value, recall = b_cubed_recall(array, numpy_array, merged_dataset, d2, d1, d4, d3)
    recall_list.append(recall_value)
    # calculating the b_cubed recall and creating list for the plot
    precision_value, precision = b_cubed_precision(array, numpy_array, merged_dataset, d2, d1, d4, d3)
    precision_list.append(precision_value)
    # calculating the b_cubed f_score and creating list for the plot
    b_cubed_precision_list = precision
    b_cubed_recall_list = recall
    f_score = b_cubed_f_score(merged_dataset, b_cubed_precision_list, b_cubed_recall_list)
    f_score_list.append(f_score)

    # Printing the B_ CUBED Metrics  for the corresponding k values
    print('\nK-value = ', x)
    print(f'Precision = {precision_value}')
    print(f'Recall = {recall_value}')
    print(f'F-Score = {f_score}')

plt.plot(K, precision_list, c='b')
plt.plot(K, recall_list, c='g')
plt.plot(K, f_score_list, c='r')
plt.legend(labels=['Precision', 'Recall', 'F-score'])
plt.xlabel('K-Values')
plt.ylabel('B-Cubed Metrics')
plt.title('K-Medians algorithm using Normalized data')
plt.show()
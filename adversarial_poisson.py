# adversarial_poisson.py
#
# This script calculates the accuracy of a perturbed dataset using Poisson Learning.
# First, we calculate the unpertubed accuracy. Then, we run trials pitting the adversarially
# perturbed weight matrix against a randomly pertubed weight matrix, and compare their
# accuracies for differing values of epsilon. There epsilon values control to what degree the 
# perturbation matrix affects the weight matrix.
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from scipy import sparse


# Calculate and return the adversarial perturbation matrix V, Given the weight matrix W and initial solution u.
def run_grad(W, u, num_classes, iterations):
    # create a ones matrix B, where B_ij = 1 iff W_ij>0.
    B = 1 * (W > 0)
    V = B.copy()
    # normalize V
    normV = np.sqrt(np.sum(V.multiply(V)))
    V = V / (normV)
    dt = 0.01
    gradu = []

    for _ in range(iterations):
        for i in range(num_classes):
            gradu.append(gl.graph_gradient(B, u[:, i]))

        sum_gradu_squared = gradu[0].multiply(gradu[0])
        for i in range(1, num_classes):
            sum_gradu_squared = sum_gradu_squared + gradu[i].multiply(gradu[i])

        gradV = V.multiply(sum_gradu_squared)
        V = V + dt * gradV

        # project V to constraint ||V|| < 1
        normV = np.sqrt(np.sum(V.multiply(V)))
        V = V / (normV)
    return V


# return a randomly perturbed matrix.
def random_perturb(W, dimension):
    I, J, V = sparse.find(W)
    V = sparse.coo_matrix(
        (np.random.rand(len(I)), (I, J)), shape=(dimension, dimension)
    ).tocsr()
    # symmetrize
    V = (V + V.T) / 2
    # project V to constraint ||V||<1
    normV = np.sqrt(np.sum(V.multiply(V)))
    V = V / (normV)
    return sparse.csr_matrix(V)


# Synthetic Moons

n = 500
X, L = datasets.make_moons(n_samples=n, noise=0.1)
classes = 2
k = 10
W = gl.knn_weight_matrix(k, data=X)

# MNist Data set

# L = gl.load_labels("mnist")
# classes = len(set(L))
# I, J, D = gl.load_kNN_data("mnist", metric="vae")
# W = gl.weight_matrix(I, J, D, 10)
# n = W.get_shape()[0]

# Randomly choose labels
m = 5  # 5 labels per class
ind = gl.randomize_labels(L, m)  # indices of labeled points

# Semi-supervised learning
# Returns kxn matrix u, can use default conjgrad solver now
u, T = gl.poisson(W, ind, L[ind])
u = u.T
l = np.argmax(u, axis=1)

# Compute accuracy
unpertrubed_acc = gl.accuracy(l, L, m)
print("Accuracy=%f" % unpertrubed_acc)


# compute trials of adversrial vs random perturbation
adversary_acc = []
random_acc = []
epsilon_vals = []
unperturbed_vals = []
V_adversary = run_grad(W, u, classes, 10)
epsilon = 1
for i in range(7):
    # calculate accuracy from gradient acsent
    W_perturb = W + epsilon * V_adversary
    u_perturb, T = gl.poisson(W_perturb, ind, L[ind])
    u_perturb = u_perturb.T
    l = np.argmax(u_perturb, axis=1)
    adversary_acc.append(gl.accuracy(l, L, m))
    for _ in range(5):
        # calculate accuracy from random pertubation
        sub_random_acc = []
        V_random = random_perturb(W, n)
        W_perturb = W + epsilon * V_random
        u_perturb, T = gl.poisson(W_perturb, ind, L[ind])
        u_perturb = u_perturb.T
        l = np.argmax(u_perturb, axis=1)
        sub_random_acc.append(gl.accuracy(l, L, m))

    random_acc.append(np.average(sub_random_acc)) #average the accuracies of the random perturbations for current epsilon
    print("Epsilon equals:%.3f" % epsilon)
    print("Accuracy from adversarial perturbation:%f" % adversary_acc[i])
    print("Average accuracy from random perturbation:%f" % random_acc[i])
    epsilon_vals.append(epsilon)
    unperturbed_vals.append(unpertrubed_acc)
    epsilon *= 10


# plot results
plt.scatter(epsilon_vals, adversary_acc, label="Adversarial Perturbation")
plt.scatter(epsilon_vals, random_acc, label="Random Perturbation")
plt.scatter(epsilon_vals, unperturbed_vals, label="Unperturbed")
plt.plot(epsilon_vals, adversary_acc)
plt.plot(epsilon_vals, random_acc)
plt.plot(epsilon_vals, unperturbed_vals)
# plt.plot(unpertrubed_acc, epsilon_vals, label = "Unperturbed")
plt.xscale("log")
plt.xlabel("Epsilon")
plt.ylabel("% Accuracy")
plt.legend()
plt.show()

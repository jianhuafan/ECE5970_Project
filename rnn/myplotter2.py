#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plots results for each epoch of BinaryDenseNet.

Author: Gustavo Angarita
"""

import os
import sys
import re
import xlsxwriter
import numpy as np
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pylab import *


def check_for_validation(content):
    return True
    '''
    curr_line = 0
    while curr_line < len(content):
        first_line = content[curr_line].split()
        if first_line and first_line[0] == 'Epoch':
            third_line_below = content[curr_line + 3].split()
            if third_line_below[0] == 'Epoch':
                return False
            return True
        curr_line += 1
    '''


def get_values_as_array(content):
    train_loss =[]
    curr_line = 0
    while curr_line < len(content):
        words = content[curr_line].split()
        if words and words[0] == 'epoch':
            losslist = []
            accuracylist = []
            curr_line += 1
            words = content[curr_line].split()
            while words and words[0] == 'loss':
                loss = [float(words[-1])]
                losslist += loss
                if curr_line < len(content) - 1:
                    curr_line += 1
                    words = content[curr_line].split()
            train_loss.append(losslist)
            curr_line -= 1
        curr_line += 1
    return (train_loss)


def make_figure(mode, file_name1, file_name2, epochs1, epochs2, name, y1_axis, y2_axis):
    x1_axis = arange(1, epochs1 + 1, float(epochs1)/len(y1_axis))
    x2_axis = arange(1, epochs2 + 1, float(epochs2)/len(y2_axis))
    #x3_axis = arange(1, epochs3 + 1, float(epochs3)/len(y3_axis))
    fig = plt.figure()
    fig.canvas.set_window_title(name)
    plt.title("%s vs. Epoches" % name)
    plt.xlabel("Epoches")
    plt.ylabel(name)
    savename1 = file_name1.split('.')
    savename2 = file_name2.split('.')
    #savename3 = file_name3.split('.')
    plt.plot(x1_axis, y1_axis, label = savename1[0])
    plt.plot(x2_axis, y2_axis, label = savename2[0])
    #plt.plot(x3_axis, y3_axis, label = savename3[0])
    # plt.axhline(y=1.505, color='y', linestyle='-')
    # plt.axhline(y=1.372, color='y', linestyle='-')
    # plt.axhline(y=1.05, color='k', linestyle='-')
    # plt.axhline(y=1.81, color='k', linestyle='-')
    plt.legend(loc='best')
    plt.grid(True)
    plt.draw()
    if mode == 'mean':
        plt.savefig("compare_{}_{}.png".format(name, savename1[0]))
    else:
        plt.savefig("{}_{}.png".format(name, savename1[0]))

def get_special_values_as_array(content):
    train_loss =[]
    train_accuracy = []
    curr_line = 0
    while curr_line < len(content):
        words = content[curr_line].split()
        if len(words) > 2 and words[1] == 'loss:':
            loss = [float(words[-1])]
            train_loss += [loss]
        if len(words) >2 and words[1] == 'accuracy:':
            accuracy = [float(words[-1])]
            train_accuracy += [accuracy]
        curr_line += 1
    return (train_accuracy, train_loss)

def plot_mean(file_name1, file_name2, content1, content2):
    
    #includes_validation = check_for_validation(content)

    # train_loss, l2_loss, valid_loss, valid_error, train_error, test_error, lowest_valid, learning_rate = get_values_as_array(
    (loss1) = get_values_as_array(content1)
    (loss2) = get_values_as_array(content2)
    # (accuracy3, loss3) = get_special_values_as_array(content3)
    epochs1 = len(loss1)
    epochs2 = len(loss2)
    print(epochs1)
    print(epochs2)
    # epochs3 = len(accuracy3)

    mean_loss1 = []
    
    total_loss1 = 0
    
    loss1 = np.array(loss1)
    
    min_loss = 10

    
    for i in range(epochs1):
        temp1 = np.mean(loss1[i])
        mean_loss1 += [temp1]

    mean_loss2 = []
    
    total_loss2 = 0
    
    loss2 = np.array(loss2)
    
    min_loss = 10
    for i in range(epochs2):
        temp2 = np.mean(loss2[i])
        mean_loss2 += [temp2]

    #make_figure("accuracy", accuracy)
    make_figure('mean', file_name1, file_name2, epochs1, epochs2, "Mean_Training_loss", mean_loss1, mean_loss2)
    #make_figure('mean', file_name1, file_name2, epochs1, epochs2, "Mean_Training_accuracy", mean_accuracy1, mean_accuracy2)
    plt.show(block=False)
    raw_input("press enter to close plots")
    plt.close("all")

def plot_interval(file_name, content):
    #includes_validation = check_for_validation(content)

    # train_loss, l2_loss, valid_loss, valid_error, train_error, test_error, lowest_valid, learning_rate = get_values_as_array(
    (accuracy, loss) = get_values_as_array(content)
    x_loss = []
    y_loss = []
    interval = 10000
    epochs = len(loss)
    for i in range(len(loss)):
        j = 0
        while j < len(loss[i]):
            y_loss += [loss[i][j]]
            j += interval
    x_accuracy = []
    y_accuracy = []
    interval =10000
    epochs = len(accuracy)
    for i in range(len(accuracy)):
        j = 0
        while j < len(accuracy[i]):
            y_accuracy += [accuracy[i][j]]
            j += interval
    #make_figure("accuracy", accuracy)
    make_figure('interval', interval, file_name, epochs,"Training_loss", y_loss)
    make_figure('interval', interval, file_name, epochs,"Training_accuracy", y_accuracy)
    plt.show(block=False)
    #raw_input("press enter to close plots")
    plt.close("all")

def plot_last(file_name, content):
    last = 100
    #includes_validation = check_for_validation(content)

    # train_loss, l2_loss, valid_loss, valid_error, train_error, test_error, lowest_valid, learning_rate = get_values_as_array(
    (accuracy, loss) = get_values_as_array(content)
    y_loss = []
    y_accuracy = []
    epochs = len(loss)
    for i in range(len(loss)):
        for j in range(len(loss[i])- last, len(loss[i])-last + 1):
            y_loss += [loss[i][j]]
    for i in range(len(accuracy)):
        for j in range(len(accuracy[i])- last, len(accuracy[i])-last + 1):
            y_accuracy += [accuracy[i][j]]

    #make_figure("accuracy", accuracy)
    interval = last
    make_figure('last', interval, file_name, epochs,"Training_loss", y_loss)
    make_figure('last', interval, file_name, epochs,"Training_accuracy", y_accuracy)
    plt.show(block=False)
    #raw_input("press enter to close plots")
    plt.close("all")

def plot_mode():
    
    file_name1 = sys.argv[1]
    file_name2 = sys.argv[2]
    mode = sys.argv[3]
    with open(file_name1) as f1:
        content1 = f1.readlines()
    with open(file_name2) as f2:
        content2 = f2.readlines()
    if mode == 'mean':
        plot_mean(file_name1, file_name2, content1, content2)
    if mode == 'interval':
        plot_interval(file_name, content)
    if mode == 'last':
        plot_last(file_name, content)


    
if __name__ == "__main__":
    plot_mode()

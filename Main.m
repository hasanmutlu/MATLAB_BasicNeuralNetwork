%Author: N17232403 - Hasan MUTLU
%In this program I implement a Neural Network.
%
%Attention!!:This application was developed with matlab 2016b version!
%please run this application with matlab 2016b or higher version.
%
%NeuralNetwork.m -> This file contains main class of Neural Network. We
%give layers of network to this class and this class handle all jobs.
%
%NeuralNetworkLayer.m -> This file contains Layers class definition. This 
%class handles Feedforward, backpropagation process for own neurons and
%also handles adding or removing bias terms to weight matrices when
%feed forwad or backpropagation are in progress.
%
%For optimization options, you can choose one of them:
%
% 'method1' - Only gradient descent. No optimizations wil bee applied.
% 'method2' - Only momentum optimization.
% 'method3' - Only adaptive learning rate.
% 'method4' - Momentum and Adaptive learning rate at the same time.
% 'method5' - Adam optimization.
clear all;
dbstop if error;
data = importdata('train_data.csv',';');
data(:,1:17) = (data(:,1:17) - mean(data(:,1:17))) ./ std(data(:,1:17));%normalize data
data = data(randperm(size(data,1)),:);%shuffle data
data = data(1:end/2,:);% to prevent overfitting, We use half of data
dataCount = size(data,1);
trainPercent = 0.7;%use %70 percent of data to train network
testPercent = 0.3;%use %30 percent of data to test and validate network
trainCount = dataCount * trainPercent;
trainData = data(1:trainCount,:);
testData = data(trainCount+1:end,:);
Layers = [NeuralNetworkLayer(17,'sigmoid'),NeuralNetworkLayer(5,'softmax')];
NNetwork = NeuralNetwork(17,Layers,'method5');%use adam optimization.
NNetwork.Shuffle = true;%enabled shuffling data for each epoch
learningRate = 0.001;
epochCount = 1000;
batchSize = 1000;%How many input will be trained same time
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);%Train network with given data set
plot(NNetwork.GraphData(:,1),NNetwork.GraphData(:,2));%draw Epoch/Loss graph
title('Loss Result');
xlabel('Epoch Count');
ylabel('Loss');
accuracy = NNetwork.EvaluateAccuracy(testData);%get accuracy of network
sprintf('Accuracy %%%f',accuracy)
confusionMatrix = NNetwork.GetConfusionMatrix(testData)%get confusion matrix of prediction








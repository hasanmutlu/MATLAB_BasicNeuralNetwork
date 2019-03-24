%Author: N17232403 - Hasan MUTLU
%this file used to draw graphs of all optimization methods together
hold off;
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
rng(10);% added to generate same random numbers When Layers initialize
NNetwork = NeuralNetwork(17,Layers,'method1');
NNetwork.Shuffle = true;
learningRate = 0.001;
epochCount = 1000;
batchSize = 1000;%How many input will be trained same time
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);
method1Graph = NNetwork.GraphData;
Accuracy(1,1)=NNetwork.EvaluateAccuracy(testData);
sprintf('Accuracy Method1 %%%f',NNetwork.EvaluateAccuracy(testData))
rng(10);
NNetwork = NeuralNetwork(17,Layers,'method2');
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);
method2Graph = NNetwork.GraphData;
Accuracy(2,1)=NNetwork.EvaluateAccuracy(testData);
sprintf('Accuracy Method2 %%%f',NNetwork.EvaluateAccuracy(testData))
rng(10);
NNetwork = NeuralNetwork(17,Layers,'method3');
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);
method3Graph = NNetwork.GraphData;
Accuracy(3,1)=NNetwork.EvaluateAccuracy(testData);
sprintf('Accuracy Method3 %%%f',NNetwork.EvaluateAccuracy(testData))
rng(10);
NNetwork = NeuralNetwork(17,Layers,'method4');
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);
method4Graph = NNetwork.GraphData;
Accuracy(4,1)=NNetwork.EvaluateAccuracy(testData);
sprintf('Accuracy Method4 %%%f',NNetwork.EvaluateAccuracy(testData))
rng(10);
NNetwork = NeuralNetwork(17,Layers,'method5');
NNetwork.TrainSet(trainData,learningRate,epochCount,batchSize);
Accuracy(5,1)=NNetwork.EvaluateAccuracy(testData);
sprintf('Accuracy Method5 %%%f',NNetwork.EvaluateAccuracy(testData))
method5Graph = NNetwork.GraphData;
title('Loss Result');
xlabel('Epoch Count');
ylabel('Loss');
plot(method1Graph(:,1),method1Graph(:,2));
hold on;
plot(method2Graph(:,1),method2Graph(:,2));
plot(method3Graph(:,1),method3Graph(:,2));
plot(method4Graph(:,1),method4Graph(:,2));
plot(method5Graph(:,1),method5Graph(:,2));
legend({'Without Optimization','Momentum','Adaptive Learning Rate','Momentum and Adaptive Learning Rate','Adam'},'Location','northeast')
hold off;
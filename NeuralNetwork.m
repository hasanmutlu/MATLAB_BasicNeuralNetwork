%Author: N17232403 - Hasan MUTLU
classdef NeuralNetwork < handle
    %add learning rate to train method
    properties
        Layers;
        BiasValue = -1;
        Loss = 'crossEntropy'; % mse (Mean Square Error), crossEntropy 
        LearningRate = 0.001;
        PreviousLoss = 0;
        Optimization = 'method5'   % method1 (only gradient descent) 
                                   % method2 (only momentum) 
                                   % method3 (only adaptive learning rate )
                                   % method4 (momentum and adaptive learning rate)
                                   % method5 (adam)
        Shuffle = false;
        GraphData;%Epoch/Loss graph data
    end
    methods
        function obj = NeuralNetwork(inputCount, layers, optimization)
            obj.Optimization = optimization;
            obj.Layers = layers;
            inputCount = inputCount + 1;%add bias value to all layers when initiazing time
            %initialize weights matrices of all layers with random numbers.
            for i = 1:size(layers,2)
                obj.Layers(i).Optimization = optimization;
                obj.Layers(i).InitializeWeights(inputCount);
                inputCount = obj.Layers(i).NeuronCount + 1;
            end
        end
        function FeedForward(self,inputData)
            %apply feed forward to all layers respectively.
            for i = 1:size(self.Layers,2)
                inputData = [inputData repmat(self.BiasValue, size(inputData,1),1)];
                self.Layers(i).FeedForward(inputData);
                inputData = self.Layers(i).Output;
            end
        end
        function r = GetOutput(self)
            %return last layer output as output result.
            r = self.Layers(size(self.Layers,2)).Output;
        end
        function BackPropagation(self, error , learningRate, iteration)
            %apply back propagation to all layers respectively.
            for i = size(self.Layers,2):-1:1
                isFirstLayer = (i == 1);
                error = self.Layers(i).BackPropagate(error,learningRate,iteration,isFirstLayer);
            end
        end
        function e = GetError(self, desired)
            e = desired - self.GetOutput();
        end
        function r = GetLoss(self,desired)
            %calculates cross entropy loss
            output = self.GetOutput();
            r = -sum(sum(desired.*log(output),2),1)/size(desired,1);
        end
        function Train(self, inputData,desiredResult, iteration)
            self.FeedForward(inputData);
            error = self.GetError(desiredResult);
            self.BackPropagation(-error,self.LearningRate, iteration);
            currentLoss = self.GetLoss(desiredResult);
            %If there is previous loss and adaptive learning rate is
            %enabled, Apply adaptive learning rate
            if strcmp('method3',self.Optimization) || strcmp('method4',self.Optimization)
                if iteration > 1
                    if currentLoss < self.PreviousLoss
                        self.LearningRate = self.LearningRate + 0.001;
                    elseif currentLoss > self.PreviousLoss
                        self.LearningRate = self.LearningRate * 0.8;
                    end
                end
            end
            self.PreviousLoss = currentLoss;          
        end
        function TrainSet(self,trainSet,learningRate,epochCount,batchCount)
            self.LearningRate = learningRate;
            trainCount = size(trainSet,1);
            self.GraphData = zeros(epochCount,2);
            for e = 1:epochCount
                if self.Shuffle == true
                    trainSet = trainSet(randperm(size(trainSet,1)),:);
                end
                epochError = 0;
                epochLoss = 0;
                for i = 1:batchCount:trainCount
                    lastIndex = i + batchCount -1;
                    if lastIndex > trainCount
                        lastIndex = trainCount;
                    end
                    trainData = trainSet(i:lastIndex,1:17);
                    desiredResult = trainSet(i:lastIndex,18:22);
                    self.Train(trainData,desiredResult,i);
                    loss = self.GetLoss(desiredResult);
                    sampleError = self.GetError(desiredResult); 
                    sampleError = sum(sampleError(:))/size(sampleError,1);
                    epochError = epochError + sampleError;
                    epochLoss = epochLoss + loss;
                end
                sprintf('Training...%s Epoch %d Progress: %%%.0f  Loss:%f',self.Optimization,e,(e/epochCount)*100,epochLoss/trainCount)
                self.GraphData(e,2) = epochLoss/trainCount;
                self.GraphData(e,1) = e;
            end
        end
        function r = Predict(self, inputData)
            self.FeedForward(inputData);
            output = self.GetOutput(); 
            %Set maximum value to 1 and set the others 0
            r = (output'>=max(output'))';
        end
        function TestSet(self, testSet)
            testCount = size(testSet,1);
            for i = 1:testCount
                testData = testSet(i,1:17);
                desiredData = testSet(i,18:end);
                output = self.Predict(testData);
                sprintf('Output %f , %f , %f , %f , %f\n',output)
                sprintf('Desired %f , %f , %f , %f , %f\n',desiredData)
            end  
        end
        function r = EvaluateAccuracy(self,testData)
            testDesiredData = testData(:,18:end);
            testData = testData(:,1:17);
            output = self.Predict(testData);
            result = (output - testDesiredData).^2;
            result = sum(result,2);
            trueCount = sum(result == 0);
            r = (trueCount/size(testData,1))*100;
        end
        function r = GetConfusionMatrix(self,testData)
            testDesiredData = testData(:,18:end);
            testData = testData(:,1:17);
            output = self.Predict(testData);
            featureCount = size(output,2);
            r = zeros(featureCount,featureCount);
            for i = 1:featureCount
                for j = 1:featureCount
                r(i,j)=sum(output(:,i) .* testDesiredData(:,j)>0);
                end
            end
            r = (r./sum(r)).*100;
        end
    end
end
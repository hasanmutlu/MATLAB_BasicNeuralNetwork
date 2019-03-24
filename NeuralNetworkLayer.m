%Author: N17232403 - Hasan MUTLU
classdef NeuralNetworkLayer < handle
    properties
        NeuronCount;
        Output;
        Weights;
        Input;
        Sum;
        Beta1 = 0.9;
        Beta2 = 0.999;
        Epsilon = 1e-8;
        Moment = 0;
        Moment2 = 0;
        PreviousDelta;
        ActivationFunction = 'sigmoid'; % sigmoid , tanh ,softmax, for ohtherwise behaves adaline 
        Optimization = 'method5';   % method1 (only gradient descent) 
                                   % method2 (only momentum) 
                                   % method3 (only adaptive learning rate )
                                   % method4 (momentum and adaptive learning rate)
                                   % method5 (adam)
    end
    methods (Access = private)
        function activate(self)
            if strcmp('sigmoid',self.ActivationFunction)
                self.Output = 1./(1+exp(-self.Sum));
            elseif strcmp('tanh',self.ActivationFunction)
                self.Output = tanh(self.Sum);
            elseif strcmp('softmax',self.ActivationFunction)
                self.Output = exp(self.Sum)./sum(exp(self.Sum),2);
            else
                self.Output = self.Sum;
            end
        end
        function r = derivative(self)
            if strcmp('sigmoid',self.ActivationFunction)
                r = self.Output .* (1 - self.Output);
            elseif strcmp('tanh',self.ActivationFunction)
                r = 1 - (self.Output .^2);
            elseif strcmp('softmax',self.ActivationFunction)
                r = 1; % when crossentropy used with softmax. derivative of softmax is equal to 1
            else
                r = 1;
            end      
        end
    end
    methods
        function obj = NeuralNetworkLayer(neuronCount, activationFunction)
            obj.NeuronCount = neuronCount;
            obj.ActivationFunction = activationFunction;
        end
        function InitializeWeights(obj,inputCount)
            obj.Weights = rand(inputCount,obj.NeuronCount);
        end
        function FeedForward(obj, input)
            obj.Input = input;
            obj.Sum = input * obj.Weights;
            obj.activate();
        end
        function result = BackPropagate(self,error,learningRate,iteration,firstLayer)
            deltaValue = error .* self.derivative();
            deltaValue = deltaValue ./ size(error,1);
            deltaWeights = transpose(deltaValue) * self.Input;
            deltaWeights = transpose(deltaWeights);
            result = 0;
            if firstLayer == false
                result = deltaValue * transpose(self.Weights);
                s = size(result,2);
                result = result(:,1:(s-1));% removes bias for back propagation
            end
            %if Adam optimization is enabled, apply
            if strcmp('method5',self.Optimization)
                self.Moment = self.Beta1 * self.Moment + ((1-self.Beta1)* deltaWeights);
                self.Moment2 = self.Beta2 * self.Moment2 + ((1-self.Beta2)*(deltaWeights).^2);
                dwCorrected = self.Moment / (1-(self.Beta1^iteration));
                swCorrected = self.Moment2 /(1-(self.Beta2^iteration));
                dwt = -learningRate .* (dwCorrected ./ (sqrt(swCorrected)+self.Epsilon));
            elseif strcmp('method2',self.Optimization) || strcmp('method4',self.Optimization)
                %If momentum optimization is enabled, apply
                if iteration <= 1
                    dwt = -learningRate * deltaWeights;
                else
                    dwt = 0.5*self.PreviousDelta -learningRate * deltaWeights;
                end
            else
                dwt = -learningRate * deltaWeights;
            end
            self.Weights = self.Weights + dwt;
            self.PreviousDelta = dwt;
        end        
    end
end
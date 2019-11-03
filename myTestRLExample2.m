%DDPG ²Î¿¼ https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-ddpg-agent-to-balance-double-integrator-system.html
function myTestRLExample2
clc;
clear ll;
close all;
env = myRLExample2;
validateEnvironment(env)
% InitialObs = reset(env)
% 
% [NextObs,Reward,IsDone,LoggedSignals] = step(env,10*pi/180);
% NextObs
% testDQN(env);
  testDDPG(env)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function testDDPG(env)
obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);
numActions = numel(actInfo);
hid1neuNum = 50;
hid2neuNum = 20;
statePath = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hid1neuNum,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(hid2neuNum,'Name','CriticStateFC2')];

actionPath = [
    imageInputLayer([numActions 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(hid2neuNum ,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(numActions,'Name','CriticOutput')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

figure
plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);

actorNetwork = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hid1neuNum ,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(hid2neuNum,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(numActions,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];


actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'ActorScaling'},actorOpts);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',env.Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e4,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64,'NumStepsToLookAhead',1);
% agentOpts.NoiseOptions.Variance = 5;
% agentOpts.NoiseOptions.Mean = 0;
% agentOpts.NoiseOptions.VarianceDecayRate = 1e-7;
% agentOpts.NoiseOptions.MeanAttractionConstant = 10;
%  agentOpts.NoiseOptions.InitialAction = 0;
 
agent = rlDDPGAgent(actor,critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 200, ...
    'Verbose', false, ...
    'Plots','training-progress','UseParallel',false);

trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "savedAgents";
doTraining =false;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    save('ex2.mat');
else
    % Load pretrained agent for the example.
    load('ex2.mat');
end
plot(env)

simOptions = rlSimulationOptions('MaxSteps',200,'NumSimulations' ,10);
experience = sim(env,agent,simOptions);

totalReward = sum(experience(1).Reward.data)

states = experience(1).Observation.simpleVehicleStates;
data= states.data;
x= data(1,:,:);
y= data(2,:,:);
theta= data(7,:,:);
figure
 plot(x(:),y(:))
figure
 plot(theta(:))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function testDQN(env)

observationInfo = getObservationInfo(env);
numObservations = observationInfo.Dimension(1);
actionInfo = getActionInfo(env);
numActions = numel(actionInfo);

L = 24; % number of neurons
statePath = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    imageInputLayer([numActions 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(L,'Name','fc5')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');


criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4,'UseDevice',"cpu");
% criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4,'UseDevice',"gpu");
critic = rlRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'state'},'Action',{'action'},criticOptions);

agentOptions = rlDQNAgentOptions(...
    'SampleTime',env.Ts,...
    'UseDoubleDQN',true,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.99,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',8);

agent = rlDQNAgent(critic,agentOptions);
T =20;
Ts =0.1;
maxepisodes = 5000;
maxsteps = ceil(T/Ts);
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'UseParallel',true);

trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 100;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "Gradients";

doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOpts);
    save ex1_DQN.mat
else
    % Load pretrained agent for the example.
    load ex1_DQN.mat       
end

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);

totalReward = sum(experience.Reward)

states = experience.Observation.simpleVehicleStates;
data= states.data;
x= data(1,:,:);
y= data(2,:,:);
theta= data(7,:,:);
figure
 plot(x(:),y(:))
figure
 plot(theta(:))
end
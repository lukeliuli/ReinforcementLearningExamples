%DDPG 参考 https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-ddpg-agent-to-balance-double-integrator-system.html
%1升级网络结构简单
%2输入有1个
function myTestRLExample1_simple1
clc;
clear ll;
close all;
env = myRLExample1;
validateEnvironment(env)
% InitialObs = reset(env)
% 
% [NextObs,Reward,IsDone,LoggedSignals] = step(env,10*pi/180);
% NextObs

  testDDPG(env)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function testDDPG(env)
obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);
numActions = actInfo.Dimension(1);


statePath = imageInputLayer([numObservations 1 1],'Normalization','none','Name','state');
actionPath = imageInputLayer([numActions 1 1],'Normalization','none','Name','action');
commonPath = [concatenationLayer(1,2,'Name','concat')
             quadraticLayer('Name','quadratic')
             fullyConnectedLayer(1,'Name','StateValue')];


criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork,'state','concat/in1');
criticNetwork = connectLayers(criticNetwork,'action','concat/in2');

figure
plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);

actorNetwork = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(numActions,'Name','action')];


actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},actorOpts);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',env.Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e4,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',128,'NumStepsToLookAhead',1);

 
agent = rlDDPGAgent(actor,critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 200, ...
    'MaxStepsPerEpisode', 200, ...
    'Verbose', false, ...
    'Plots','training-progress','UseParallel',false);

trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "savedAgents";
doTraining =true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    save('ex1_simple1.mat');
else
    % Load pretrained agent for the example.
    load('ex1_simple1.mat');
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

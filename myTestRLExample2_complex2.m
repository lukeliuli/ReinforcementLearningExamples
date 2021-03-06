%DDPG 参考 https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-ddpg-agent-for-path-following-control.html
%1升级网络结构复杂
%2输入有2个
function myTestRLExample2_complex2
clc;
clear ll;
close all;
env = myRLExample2B;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = 100; % number of neurons
statePath = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','observation')
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
criticNetwork = addLayers(criticNetwork,actionPath);
    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');


% figure
% plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scale = reshape([0.2 2],[1 1 2]);
bias = reshape([0 0],[1 1 2]);
actorNetwork = [
    imageInputLayer([numObservations  1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(numActions,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',scale)];
actorOptions = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',env.Ts,...
    'ExperienceBufferLength',1e4,...
    'MiniBatchSize',32,'NumStepsToLookAhead',32);

 
agent = rlDDPGAgent(actor,critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 20000, ...
    'MaxStepsPerEpisode', 300, ...
    'Verbose', false, ...
    'Plots','training-progress','UseParallel',false,...;
    'ScoreAveragingWindowLength',10,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',100000);
doTraining=true;
if doTraining
    % Train the agent.
%     load('ex2_complex2.mat','agent');
    trainingStats = train(agent,env,trainOpts);
    save('ex2_complex2.mat','agent');
else
    % Load pretrained agent for the example.

    load('ex2_complex2.mat','agent');
end
plot(env)

simOptions = rlSimulationOptions('MaxSteps',300,'NumSimulations' ,1);
experience = sim(env,agent,simOptions);

totalReward = sum(experience(1).Reward.data);

states = experience(1).Observation.simpleVehicleStates;
data= states.data;
x= data(1,:,:);
y= data(3,:,:);

figure
 plot(x(:),y(:))

end

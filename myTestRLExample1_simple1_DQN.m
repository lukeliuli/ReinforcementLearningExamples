%DDPG 参考 https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-ddpg-agent-to-balance-double-integrator-system.html
%1级网络结构简单
%2输入有1个
function myTestRLExample1_simple1_DQN
clc;
clear ll;
close all;
env = myRLExample1_DQN;
validateEnvironment(env)
% InitialObs = reset(env)
% 
% [NextObs,Reward,IsDone,LoggedSignals] = step(env,10*pi/180);
% NextObs

  testDQN1(env)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function testDQN1(env)
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

% figure
% plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1,'UseDevice',"gpu");
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);


agentOptions = rlDQNAgentOptions(...
    'SampleTime',env.Ts,...
    'UseDoubleDQN',true,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',64,'NumStepsToLookAhead',32);

agent = rlDQNAgent(critic,agentOptions);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 20000, ...
    'MaxStepsPerEpisode', 300, ...
    'Verbose', false, ...
    'Plots','training-progress','UseParallel',false,...;
    'ScoreAveragingWindowLength',10,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',100000,'SaveAgentDirectory', pwd + "\simple1DQNAgents");

doTraining =true;
if doTraining
    % Train the agent.
%     load('ex1_simple1_DQN.mat');
    trainingStats = train(agent,env,trainOpts);
    save('ex1_simple1_DQN.mat','agent');
else
    % Load pretrained agent for the example.
    load('ex1_simple1_DQN.mat','agent');
end
plot(env)

simOptions = rlSimulationOptions('MaxSteps',200,'NumSimulations' ,1);
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

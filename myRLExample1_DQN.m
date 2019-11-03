classdef myRLExample1_DQN < rl.env.MATLABEnvironment
    %MYRLEXAMPLE1: Template for defining custom environment in MATLAB.    
   %������̬ģ�ͣ�https://blog.csdn.net/u013914471/article/details/82968608
   %�ο��� https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-agent-to-control-flying-robot.html
    %% Properties (set properties' attributes accordingly)
    properties
        % initial model state variables
        phi0 = 0;%��ʼ�����
        theta0 = 0;%��ʼǰ��ת��
        x0 = -20;%��ʼXλ��
        y0 = -20;%��ʼYλ��
        vel0 =5/3.5;%��ʼ�ٶ�5km/h
        % sample time
        Ts = 0.1;
        vl = 2.7;%���
        % simulation length
        Tf = 20;
        DisplacementThreshold =1;
        AngleThreshold = 20*pi/180;
        h =0;
       counter = 0;
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State = zeros(7,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = myRLExample1_DQN()
            
          
            % Initialize Observation settings
            
            ObservationInfo = rlNumericSpec([7 1]);
            ObservationInfo.Name = 'simple vehicle States';
            ObservationInfo.Description = 'x, dx, y,dy,phi, dphi,theta';
            
            % Initialize Action settings   
            
%           ActionInfo = rlNumericSpec([1 1],'LowerLimit',[-0.4],'UpperLimit',[0.4]);
       
             ActionInfo = rlFiniteSetSpec((-23:23)*pi/180);
            ActionInfo.Name = 'steering';
            % The following line implements built-in functions of RL env
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
  
           
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
             LoggedSignals = [];
             vel = this.vel0;%��ʱ��Ϊ����
             Ts = this.Ts;
%             'x, dx, y,dy,phi, dphi,phi,theta';
            % Get action
            if( this.IsDone==true)
                vel= 0.1;
            end
           this.counter =this.counter+1;;
            Theta = Action;%gred2deg            
            
            % Unpack state vector
            XP = this.State(1);
         
            YP = this.State(3);
          
            PhiP = this.State(5);
          

            PhiDot = tan(Theta)/this.vl*vel;
            Phi = PhiP+Ts*PhiDot;
            
            XDot=cos(Phi)*vel;
            YDot =sin(Phi)*vel;
            
            X=XP+Ts*XDot;
            Y=YP+Ts*YDot;
            
            
            
            % Euler integration
            Observation =[X;XDot;Y;YDot;Phi;PhiDot;Theta];

            % Update system states
            this.State = Observation;
            LoggedSignal.State =  this.State;
            
            % Check terminal condition
            
            IsDone = abs(X) < this.DisplacementThreshold && abs(Y) < this.DisplacementThreshold && abs(Phi) < this.AngleThreshold;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            % Theta (+- .05 rad)
            if this.h ~=0
                 close(this.h )
            end
               this.h =figure(1001) ;
            
           
            Tphi0 = 0;%��ʼ�����
            Ttheta0 = 0;%��ʼǰ��ת��
            Tx0 = this.x0+0*randn;%��ʼXλ��
            Ty0 = this.y0+0*randn;%��ʼYλ��
            Tvel0 =5/3.5;%�㶨5km/h
            Tdx0 = Tvel0;
            Tdy0 =0;
            TdPhi0=0;
            InitialObservation = [Tx0; Tdx0;Ty0;Tdy0;Tphi0;TdPhi0;Ttheta0];
            this.State = InitialObservation;
             
              this.counter =0;
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
%             notifyEnvUpdated(this);
        end
        

    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
       
        % Reward function
        function Reward = getReward(this)
            XP = this.State(1);
            XDotP = this.State(2);
            YP = this.State(3);
            YDotP = this.State(4);
            PhiP = this.State(5);
            PhiDotP = this.State(6);
            ThetaP = this.State(7);
%              r1 =10*((XP^2+YP^2+PhiP^2)<10);
%              r2 = -100*(abs(XP)>100||abs(YP)>100);
%              r3 = -0.01*ThetaP^2-0.02*XP^2-0.02*YP^2-0.02*PhiP^2;
             
               r1 =1000*((XP^2+YP^2)<50);
              r2 = -1000*(abs(XP)>10||abs(YP)>10);
             r3 = -120*XP^2-120*YP^2;
             Reward =r1+ r2+r3;
             
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            
%             this.h = figure;
            % Update the visualization
             envUpdatedCallback(this)
        end
        
  
        function set.AngleThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','AngleThreshold');
            this.AngleThreshold = val;
        end
        function set.DisplacementThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','DisplacementThreshold');
            this.DisplacementThreshold = val;
        end
      
    end
    
    methods (Access = protected) 
        function envUpdatedCallback(this)
            % Set the visualization figure as the current figure
           
          
            
            % Extract the cart position and pole angle
            XP = this.State(1);
            XDotP = this.State(2);
            YP = this.State(3);
            YDotP = this.State(4);
            PhiP = this.State(5);
            PhiDotP = this.State(6);
            ThetaP = this.State(7);
           
            figure(this.h)
            
            hold on;
            subplot(2,1,1)
            plot(0, 0,'bo');
            plot(XP, YP,'rs');
            
            xlim([-5+this.x0 10])
             ylim([-5+this.y0 10])
               hold off
                hold on;
              subplot(2,1,2)
               plot(this.counter,rad2deg(ThetaP),'rs' );
            xlim([0 this.counter+100])
             ylim([-40 50])
              hold off
              
%          figure(this.h2)
%          hold on
%             plot(this.counter,rad2deg(ThetaP),'rs' );
%             xlim([0 this.counter+100])
%              ylim([-40 50])
%             hold off
    
            end
    end
  
end

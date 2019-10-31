classdef myRLExample1 < rl.env.MATLABEnvironment
    %MYRLEXAMPLE1: Template for defining custom environment in MATLAB.    
   %车辆动态模型：https://blog.csdn.net/u013914471/article/details/82968608
   %参考： https://www.mathworks.com/help/releases/R2019b/reinforcement-learning/ug/train-agent-to-control-flying-robot.html
    %% Properties (set properties' attributes accordingly)
    properties
        % initial model state variables
        phi0 = 0;%初始航向角
        theta0 = 0;%初始前轮转角
        x0 = 20;%初始X位置
        y0 = 20;%初始Y位置
        vel0 =5/3.5;%恒定5km/h
        % sample time
        Ts = 0.1;
        vl = 2.7;%轴距
        % simulation length
        Tf = 20;
        DisplacementThreshold =0.2;
        AngleThreshold = 10*pi/180;
       h =[];
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
        function this = myRLExample1()
            
          
            % Initialize Observation settings
            
            ObservationInfo = rlNumericSpec([7 1]);
            ObservationInfo.Name = 'simple vehicle States';
            ObservationInfo.Description = 'x, dx, y,dy,phi, dphi,theta';
            
            % Initialize Action settings   
            
%           ActionInfo = rlNumericSpec([1 1],'LowerLimit',-23*pi/180,'UpperLimit',23*pi/180);
       
           ActionInfo = rlFiniteSetSpec((-23:23)*pi/180);
            
            % The following line implements built-in functions of RL env
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
  
           
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
             LoggedSignals = [];
             vel = this.vel0;%暂时定为匀速
             Ts = this.Ts;
%             'x, dx, y,dy,phi, dphi,phi,theta';
            % Get action
           
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
            Tphi0 = 0;%初始航向角
            Ttheta0 = 0;%初始前轮转角
            Tx0 = 20;%初始X位置
            Ty0 = 20;%初始Y位置
            Tvel0 =5/3.5;%恒定5km/h
            Tdx0 = Tvel0;
            Tdy0 =0;
            TdPhi0=0;
            InitialObservation = [Tx0; Tdx0;Ty0;Tdy0;Tphi0;TdPhi0;Ttheta0];
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
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
             r1 =10*((XP^2+YP^2+PhiP^2)<1);
             r2 = -100*(abs(XP)>100||abs(YP)>100);
             r3 = -0.01*ThetaP^2-0.02*XP^2-0.02*YP^2-0.02*PhiP^2;
             Reward = r1+r2+r3;
             
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            
%            this.h = figure;
            % Update the visualization
%             envUpdatedCallback(this)
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
           
          
                return;
            % Extract the cart position and pole angle
            XP = this.State(1);
            XDotP = this.State(2);
            YP = this.State(3);
            YDotP = this.State(4);
            PhiP = this.State(5);
            PhiDotP = this.State(6);
            ThetaP = this.State(7);

            quiver(XP,YP,cos(PhiP),sin(PhiP))

            hold off
            xlim([-100 100])
            ylim([-100 200])
            end
    end
  
end

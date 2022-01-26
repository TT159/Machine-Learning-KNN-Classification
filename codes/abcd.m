% EC 503 Learning from Data
% Fall semester, 2021
% Homework 2
% by (Tian Tan)
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d

clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()

% label axis and include title
xlabel('First Column');
ylabel('Second Column');
title('Scatter Plot of Training Data');

% use gscatter function to create the scatter plot
figure(1);
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,'rgb');


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]); 
Xtest = [Xgrid(:),Ygrid(:)];% Xgrid & Ygrid are 96 x 96 matrix
[Ntest,dim]=size(Xtest); % Ntest = rows, dim = columns

% compute probabilities of being in class 2 for each point on grid
probabilities=zeros(9216,1);
Xdis=zeros(200,1);
% Prob of class 2
for i= 1:Ntest
    num=0; 
%      for j= 1:Ntrain
%          Xdis=sqrt(dot(Xtest(i,:),Xtest(i,:)')+dot(Xtrain(j,:),Xtrain(j,:)')-2*dot(Xtest(i,:),Xtrain(j,:)));
%      end
    Xdis=sqrt(Xtest(i,:).^2*ones(size(Xtrain'))+ones(size(Xtest(i,:)))*(Xtrain').^2-2*Xtest(i,:)*Xtrain');
    %Xdis=sqrt((Xtrain(:,1)-Xtest(i,1)).^2+(Xtrain(:,2)-Xtest(i,2)).^2);
    for m= 1:K
        [minvalue,index]=min(Xdis);
        Xdis(index)=inf;
        if ytrain(index)==2
            num=num+1;
        end
    end
    probabilities(i) = num/K;
end
% Figure for class 2
figure(2);
class2ProbonGrid = reshape(probabilities,size(Xgrid));%reshape the probabilities(9216x1) matrix to Xgrid size matrix(96 x96)
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X-axis')
ylabel('Y-axis')
title('Probabilities of Class 2')

% repeat steps above for class 3 below
% compute probabilities of being in class 3 for each point on grid

% probabilities=zeros(9216,1);
% Xdis=zeros(200,1);
% Class 3
for i= 1:Ntest
    num=0; 
    %Xdis=sqrt(Xtest(i,:).^2*ones(size(Xtrain'))+ones(size(Xtest(i,:)))*(Xtrain').^2-2*Xtest(i,:)*Xtrain');
    Xdis=sqrt((Xtrain(:,1)-Xtest(i,1)).^2+(Xtrain(:,2)-Xtest(i,2)).^2);
    for m= 1:K
        [minvalue,index]=min(Xdis);
        Xdis(index)=inf;% set this point as infinite
        if ytrain(index)==3
            num=num+1;
        end
    end
    probabilities(i) = num/K;
end
% Figure for class 3
figure(3);
class3ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X-axis')
ylabel('Y-axis')
title('Probabilities of Class 3')

%% c) Class label predictions
% K = 1 case
% compute predictions 
ypred = zeros(9216,1);
Xdis=zeros(200,1);

% Class 1,2,3
for i= 1:Ntest
    num=0; 
    %Xdis=sqrt(Xtest(i,:).^2*ones(size(Xtrain'))+ones(size(Xtest(i,:)))*(Xtrain').^2-2*Xtest(i,:)*Xtrain');
    Xdis=sqrt((Xtrain(:,1)-Xtest(i,1)).^2+(Xtrain(:,2)-Xtest(i,2)).^2);
    [minvalue,index]=min(Xdis);
%     if ytrain(index)==1
%        ypred(i) = 1;
%     elseif ytrain(index)==2  
%        ypred(i) = 2;
%     elseif ytrain(index)==3
%        ypred(i) = 3;
%     end
     ypred(i)=ytrain(index);
end
class1ProbonGrid = reshape(ypred,size(Xgrid));
%sketch the plot
figure(4);
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('X-axis')
ylabel('Y-axis')
title('Prediction of Test Points (K=1)')

% repeat steps above for the K=5 case. Include code for this below.
K = 5 ; % K = 1 case
ypred = zeros(9216,1);% predictions matrix
Xdis=zeros(200,1); % distance matrix
for i= 1:Ntest
    % To calculate min distance 
    % method 1:
    % Xdis=sqrt(Xtest(i,:).^2*ones(size(Xtrain'))+ones(size(Xtest(i,:)))*(Xtrain').^2-2*Xtest(i,:)*Xtrain');
    % method 2:
    Xdis=sqrt((Xtrain(:,1)-Xtest(i,1)).^2+(Xtrain(:,2)-Xtest(i,2)).^2);
    
    % To find the cloest point
    XclassK=zeros(K,1);  % the martix to store the class of K cloest neighboors
    XdisK=zeros(K,1);    % the matrix to store the distance of K cloest neighboors 
    for m = 1:K
        [mindist,index]=min(Xdis);  % to find the cloest point
        XclassK(m)=ytrain(index);   % to store the class of this point  
        XdisK(m)=mindist;   % to store the distance of this point
        Xdis(index)=inf;    % then, set this point to infinite
    end
    
    % To calculate the # of each class           this two loops can't be combination into one 
    Xnum=zeros(3,1);    % the matrix to store the # of each class                         p.s: can't in the loop otherwise will be refresh
    for m = 1:K    
        if XclassK(m)==1
            Xnum(1,:)=Xnum(1,:)+1;
        elseif XclassK(m)==2
            Xnum(2,:)=Xnum(2,:)+1;
        elseif XclassK(m)==3
            Xnum(3,:)=Xnum(3,:)+1;
        end
    end
    
    % To predicate the class 
    [maxvalue, classindex]=max(Xnum);    % Actually, the classindex indicates each class. both 1, 2, 3
    % maxvalue can be 3 or 4 or 5 -> 005,014,023,113
    %if maxvalue >= 2     
    if maxvalue > 2
        ypred(i)=classindex;    
    % maxvalue = 2 -> 1 2 2
    elseif maxvalue==2   
        [mindist,index]=min(XdisK); % Cauz two same #, we choose the cloest point among the K points as the best ans
        ypred(i)=XclassK(index);    %key point:XclassK与XdisK在同一index下，表示的是同一个点的类别和距离，所以在这里直接用XdisK的index就是可以的
    end
end

% To sketch the plot
class1ProbonGrid = reshape(ypred,size(Xgrid));
figure(5);
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('X-axis')
ylabel('Y-axis')
title('Prediction of Test Points (K=5)')

%% d) LOOCV CCR computations

Xtrain_tt=Xtrain;
for k = 1:2:11
    % determine leave-one-out predictions for k
    ypred = zeros(200,1); % to store the predication of training data
    for i = 1:200
        TestPoint=Xtrain(i,:); % to store the real value of test point
        Xtrain(i,:)=[inf,inf]; % in order to find the neighboor
        Xdis=sqrt((Xtrain(:,1)-TestPoint(1)).^2+(Xtrain(:,2)-TestPoint(2)).^2); % to calculate the distance to rest points
        
        % To find k cloest points
        XclassK=zeros(k,1);  % the martix to store the class of K cloest neighboors
        XdisK=zeros(k,1);    % the matrix to store the distance of K cloest neighboors 
        for m = 1:k
            [mindist,index]=min(Xdis);  % to find the cloest point
            XclassK(m)=ytrain(index);   % to store the class of this point  
            XdisK(m)=mindist;   % to store the distance of this point
            Xdis(index)=inf;    % then, set this point to infinite
        end
        
        Xtrain=Xtrain_tt;   % to restore the training data matrix
        
        % To calculate the # of each class among k points
        Xnum=zeros(3,1);          
        for m = 1:k    
            if XclassK(m)==1
                Xnum(1,:)=Xnum(1,:)+1;
            elseif XclassK(m)==2
                Xnum(2,:)=Xnum(2,:)+1;
            elseif XclassK(m)==3
                Xnum(3,:)=Xnum(3,:)+1;
            end
        end
        
        % To predicate the class 
        [maxvalue,classindex]=max(Xnum);    
        [minvalue,noneindex]=min(Xnum);
        if k==1
            ypred(i)=XclassK(1,1);
        elseif k==3 %003 012 111
            if maxvalue==1
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index); 
            else
                ypred(i)=classindex;
            end
        elseif k==5 %005,014,023,113,122, 
            if maxvalue > 2
                ypred(i)=classindex;    
            else   
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index);    
            end
        elseif k==7 %007,016,025,034,115,124,133,223,
             if maxvalue==3 && minvalue==1 
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index);       
             else   
                ypred(i)=classindex; 
             end   
        elseif k==9 %009,018,027,036,045,117,126,135,144,225,234,333,
             if maxvalue==3 
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index);       
             elseif maxvalue==4 && minvalue==1  
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index);
             else
                ypred(i)=classindex; 
             end 
        elseif k==11 %00 11,01 10,029,038,047,056,119,128,137,146,155,227,236,245,335,344
            if maxvalue==4
               [mindist,index]=min(XdisK); 
               ypred(i)=XclassK(index);   
            elseif maxvalue==5 && minvalue==1
                [mindist,index]=min(XdisK); 
                ypred(i)=XclassK(index);         
            else
                ypred(i)=classindex;
            end
        end
            
    end %for i = 1:200
    
    
    % To compute confusion matrix
    conf_mat = confusionmat(ytrain, ypred);
    % To sketch the plot
%     figure(6);
%     imagesc(conf_mat);
%     colormap(flipud(gray));
%     title('The confusion matrix of predicaton of training data ');
    % To compute CCR from confusion matrix
    sum=trace(conf_mat);
    CCR = sum/200;
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

    % plot CCR values for k = 1,3,5,7,9,11
    figure(6);
    plot(1:2:11,CCR_values);
    % label x/y axes and include title
    xlabel('the values ok k');
    ylabel('the CCR values');
    title('the CCR vaules according to k');
    
    

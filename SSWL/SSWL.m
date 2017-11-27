function [pre_Test_labels,W,W_star,L]=SSWL(Train_Matrix,Train_Label,Test_Matrix,alpha,beta,zeta)
%%  This function is SSWL method
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
%%  ATTN2
%   ATTN2: This package was developed by Mr. Hao-Chen Dong (donghc@lamda.nju.edu.cn). For any problem concerning the code,
%   please feel free to contact Mr. Dong.
%%  Some varables used in the code
%   input varables:
%       Train_Matrix: input feature vectors of training data and unlabeled data
%       Train_Label: label matrix of training data and unlabeled data
%       Test_Matrix: input feature vectors of test data
%       alpha: the parameter controls smoothness of prediction
%       beta: the parameter controls the consistence of two models
%       zeta: the parameter controls the second model's prediction on uncertain elements
%   output varables:
%       pre_Test_labels: prediction on the test data
%       W: coefficient matrix of first model
%       W_star: coefficient matrix of second model
%       L: label similarity matrix
%%  Reference:
%   H.-C. Dong, Y.-F. Li and Z.-H. Zhou. Learning from Semi-Supervised Weak-Label Data.

iter_num=20;
eps=0.01;

n = size(Train_Matrix,1);%all data = label data + unlabel data
m = size(Train_Label,2);%label number

data = sparse([Train_Matrix,ones(n,1)]); 
Test_data = sparse([Test_Matrix,ones(size(Test_Matrix,1),1)]);
train_label = double(Train_Label);

kk = size(data,2);

K_instance = 5;
S = slaffinitymat(data', [ ], {'ann', 'K', K_instance},'excludeself',true,'sym',true);
S = spdiags(1./sum(S,2),0,n,n)*S;

W=speye(kk,m);
W_star=sparse(kk,m);
L=speye(m);

label_1 = train_label;
label_1(label_1==-1) = 0;
label_F_1 = -train_label;
label_F_1(label_F_1==-1) = 0;

%initial
I_X = sparse(kron(speye(m,m),data));
I_X_T = I_X';

dia_Y = sparse(diag(label_1(:)));
dia_I_Y = sparse(diag(label_F_1(:)));

%before
p1 = data*W;
p2 = data*W_star;
O = (p1).*label_1+(p2).*label_F_1;
L_S = kron(L',S);
T = S*O;

I_X_W_star = p2(:);
loss_1=sum(sum( (p1.*label_1-label_1).^2))+alpha*sum(sum((O-T*L).^2))+...
    beta*sum(sum( ((p1-p2).*label_F_1).^2))+zeta*sum(sum( (p2.*label_F_1+label_F_1).^2));

T1 = I_X_T*dia_Y*I_X;
T2 = I_X_T*dia_I_Y*I_X;
T3 = I_X_T*dia_Y;
T4 = I_X_T*(label_1(:));
T5 = I_X_T*dia_I_Y;
T6 = I_X_T*(label_F_1(:));
I=speye(n*m,n*m);

G_W = T1 + beta*T2 ;
G_W_star = zeta*T2 + beta*T2;
clear T1 T2
clear Train_Matrix Train_Label
for k=1:iter_num
    %% ================================================================================
    %derivation-W
    G = G_W + alpha*T3*(I-L_S)'*(I-L_S)*(T3)';
    B_W = T4 + beta*T5*I_X_W_star - alpha*T3*(I-L_S)'*(I-L_S)*dia_I_Y*I_X_W_star;
    W = reshape(pcg(G,B_W,1e-6,5000),kk,m);
    
    p1 = data*W;
    I_X_W = p1(:);
    clear B_W G;
    
    %derivation-W_star
    G_star = G_W_star + alpha*T5*(I-L_S)'*(I-L_S)*(T5)';
    B_W_star = -zeta*T6 + beta*T5*I_X_W - alpha*T5*(I-L_S)'*(I-L_S)*dia_Y*I_X_W;
    W_star = reshape(pcg(G_star,B_W_star,1e-6,5000),kk,m);
    
    p2 = data*W_star;
    I_X_W_star = p2(:);
    clear G_star B_W_star;
    
    O = (p1).*label_1+(p2).*label_F_1;
    T = S*O;
    
    %derivation-L
    G1 = kron(speye(m,m),T'*T);
    B1 = T'*O;
    L = reshape(pcg(G1,B1(:),1e-6,5000),m,m);
    clear G1 B1;
    
    %% ================================================================================
    %after
    L_S = kron(L',S);
    
    loss_2=sum(sum( (p1.*label_1-label_1).^2))+alpha*sum(sum((O-T*L).^2))+...
        beta*sum(sum( ((p1-p2).*label_F_1).^2))+zeta*sum(sum( (p2.*label_F_1+label_F_1).^2));
    fprintf('loss_2 %d\n',loss_2);
    if abs(loss_1-loss_2)<eps
        break;
    end
    loss_1 = loss_2;
end

pre_Test_labels = Test_data*W;
pre_Test_labels(pre_Test_labels>0) = 1;
pre_Test_labels(pre_Test_labels<=0) = -1;
end

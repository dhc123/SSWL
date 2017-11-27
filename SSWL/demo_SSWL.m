clear;clc;
addpath(genpath('.\'))
datasetdir='.\';
dataname={'yea'};
ratiop=0.6;

%This is an example for yeaInfor0.6, you can get a basic performance on alpha = 1.6, beta = 0.06, zeta = 0.06 and you can tune the parameter to improve the performance on the specific evaluation metric.

alpha = 1.6;
beta = 0.06;
zeta = 0.06;

for idata=1:length(dataname)
    for rp=1:length(ratiop)
        dataf=strcat(datasetdir,dataname(idata),'\',dataname(idata),'Infor',num2str(ratiop(rp)),'.mat');
        datafname=cell2mat(dataf(1));
        load (datafname);
        tr_data = [Train_Matrix ; UnLabel_Matrix];
        tr_label = double([Z;-1*ones(size(UnLabel_Matrix,1),size(Z,2))]);
        [pre_labels,W,~,~]=SSWL(tr_data,tr_label,Test_Matrix,alpha,beta,zeta);
        opt.type='marco';
        [~,...
            ~,...
            macro_F]=PRF1(Test_Label',pre_labels',opt);
        opt.type='micro';
        [~,...
            ~,...
            micro_F]=PRF1(Test_Label',pre_labels',opt);
        hl=Hamming_loss(Test_Label',pre_labels');
        fprintf('hl=%d\n',hl);
        fprintf('macro_F=%d\n',macro_F);
        fprintf('micro_F=%d\n',micro_F);
    end
end


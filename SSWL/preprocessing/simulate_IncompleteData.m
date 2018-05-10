function simulate_IncompleteData( data_name, labeled_proportion, unlabeled_proportion)

% load dataset
load([data_name,'/',data_name,'.mat']);
fprintf(data_name,'\n');

% normalize the feature data
Feature_Matrix = maxminnorm(data);
Label_Matrix = target';
% get dataset size
[instance_row instance_col] = size(Feature_Matrix);
[~,label_dim] = size(Label_Matrix);
% set label 0 as -1
Label_Matrix(Label_Matrix == 0) = -1;

% train_num = round(instance_row*labeled_proportion);
train_num = 500;

random_train_Index = randperm(train_num*label_dim);
% unlabel_num = round(instance_row*unlabeled_proportion);
unlabel_num = 1000;

test_num = instance_row - train_num - unlabel_num;

incomplete_Set = [0 0.1 0.2 0.3 0.4 0.5 0.6];

Train_Matrix = Feature_Matrix(R(1:train_num),:);
Train_Label = Label_Matrix(R(1:train_num),:);
UnLabel_Matrix = Feature_Matrix(R(train_num+1:train_num + unlabel_num),:);
Test_Matrix = Feature_Matrix(R(train_num+unlabel_num+1:end),:);
Test_Label = Label_Matrix(R(train_num+unlabel_num+1:end),:);

% dimension reduction
%     PCA_K = 50;
%     [COEFF,SCORE,latent] = princomp([Train_Matrix;UnLabel_Matrix]);
%     Train_Matrix = Train_Matrix*COEFF(:,1:PCA_K);
%     UnLabel_Matrix = UnLabel_Matrix*COEFF(:,1:PCA_K);
%     Test_Matrix = Test_Matrix*COEFF(:,1:PCA_K);

for inc = 1 : 7
    incomplete = incomplete_Set(1,inc);    
    
    R = randperm(instance_row);
        
    Z = createIncomplete_Label(Train_Label, incomplete,random_train_Index);

    save([data_name,'/',data_name,'Infor',num2str(incomplete),'.mat'],'Train_Matrix','Test_Matrix','Z','Train_Label','Test_Label','UnLabel_Matrix','-v7');
end

end


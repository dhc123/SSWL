function [ Incomplete_Label ] = createIncomplete_Label(Train_Label, proportion,random_train_Index)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Incomplete_Label = Train_Label;
[row col] = size(Train_Label);

pick = round(row*col * proportion);

for i = 1 : pick
    index = random_train_Index(i);
    Incomplete_Label(index) = -1;

end

end

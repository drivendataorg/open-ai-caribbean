%% Train and ensemble result
clear;
clc;

%% Set some parameters
train_indicator = false; %change this value to true if wish to train on your machine.

Folder = cd;
Folder = fullfile(Folder, '..');
train_path = fullfile(Folder, 'data\processed\train'); %path of the generated train data
test_path = fullfile(Folder, 'data\processed\test'); %path of the generated train data
mask_path = fullfile(Folder, 'data\processed\data_mask'); %path of the generated mask data
model_path = fullfile(Folder, 'models'); %path of the generated train data

%% Load Training and testing data
%Training data
imds = imageDatastore(train_path,"IncludeSubfolders",true, ...
    "FileExtensions",".png","LabelSource","foldernames");

% Testing data
imdsTest = imageDatastore(test_path,"FileExtensions",".png");

%% Train resnet18
net = resnet18;

numClasses = numel(categories(imds.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

inputSize = net.Layers(1).InputSize;
imds.ReadFcn = @(im)readAugAndResize(im,inputSize,mask_path); 


%[imdsTrain,imdsVal] = splitEachLabel(imds,0.9,"randomized");
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
%     'MiniBatchSize',32, ...
%     'MaxEpochs',8, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',60 , ...
%     'LearnRateDropFactor',0.1, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsVal, ...
%     'ValidationFrequency',floor(numel(imdsTrain.Files)/(32)), ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

if train_indicator == true
    netTransfer = trainNetwork(imds,lgraph,options);
else
    load(fullfile(model_path, 'resnet18.mat'));
end

%% Create resnet18 test result
imdsTest.ReadFcn = @(im)readAndResize(im,inputSize,mask_path);
sub_table = readtable('submission_format.csv'); 
id_list = sub_table.id;
id_list_file = {}; 
for i = 1:numel(id_list)
    id_list_file{i} = fullfile(test_path, [id_list{i} '.png']); 
end

id_list_file = id_list_file'; 
imdsTest.Files = id_list_file; 

[testMaterial,testScores] = classify(netTransfer,imdsTest);

testResults = table(id_list,testScores(:,1),testScores(:,2), ...
    testScores(:,3),testScores(:,4),testScores(:,5), ...
    'VariableNames',['id';categories(testMaterial)]);

writetable(testResults,'resnet18_result.csv');


%% train resnet50
net = resnet50;

numClasses = numel(categories(imds.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);

inputSize = net.Layers(1).InputSize;
imds.ReadFcn = @(im)readAugAndResize(im,inputSize,mask_path); % Refers to a helper function at the end of this script

%[imdsTrain,imdsVal] = splitEachLabel(imds,0.9,"randomized");
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
%     'MiniBatchSize',32, ...
%     'MaxEpochs',8, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',60 , ...
%     'LearnRateDropFactor',0.1, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsVal, ...
%     'ValidationFrequency',floor(numel(imdsTrain.Files)/(32)), ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');


if train_indicator == true
    netTransfer = trainNetwork(imds,lgraph,options);
else
    load(fullfile(model_path, 'resnet50.mat'));
end

%% Create resnet50 test result
imdsTest.ReadFcn = @(im)readAndResize(im,inputSize,mask_path);
sub_table = readtable('submission_format.csv'); 
id_list = sub_table.id;
id_list_file = {}; 
for i = 1:numel(id_list)
    id_list_file{i} = fullfile(test_path, [id_list{i} '.png']); 
end

id_list_file = id_list_file'; 
imdsTest.Files = id_list_file; 

[testMaterial,testScores] = classify(netTransfer,imdsTest);

testResults = table(id_list,testScores(:,1),testScores(:,2), ...
    testScores(:,3),testScores(:,4),testScores(:,5), ...
    'VariableNames',['id';categories(testMaterial)]);

writetable(testResults,'resnet50_result.csv');

%% Train resnet101
net = resnet101;

numClasses = numel(categories(imds.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

inputSize = net.Layers(1).InputSize;
imds.ReadFcn = @(im)readAugAndResize(im,inputSize,mask_path); % Refers to a helper function at the end of this script

% [imdsTrain,imdsVal] = splitEachLabel(imds,0.9,"randomized");
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
%     'MiniBatchSize',32, ...
%     'MaxEpochs',8, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',20 , ...
%     'LearnRateDropFactor',0.1, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsVal, ...
%     'ValidationFrequency',floor(numel(imdsTrain.Files)/(32)), ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','parallel', ... % Turn on automatic parallel support.
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

if train_indicator == true
    netTransfer = trainNetwork(imds,lgraph,options);
else
    load(fullfile(model_path, 'resnet101.mat'));
end

%% Create resnet101 test result
imdsTest.ReadFcn = @(im)readAndResize(im,inputSize,mask_path);
sub_table = readtable('submission_format.csv'); 
id_list = sub_table.id;
id_list_file = {}; 
for i = 1:numel(id_list)
    id_list_file{i} = fullfile(test_path, [id_list{i} '.png']); 
end

id_list_file = id_list_file'; 
imdsTest.Files = id_list_file; 

[testMaterial,testScores] = classify(netTransfer,imdsTest);

testResults = table(id_list,testScores(:,1),testScores(:,2), ...
    testScores(:,3),testScores(:,4),testScores(:,5), ...
    'VariableNames',['id';categories(testMaterial)]);

writetable(testResults,'resnet101_result.csv');

%% Ensemble the data by simply finding the average
T1 = readtable('resnet18_result.csv');
T2 = readtable('resnet50_result.csv');
T3 = readtable('resnet101_result.csv');

T_final = T1; 
T_final.concrete_cement = (T1.concrete_cement + T2.concrete_cement + T3.concrete_cement)/3;
T_final.healthy_metal = (T1.healthy_metal + T2.healthy_metal + T3.healthy_metal)/3;
T_final.incomplete = (T1.incomplete + T2.incomplete + T3.incomplete)/3;
T_final.irregular_metal = (T1.irregular_metal + T2.irregular_metal + T3.irregular_metal)/3;
T_final.other = (T1.other + T2.other + T3.other)/3;

writetable(T_final,'submission.csv');
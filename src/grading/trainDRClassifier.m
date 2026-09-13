function [net,info] = trainDRClassifier(trainData, opts)
%TRAINDRCLASSIFIER Import EfficientNet ONNX or prepare native transfer learning.
arguments
    trainData
    opts.ONNXPath string = ""
    opts.Mode string {mustBeMember(opts.Mode,["onnx","native"])} = "onnx"
    opts.ClassWeights double = []
    opts.ValidationData = []
end
if opts.Mode=="onnx"
    assert(strlength(opts.ONNXPath)>0 && isfile(opts.ONNXPath), ...
        'DrishtiMitra:ONNXMissing','Provide the approved prior EfficientNet-B0 ONNX file.');
    net=importONNXNetwork(opts.ONNXPath,'TargetNetwork','dlnetwork');
    info=struct('mode','ONNX imported through Deep Learning Toolbox','trained',false, ...
        'warning','Fine-tune and validate this imported network before clinical performance claims.');
    return
end
% Native path intentionally requires explicit labelled datastore; no silent demo training.
assert(~isempty(trainData),'DrishtiMitra:TrainingDataMissing','A labelled image datastore is required for native training.');
classes = validateFiveClassDatastore(trainData,'trainData');
if ~isempty(opts.ValidationData)
    validateFiveClassDatastore(opts.ValidationData,'opts.ValidationData');
end
if ~isempty(opts.ClassWeights)
    assert(isnumeric(opts.ClassWeights) && numel(opts.ClassWeights) == 5 && ...
        all(isfinite(opts.ClassWeights),'all') && all(opts.ClassWeights > 0,'all'), ...
        'DrishtiMitra:ClassWeights', ...
        'ClassWeights must contain five finite, strictly positive values ordered as [0,1,2,3,4].');
end
base=efficientnetb0; lgraph=layerGraph(base);
% Locate terminal layers by type rather than pinning a release-specific name.
layers=lgraph.Layers; fcIdx=find(arrayfun(@(x) isa(x,'nnet.cnn.layer.FullyConnectedLayer'),layers),1,'last');
classIdx=find(arrayfun(@(x) isa(x,'nnet.cnn.layer.ClassificationOutputLayer'),layers),1,'last');
assert(~isempty(fcIdx) && ~isempty(classIdx),'DrishtiMitra:NetworkHead','Unable to locate EfficientNet classifier head.');
n=numel(classes);
newFC=fullyConnectedLayer(n,'Name','dr_head','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
if isempty(opts.ClassWeights)
    newClass = classificationLayer('Name','dr_output','Classes',categorical(classes,classes,'Ordinal',true));
else
    newClass = classificationLayer('Name','dr_output', ...
        'Classes',categorical(classes,classes,'Ordinal',true), ...
        'ClassWeights',opts.ClassWeights);
end
lgraph=replaceLayer(lgraph,layers(fcIdx).Name,newFC);
lgraph=replaceLayer(lgraph,layers(classIdx).Name,newClass);
trainingArgs = {'MiniBatchSize',16,'MaxEpochs',12,'InitialLearnRate',0.0001,'Verbose',false};
if isempty(opts.ValidationData)
    trainingConfig = trainingOptions('adam',trainingArgs{:}, ...
    'ExecutionEnvironment','gpu', ...
    'Plots','training-progress', ...
    'Verbose',true);
else
    trainingConfig = trainingOptions('adam',trainingArgs{:}, ...
    'ValidationData',opts.ValidationData, ...
    'OutputNetwork','best-validation', ...
    'ExecutionEnvironment','gpu', ...
    'Plots','training-progress', ...
    'Verbose',true);
end
net=trainNetwork(trainData,lgraph,trainingConfig);
info=struct('mode','native EfficientNet-B0 transfer learning','trained',true,'classes',{classes});
end

function classes = validateFiveClassDatastore(data, inputName)
assert(isprop(data,'Labels') && ~isempty(data.Labels), ...
    'DrishtiMitra:TrainingDataLabels','%s must be a non-empty datastore with Labels.',inputName);
assert(iscategorical(data.Labels),'DrishtiMitra:TrainingDataLabels', ...
    '%s.Labels must be categorical.',inputName);
classes = categories(data.Labels);
expectedClasses = ["0","1","2","3","4"];
assert(numel(classes) == 5 && isequal(string(classes(:)).',expectedClasses), ...
    'DrishtiMitra:TrainingDataClasses', ...
    '%s.Labels must have exactly these ordered categories: ["0","1","2","3","4"].',inputName);
end

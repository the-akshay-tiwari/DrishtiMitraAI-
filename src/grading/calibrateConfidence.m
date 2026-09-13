function calibration = calibrateConfidence(scores, labels, method)
%CALIBRATECONFIDENCE Fit held-out probability calibration (binary referable DR).
arguments; scores (:,1) double; labels (:,1) logical; method string = "platt"; end
assert(numel(scores)==numel(labels),'DrishtiMitra:CalibrationSize','Scores and labels must align.');
switch method
    case "platt"
        mdl=fitglm(scores,labels,'Distribution','binomial','Link','logit');
        predictFcn=@(x) predict(mdl,x);
    otherwise
        [xs,idx]=sort(scores); ys=labels(idx); cumulative=cumsum(ys)./(1:numel(ys))';
        mdl=struct('x',xs,'y',cumulative);
        predictFcn=@(x) interp1(xs,cumulative,x,'linear','extrap');
end
calibration=struct('method',method,'model',mdl,'predict',predictFcn);
end

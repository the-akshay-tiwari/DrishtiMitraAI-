function qwk = computeQWK(actual, predicted, nClasses)
%COMPUTEQWK Quadratic weighted kappa, consistent with ordinal DR evaluation.
arguments; actual (:,1) double; predicted (:,1) double; nClasses (1,1) double = max([actual;predicted])+1; end
assert(numel(actual)==numel(predicted),'DrishtiMitra:QWKSize','Inputs must have equal lengths.');
actual=actual+1; predicted=predicted+1;
O=confusionmat(actual,predicted,'Order',1:nClasses); E=(sum(O,2)*sum(O,1))/sum(O,'all');
[i,j]=ndgrid(1:nClasses); W=(i-j).^2/(nClasses-1)^2;
qwk=1-sum(W.*O,'all')/max(eps,sum(W.*E,'all'));
end

function metrics = computeSensSpec(scores, labels, threshold)
%COMPUTESENSSPEC Report ROC and select an honest operating point.
arguments; scores (:,1) double; labels (:,1) logical; threshold (1,1) double = NaN; end
[fpr,tpr,rocThresholds,auc]=perfcurve(labels,scores,true);
if isnan(threshold)
    d=hypot(tpr-.90,(1-fpr)-.85); [~,idx]=min(d); threshold=rocThresholds(idx);
end
pred=scores>=threshold; tp=nnz(pred&labels); fn=nnz(~pred&labels); tn=nnz(~pred&~labels); fp=nnz(pred&~labels);
metrics=struct('threshold',threshold,'sensitivity',tp/max(1,tp+fn),'specificity',tn/max(1,tn+fp), ...
 'confusionMatrix',[tn fp;fn tp],'rocFPR',fpr,'rocTPR',tpr,'rocThresholds',rocThresholds,'auc',auc, ...
 'targetMet',tp/max(1,tp+fn)>.90 && tn/max(1,tn+fp)>.85);
end

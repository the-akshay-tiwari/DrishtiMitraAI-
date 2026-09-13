function evidence = lesionEvidenceReport(attention, lesions)
%LESIONEVIDENCEREPORT Quantify whether attention overlaps segmented evidence.
attention=im2double(attention); if ndims(attention)==3, attention=rgb2gray(attention); end
names=fieldnames(lesions); rows=table();
for k=1:numel(names)
    v=lesions.(names{k}); if ~isstruct(v)||~isfield(v,'mask'), continue, end
    m=imresize(logical(v.mask),size(attention),'nearest'); at=attention>=prctile(attention(:),85);
    inter=nnz(at&m); union=nnz(at|m); corrVal=corr(attention(:),double(m(:)),'Rows','complete');
    rows=[rows; table(string(names{k}),inter/max(1,nnz(m)),inter/max(1,union),corrVal, ...
        'VariableNames',{'Lesion','AttentionRecall','IoU','SpatialCorrelation'})]; %#ok<AGROW>
end
evidence=rows;
end

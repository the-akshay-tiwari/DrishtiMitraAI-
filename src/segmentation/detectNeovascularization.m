function result = detectNeovascularization(I, opticDisc)
%DETECTNEOVASCULARIZATION Flag tortuous/branch-dense vessel irregularity (NVD/NVE).
if nargin<2, opticDisc=localizeOpticDiscFovea(I); end
v=segmentVessels(I,'ClassicalOnly',true).mask; sk=bwskel(v); bp=bwmorph(sk,'branchpoints');
[X,Y]=meshgrid(1:size(v,2),1:size(v,1)); d=hypot(X-opticDisc.opticDiscCenter(1),Y-opticDisc.opticDiscCenter(2));
nvdRegion=d<2.5*opticDisc.opticDiscRadius; nvdDensity=mean(sk(nvdRegion)); nveDensity=mean(sk(~nvdRegion));
branchNVD=nnz(bp & nvdRegion)/max(1,nnz(nvdRegion)); branchNVE=nnz(bp & ~nvdRegion)/max(1,nnz(~nvdRegion));
result=struct('vesselMask',v,'nvdScore',nvdDensity+20*branchNVD,'nveScore',nveDensity+20*branchNVE, ...
    'nvdFlag',nvdDensity+20*branchNVD>.12,'nveFlag',nveDensity+20*branchNVE>.06, ...
    'note','Thresholds require calibration against labelled proliferative-DR data.');
end

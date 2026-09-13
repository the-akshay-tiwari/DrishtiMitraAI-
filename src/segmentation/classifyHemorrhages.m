function result = classifyHemorrhages(I, opticDisc)
%CLASSIFYHEMORRHAGES Detect dark red-free lesions and score ICDR quadrants.
if nargin<2, opticDisc=localizeOpticDiscFovea(I); end
I=im2double(I); redFree=I(:,:,min(2,size(I,3)));
marker=imerode(redFree,strel('disk',9)); reconstructed=imreconstruct(marker,redFree);
mask=bwareaopen(reconstructed-redFree > prctile(reconstructed(:)-redFree(:),94),12);
cx=opticDisc.opticDiscCenter(1); cy=opticDisc.opticDiscCenter(2); [X,Y]=meshgrid(1:size(mask,2),1:size(mask,1));
q=[nnz(mask & X<=cx & Y<=cy),nnz(mask & X>cx & Y<=cy),nnz(mask & X<=cx & Y>cy),nnz(mask & X>cx & Y>cy)];
result=struct('mask',mask,'quadrantAreaPixels',q,'affectedQuadrants',sum(q>0),'icdrExtent',sum(q>0));
end

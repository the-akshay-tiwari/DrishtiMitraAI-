function result = detectMicroaneurysms(I, model)
%DETECTMICROANEURYSMS Extract candidate features and classify with a trained SVM.
if nargin<2, model=[]; end
I=im2uint8(I); G=im2double(I(:,:,min(2,size(I,3))));
C=imtophat(imcomplement(G),strel('disk',5)); bw=bwareaopen(C>graythresh(C),2);
s=regionprops(bw,C,'Area','Eccentricity','MeanIntensity','PixelIdxList','Centroid');
X=[[s.Area]' [s.Eccentricity]' [s.MeanIntensity]']; keep=true(numel(s),1);
if ~isempty(model) && ~isempty(s), keep=predict(model,X)==1; end
out=false(size(bw)); for k=find(keep)', out(s(k).PixelIdxList)=true; end
result=struct('mask',out,'candidates',{s},'features',X,'count',nnz(keep),'classifier',class(model));
end

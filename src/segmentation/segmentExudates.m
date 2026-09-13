function result = segmentExudates(I, opticDisc)
%SEGMENTEXUDATES Segment bright yellow-white candidate lesions outside optic disc.
if nargin<2, opticDisc=localizeOpticDiscFovea(I); end
L=rgb2lab(im2uint8(I)); a=L(:,:,2); b=L(:,:,3); lum=L(:,:,1);
mask=lum>prctile(lum(:),88) & b>prctile(b(:),55) & a<prctile(a(:),75);
[X,Y]=meshgrid(1:size(mask,2),1:size(mask,1));
disc=(X-opticDisc.opticDiscCenter(1)).^2+(Y-opticDisc.opticDiscCenter(2)).^2 <= (1.25*opticDisc.opticDiscRadius).^2;
mask=bwareaopen(mask & ~disc,10); mask=imclose(mask,strel('disk',2));
result=struct('mask',mask,'areaPixels',nnz(mask),'opticDiscExcluded',true);
end

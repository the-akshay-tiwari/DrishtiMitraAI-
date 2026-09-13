function result = localizeOpticDiscFovea(I)
%LOCALIZEOPTICDISCFOVEA Locate bright disc and infer/refine temporal fovea.
I = im2uint8(I); if ndims(I)==3, G=I(:,:,2); else, G=I; end
R = [round(min(size(G))/20) round(min(size(G))/6)];
[centres,radii,metric] = imfindcircles(imadjust(G),R,'ObjectPolarity','bright','Sensitivity',.88);
if isempty(centres)
    [~,idx] = max(imgaussfilt(im2double(G),8),[],'all','linear'); [y,x]=ind2sub(size(G),idx);
    centre=[x y]; radius=round(mean(R)); confidence=0;
else
    [confidence,k]=max(metric); centre=centres(k,:); radius=radii(k);
end
% The fovea is about 2.5 disc diameters temporal to the disc. Select the
% darker of symmetric candidates to avoid hard-coding left/right eye.
candidates = [centre + [5*radius 0]; centre - [5*radius 0]];
candidates = max(candidates,1); candidates(:,1)=min(candidates(:,1),size(G,2)); candidates(:,2)=min(candidates(:,2),size(G,1));
vessel = segmentVessels(I,'ClassicalOnly',true); scores=zeros(2,1);
for n=1:2
    x=round(candidates(n,1)); y=round(candidates(n,2)); rr=max(4,round(radius/2));
    x1=max(1,x-rr);x2=min(size(G,2),x+rr);y1=max(1,y-rr);y2=min(size(G,1),y+rr);
    scores(n)=mean(G(y1:y2,x1:x2),'all') + 20*mean(vessel.mask(y1:y2,x1:x2),'all');
end
[~,k]=min(scores);
result=struct('opticDiscCenter',centre,'opticDiscRadius',radius,'opticDiscConfidence',confidence, ...
    'foveaCenter',candidates(k,:),'method','bright-disc Hough + temporal/vessel refinement');
end

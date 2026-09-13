function result = segmentVessels(I, varargin)
%SEGMENTVESSELS Classical matched-filter baseline; accepts an optional trained net.
p=inputParser; addParameter(p,'Network',[]); addParameter(p,'ClassicalOnly',false); parse(p,varargin{:});
I=im2uint8(I); G=im2double(I(:,:,min(2,size(I,3))));
bg=imopen(G,strel('disk',15)); C=imadjust(mat2gray(bg-G));
angles=0:15:165; response=zeros(size(G));
for a=angles
    h=fspecial('gaussian',[15 15],1.2); h=imrotate(h,a,'crop');
    response=max(response,imfilter(C,h,'replicate'));
end
tophat=max(0,C-imopen(C,strel('disk',5)));
classical=bwareaopen(response + .6*tophat > graythresh(response + .6*tophat),20);
mask=classical; method='classical matched-filter + multi-scale top-hat';
if ~p.Results.ClassicalOnly && ~isempty(p.Results.Network)
    try
        pred=semanticseg(imresize(I,p.Results.Network.Layers(1).InputSize(1:2)),p.Results.Network);
        mask=imresize(pred ~= "background",size(G),'nearest'); method='trained U-Net/DeepLab';
    catch ME
        warning('DrishtiMitra:VesselFallback','Network inference failed (%s); classical baseline used.',ME.message)
    end
end
result=struct('mask',mask,'classicalMask',classical,'method',method);
end

function patches = extractIDRiDPatches(datasetReport, imagesInput, patchSize, patchesPerImage, scaleFactor, randomSeed)
%EXTRACTIDRIDPATCHES Extract channel-aware 512x512 patches from IDRiD dataset.
%
%   patches = extractIDRiDPatches(datasetReport, imagesInput, patchSize, patchesPerImage, scaleFactor, randomSeed)
%   - datasetReport: output from validateIDRiDMasks()
%   - imagesInput: "training", "testing", or table slice of image metadata
%   - patchSize: [H, W], default [512, 512]
%   - patchesPerImage: default 8 (or per opts)
%   - scaleFactor: default 0.5 (downsamples 4288x2848 to 2144x1424)
%   - randomSeed: default 42 for fixed reproducibility

if nargin < 1 || isempty(datasetReport)
    datasetReport = validateIDRiDMasks();
end
if nargin < 2 || isempty(imagesInput)
    imagesInput = "training";
end
if nargin < 3 || isempty(patchSize)
    patchSize = [512, 512];
end
if nargin < 4 || isempty(patchesPerImage)
    patchesPerImage = 8;
end
if nargin < 5 || isempty(scaleFactor)
    scaleFactor = 0.5;
end
if nargin < 6 || isempty(randomSeed)
    randomSeed = 42;
end

rng(randomSeed);

if istable(imagesInput)
    images = imagesInput;
    imageSet = string(images.ImageSet(1));
else
    imageSet = string(imagesInput);
    images = datasetReport.images(datasetReport.images.ImageSet == imageSet, :);
end

nImages = height(images);
assert(nImages > 0, 'DrishtiMitra:ExtractPatches', ...
    'No images found for split: %s', imageSet);

pH = patchSize(1);
pW = patchSize(2);
types = ["MA", "HE", "EX", "SE", "OD"];

totalPatches = nImages * patchesPerImage;
imagesCell = cell(totalPatches, 1);
masksCell = cell(totalPatches, 1);
validityCell = cell(totalPatches, 1);
imageIDCell = cell(totalPatches, 1);

patchIdx = 0;

% For tracking patch statistics
channelPosCounts = zeros(1, 5);
multiPosCount = 0;
pureBackgroundCount = 0;

for i = 1:nImages
    imgID = images.ID(i);
    imgPath = images.ImageFile(i);
    curImgSet = images.ImageSet(i);
    
    img = imread(imgPath);
    if ismatrix(img)
        img = repmat(img, 1, 1, 3);
    end
    
    imgScaled = imresize(img, scaleFactor, 'bilinear');
    [H, W, ~] = size(imgScaled);
    
    masksScaled = false(H, W, 5);
    validity = false(1, 5);
    
    % Store pixel coordinates per channel for channel-aware sampling
    channelPosCoords = cell(5, 1);
    validPosChannels = [];
    
    for k = 1:numel(types)
        typeMatches = false(height(datasetReport.records), 1);
        for r = 1:height(datasetReport.records)
            typeMatches(r) = matchesType(datasetReport.records.Type(r), types(k));
        end
        rows = datasetReport.records(datasetReport.records.ID == imgID & ...
            datasetReport.records.ImageSet == curImgSet & typeMatches, :);
        if height(rows) == 1
            rawMask = readValidatedBinaryMask(rows.MaskFile(1));
            bMask = imresize(rawMask, [H, W], 'nearest') > 0;
            masksScaled(:,:,k) = bMask;
            validity(k) = true;
            
            [rIdx, cIdx] = find(bMask);
            if ~isempty(rIdx)
                channelPosCoords{k} = [rIdx, cIdx];
                validPosChannels(end+1) = k; %#ok<AGROW>
            end
        end
    end
    
    numLesionPatches = ceil(patchesPerImage / 2);
    numPosChan = numel(validPosChannels);
    
    for p = 1:patchesPerImage
        isLesionPatch = (p <= numLesionPatches) && (numPosChan > 0);
        
        if isLesionPatch
            % Channel-aware sampling: cycle through valid positive channels
            targetChanIdx = validPosChannels(mod(p - 1, numPosChan) + 1);
            coords = channelPosCoords{targetChanIdx};
            pickIdx = randi(size(coords, 1));
            centerR = coords(pickIdx, 1);
            centerC = coords(pickIdx, 2);
        else
            % Random sampling across full image
            centerR = randi([1, H]);
            centerC = randi([1, W]);
        end
        
        startR = max(1, min(H - pH + 1, centerR - floor(pH / 2)));
        startC = max(1, min(W - pW + 1, centerC - floor(pW / 2)));
        endR = startR + pH - 1;
        endC = startC + pW - 1;
        
        patchImg = imgScaled(startR:endR, startC:endC, :);
        patchMsk = masksScaled(startR:endR, startC:endC, :);
        
        patchIdx = patchIdx + 1;
        imagesCell{patchIdx} = patchImg;
        masksCell{patchIdx} = patchMsk;
        validityCell{patchIdx} = validity;
        imageIDCell{patchIdx} = imgID;
        
        % Compute channel statistics
        posPixCounts = squeeze(sum(patchMsk, [1 2])); % [1x5]
        isPos = posPixCounts > 0;
        channelPosCounts = channelPosCounts + double(isPos');
        
        nPos = sum(isPos);
        if nPos >= 2
            multiPosCount = multiPosCount + 1;
        elseif nPos == 0
            pureBackgroundCount = pureBackgroundCount + 1;
        end
    end
end

channelPosPct = (channelPosCounts / totalPatches) * 100;

patches = struct(...
    'images', {imagesCell}, ...
    'masks', {masksCell}, ...
    'validity', {validityCell}, ...
    'imageIDs', {imageIDCell}, ...
    'patchSize', patchSize, ...
    'scaleFactor', scaleFactor, ...
    'totalPatches', totalPatches, ...
    'randomSeed', randomSeed, ...
    'channelPosCounts', channelPosCounts, ...
    'channelPosPct', channelPosPct, ...
    'multiPosCount', multiPosCount, ...
    'pureBackgroundCount', pureBackgroundCount);
end

function tf = matchesType(recordType, targetType)
rec = upper(string(recordType));
tgt = upper(string(targetType));
switch tgt
    case "MA"
        tf = contains(rec,"MA") || contains(rec,"MICROANEURYSM");
    case "HE"
        tf = contains(rec,"HE") || contains(rec,"HAEMORRHAGE");
    case "EX"
        tf = (contains(rec,"EX") && ~contains(rec,"SOFT")) || contains(rec,"HARD");
    case "SE"
        tf = contains(rec,"SE") || contains(rec,"SOFT");
    case "OD"
        tf = contains(rec,"OD") || contains(rec,"OPTIC");
    otherwise
        tf = (rec == tgt);
end
end

function mask = readValidatedBinaryMask(maskFile)
info = imfinfo(maskFile);
assert(numel(info) == 1, 'DrishtiMitra:IDRiDMaskPages', 'TIFF has %d pages: %s', numel(info), maskFile);
raw = imread(maskFile);
if ismatrix(raw)
    mask = raw ~= mode(raw(:));
    return
end
rgb = raw(:,:,1:3);
pixelRows = reshape(permute(rgb,[3 1 2]),3,[])';
[colours,~,colourIndex] = unique(pixelRows,'rows','stable');
counts = accumarray(colourIndex,1);
[~,backgroundIndex] = max(counts);
mask = reshape(colourIndex ~= backgroundIndex, size(raw,1), size(raw,2));
end

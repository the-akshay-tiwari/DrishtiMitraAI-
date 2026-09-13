function figureHandle = visualizeIDRiDMasks(imageID,datasetRoot,maxDisplayDimension)
%VISUALIZEIDRIDMASKS Inspect one validated official IDRiD image and its masks.
% imageID accepts IDRiD_1 through IDRiD_081. The source image and TIFF files
% are read only; display is downsampled only when needed for responsiveness.

arguments
    imageID (1,1) string
    datasetRoot (1,1) string = ""
    maxDisplayDimension (1,1) double {mustBePositive,mustBeFinite} = 1600
end

if strlength(datasetRoot) == 0
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    segmentationRoot = fullfile(projectRoot,'data','raw','IDRiD_Segmentation','A. Segmentation');
    archiveRoot = fullfile(projectRoot,'data','raw','archive');
    if isfolder(segmentationRoot)
        datasetRoot = string(segmentationRoot);
    else
        datasetRoot = string(archiveRoot);
    end
end

% Reuse the validator's actual package discovery and official image/mask map.
report = validateIDRiDMasks(datasetRoot);
id = normaliseID(imageID);
imageRows = report.images(report.images.ID == id,:);
assert(height(imageRows) == 1,'DrishtiMitra:IDRiDVisualizeImage', ...
    'Expected one validated original image for %s, found %d.',id,height(imageRows));

sourceImage = imread(imageRows.ImageFile(1));
sourceImage = rgbImage(sourceImage,imageRows.ImageFile(1));
types = ["MA","HE","EX","SE","OD"];
colours = [0.00 0.85 1.00; 1.00 0.10 0.10; 1.00 0.90 0.00; 1.00 0.10 0.85; 0.10 1.00 0.20];
masks = cell(numel(types),1);
present = false(numel(types),1);

for k = 1:numel(types)
    typeMatches = false(height(report.records),1);
    for r = 1:height(report.records)
        typeMatches(r) = matchesType(report.records.Type(r),types(k));
    end
    rows = report.records(report.records.ID == id & ...
        report.records.ImageSet == imageRows.ImageSet(1) & typeMatches,:);
    assert(height(rows) <= 1,'DrishtiMitra:IDRiDVisualizeDuplicate', ...
        'The validated mapping unexpectedly has multiple %s masks for %s.',types(k),id);
    if height(rows) == 1
        masks{k} = readValidatedBinaryMask(rows.MaskFile(1));
        assert(isequal(size(masks{k}),size(sourceImage,[1 2])), ...
            'DrishtiMitra:IDRiDVisualizeDimensions', ...
            'Validated %s mask does not match the source image dimensions for %s.',types(k),id);
        present(k) = true;
    end
end

[displayImage,displayMasks,wasResized] = prepareDisplay(sourceImage,masks,present,maxDisplayDimension);
figureHandle = figure('Name',sprintf('IDRiD masks: %s (%s)',id,imageRows.ImageSet(1)), ...
    'Color','w','NumberTitle','off');
layout = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
title(layout,sprintf('%s — %s set (source %d x %d; %s)', ...
    id,imageRows.ImageSet(1),size(sourceImage,1),size(sourceImage,2), ...
    ternary(wasResized,'display downsampled only','full-resolution display')));

nexttile; imshow(displayImage); title('Original fundus image');
for k = 1:numel(types)
    nexttile;
    if present(k)
        imshow(displayMasks{k},'InitialMagnification','fit');
        title(sprintf('%s binary mask',types(k)));
    else
        showUnavailableMask(types(k));
    end
end

nexttile;
imshow(displayImage); hold on;
for k = 1:numel(types)
    if present(k)
        colourLayer = uint8(255 * reshape(colours(k,:),1,1,3));
        colourLayer = repmat(colourLayer,size(displayImage,1),size(displayImage,2));
        h = image(colourLayer);
        alpha = 0.45;
        if types(k) == "OD", alpha = 0.28; end
        set(h,'AlphaData',alpha * double(displayMasks{k}));
    end
end
hold off; title('Overlay: MA cyan, HE red, EX yellow, SE magenta, OD green');

nexttile; axis off;
missing = types(~present);
if isempty(missing)
    status = 'All five annotation types are available.';
else
    status = sprintf('Not annotated / unavailable:\n%s',strjoin(missing,', '));
end
text(0.5,0.60,status,'HorizontalAlignment','center','FontWeight','bold');
text(0.5,0.32,sprintf('Mask display size: %d x %d\nBinary masks use nearest-neighbor resizing.', ...
    size(displayImage,1),size(displayImage,2)),'HorizontalAlignment','center');
end

function id = normaliseID(imageID)
token = regexp(char(imageID),'(?i)^(?:IDRiD_)?(\d{1,3})(?:test)?$','tokens','once');
assert(~isempty(token),'DrishtiMitra:IDRiDVisualizeID', ...
    'imageID must have the form IDRiD_# or # (for example, IDRiD_55 or 55).');
number = str2double(token{1});
assert(isfinite(number) && number >= 1 && number <= 999, ...
    'DrishtiMitra:IDRiDVisualizeID','imageID is outside the supported IDRiD range.');
id = string(sprintf('IDRiD_%03d',number));
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

function image = rgbImage(image,imageFile)
assert(isnumeric(image) || islogical(image),'DrishtiMitra:IDRiDVisualizeImageType', ...
    'Unsupported fundus image datatype in %s.',imageFile);
if ismatrix(image)
    image = repmat(image,1,1,3);
else
    assert(ndims(image) == 3 && size(image,3) == 3, ...
        'DrishtiMitra:IDRiDVisualizeImageChannels', ...
        'Expected a grayscale or RGB fundus image, got %s in %s.',mat2str(size(image)),imageFile);
end
end

function [imageOut,masksOut,wasResized] = prepareDisplay(image,masks,present,maxDimension)
scale = min(1,maxDimension / max(size(image,1),size(image,2)));
wasResized = scale < 1;
if wasResized
    imageOut = imresize(image,scale,'bilinear');
else
    imageOut = image;
end
masksOut = masks;
for k = 1:numel(masks)
    if present(k)
        if wasResized
            masksOut{k} = imresize(masks{k},size(imageOut,[1 2]),'nearest');
        else
            masksOut{k} = masks{k};
        end
        masksOut{k} = logical(masksOut{k});
    end
end
end

function showUnavailableMask(type)
axis off;
text(0.5,0.55,sprintf('%s\nnot annotated / unavailable',type), ...
    'HorizontalAlignment','center','FontWeight','bold');
title(sprintf('%s mask',type));
end

function [mask,details] = readValidatedBinaryMask(maskFile)
% Mirrors the validator's strict canonicalization: TIFF pages are rejected;
% RGB/RGBA is accepted only for a two-colour encoding with constant alpha.
info = imfinfo(maskFile);
assert(numel(info) == 1,'DrishtiMitra:IDRiDMaskPages', ...
    'TIFF has %d pages; page-to-mask interpretation is ambiguous: %s',numel(info),maskFile);
raw = imread(maskFile);
details = struct('SourceSize',string(mat2str(size(raw))),'SourceClass',string(class(raw))); %#ok<NASGU>

if ismatrix(raw)
    values = unique(raw(:));
    assert(numel(values) <= 2,'DrishtiMitra:IDRiDMaskValues', ...
        'Single-channel TIFF has %d values, not a binary mask: %s',numel(values),maskFile);
    mask = raw ~= mode(raw(:));
    return
end

assert(ndims(raw) == 3 && (size(raw,3) == 3 || size(raw,3) == 4), ...
    'DrishtiMitra:IDRiDMaskChannels', ...
    'Unsupported TIFF representation %s in %s.',mat2str(size(raw)),maskFile);
rgb = raw(:,:,1:3);
if size(raw,3) == 4
    alphaValues = unique(raw(:,:,4));
    assert(numel(alphaValues) == 1,'DrishtiMitra:IDRiDMaskAlpha', ...
        'RGBA TIFF has nonconstant alpha and is ambiguous: %s',maskFile);
end
pixelRows = reshape(permute(rgb,[3 1 2]),3,[])';
[colours,~,colourIndex] = unique(pixelRows,'rows','stable');
assert(size(colours,1) <= 2,'DrishtiMitra:IDRiDMaskColourEncoding', ...
    'RGB/RGBA TIFF has %d colours, not an unambiguous binary encoding: %s',size(colours,1),maskFile);
counts = accumarray(colourIndex,1);
[~,backgroundIndex] = max(counts);
mask = reshape(colourIndex ~= backgroundIndex,size(raw,1),size(raw,2));
end

function value = ternary(condition,trueValue,falseValue)
if condition
    value = trueValue;
else
    value = falseValue;
end
end

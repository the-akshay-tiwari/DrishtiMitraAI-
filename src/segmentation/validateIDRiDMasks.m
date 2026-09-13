function report = validateIDRiDMasks(datasetRoot)
%VALIDATEIDRIDMASKS Validate the discovered IDRiD pixel-level annotation package.
% The function discovers TIFFs and JPG originals recursively; it does not rely
% on internal annotation folder names. It derives IDs/types from actual names.

if nargin < 1 || isempty(datasetRoot)
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    segmentationRoot = fullfile(projectRoot,'data','raw','IDRiD_Segmentation','A. Segmentation');
    archiveRoot = fullfile(projectRoot,'data','raw','archive');
    if isfolder(segmentationRoot)
        datasetRoot = segmentationRoot;
    else
        datasetRoot = archiveRoot;
    end
end
assert(isfolder(datasetRoot),'DrishtiMitra:IDRiDPackageMissing', ...
    'IDRiD segmentation package folder does not exist: %s',datasetRoot);

images = discoverImages(datasetRoot);
assert(~isempty(images),'DrishtiMitra:IDRiDOriginalImagesMissing', ...
    'No original images named IDRiD_#.jpg were found below %s.',datasetRoot);
imageKey = images.ImageSet + "|" + images.ID;
assert(numel(unique(imageKey)) == height(images),'DrishtiMitra:IDRiDDuplicateImage', ...
    'Duplicate original-image mappings were found for the same image set and ID.');

maskFiles = [dir(fullfile(datasetRoot,'**','*.tif')); dir(fullfile(datasetRoot,'**','*.tiff'))];
assert(~isempty(maskFiles),'DrishtiMitra:IDRiDMasksMissing', ...
    'No .tif/.tiff annotation files were found below %s.',datasetRoot);
records = parseMasks(string(fullfile({maskFiles.folder},{maskFiles.name}))',datasetRoot);
maskKey = records.ImageSet + "|" + records.ID;
assert(all(ismember(maskKey,imageKey)),'DrishtiMitra:IDRiDMaskImageMissing', ...
    'At least one mask has no matching original image in the same discovered data split.');

mappingKey = maskKey + "|" + records.Type;
assert(numel(unique(mappingKey)) == height(records),'DrishtiMitra:IDRiDDuplicateMask', ...
    'Duplicate mask mappings were found for the same image set, ID, and annotation type.');
assert(isempty(setdiff(imageKey,unique(maskKey))),'DrishtiMitra:IDRiDMaskMissing', ...
    'At least one discovered original image has no annotation mask at all.');

records = validatePixelsAndDimensions(records,images);
[coverage,missingMaskPairs] = summarizeCoverage(records,images);
structure = inferTargetStructure(records);

folders = unique(records.RelativeFolder,'stable');
patterns = unique(records.FilenamePattern,'stable');
types = unique(records.Type,'stable');
annotatedIDs = unique(maskKey);
report = struct('datasetRoot',string(datasetRoot),'annotationFolders',folders, ...
    'filenamePatterns',patterns,'annotationTypes',types,'images',images, ...
    'records',records,'uniqueAnnotatedImages',numel(annotatedIDs), ...
    'coverage',coverage,'missingMaskPairs',missingMaskPairs,'targetStructure',structure);

fprintf('\nIDRiD pixel-level mask validation\n');
fprintf('Package root: %s\n',datasetRoot);
fprintf('Original images: %d (%d training, %d testing, %d unspecified)\n',height(images), ...
    nnz(images.ImageSet == "training"),nnz(images.ImageSet == "testing"),nnz(images.ImageSet == "unspecified"));
fprintf('Annotation TIFF files: %d\n',height(records));
fprintf('Annotation folders (%d): %s\n',numel(folders),strjoin(folders,', '));
fprintf('Filename patterns (%d): %s\n',numel(patterns),strjoin(patterns,', '));
fprintf('Annotation types (%d): %s\n',numel(types),strjoin(types,', '));
fprintf('Unique annotated original images: %d\n',numel(annotatedIDs));
fprintf('Dimensions: every mapped mask matches its source image.\n');
fprintf('Per-lesion coverage:\n');
for k = 1:height(coverage)
    fprintf('  %s / %s: %d of %d images (%d type-specific files absent)\n', ...
        coverage.ImageSet(k),coverage.Type(k),coverage.MaskCount(k), ...
        coverage.ImageCount(k),coverage.MissingCount(k));
end
for k = 1:numel(types)
    subset = records(records.Type == types(k),:);
    values = unique([subset.UniqueValues{:}]);
    fprintf('  %s: %d masks, imread datatype %s, canonical values [%s]\n',types(k),height(subset), ...
        strjoin(unique(subset.MaskClass),', '),strjoin(string(values),', '));
    for representation = unique(subset.Representation,'stable')'
        fprintf('    Representation: %s\n',representation);
    end
end
if isempty(missingMaskPairs)
    fprintf('Missing type-specific mask files: none.\n');
else
    fprintf('Missing type-specific mask files: %d (reported as sparse coverage; see report.missingMaskPairs).\n',height(missingMaskPairs));
end
fprintf('Target structure: %s\n\n',structure);
end

function images = discoverImages(datasetRoot)
files = [dir(fullfile(datasetRoot,'**','*.jpg')); dir(fullfile(datasetRoot,'**','*.jpeg'))];
paths = strings(0,1); IDs = strings(0,1); imageSets = strings(0,1);
for k = 1:numel(files)
    [~,name] = fileparts(files(k).name);
    token = regexp(name,'(?i)^IDRiD_(\d{1,3})(test)?$','tokens','once');
    if isempty(token), continue, end
    paths(end+1,1) = string(fullfile(files(k).folder,files(k).name)); %#ok<AGROW>
    IDs(end+1,1) = sprintf('IDRiD_%03d',str2double(token{1})); %#ok<AGROW>
    imageSets(end+1,1) = inferSplit(string(files(k).folder),~isempty(token{2})); %#ok<AGROW>
end
images = table(paths,IDs,imageSets,'VariableNames',{'ImageFile','ID','ImageSet'});
end

function records = parseMasks(maskPaths,datasetRoot)
n = numel(maskPaths);
ID = strings(n,1); ImageSet = strings(n,1); Type = strings(n,1);
RelativeFolder = strings(n,1); FilenamePattern = strings(n,1);
for k = 1:n
    [folder,name,ext] = fileparts(maskPaths(k));
    token = regexp(name,'(?i)^IDRiD_(\d{1,3})(test)?(?:[_.-]+(.*))?$','tokens','once');
    assert(~isempty(token),'DrishtiMitra:IDRiDMaskName', ...
        'Cannot derive an IDRiD image ID from annotation filename: %s',maskPaths(k));
    ID(k) = sprintf('IDRiD_%03d',str2double(token{1}));
    ImageSet(k) = inferSplit(string(folder),~isempty(token{2}));
    annotationType = "";
    if numel(token) >= 3 && ~isempty(token{3}), annotationType = string(token{3}); end
    relativeFolder = strip(replace(erase(string(folder),string(datasetRoot)),char(92),'/'),'/');
    if strlength(annotationType) == 0
        folderParts = split(relativeFolder,'/');
        isGenericFolder = any(contains(lower(folderParts), ...
            ["training","testing","test","mask","annotation","groundtruth"]),2);
        folderParts = folderParts(folderParts ~= "" & ~isGenericFolder);
        assert(~isempty(folderParts),'DrishtiMitra:IDRiDMaskType', ...
            'Cannot derive an annotation type from filename or folder: %s',maskPaths(k));
        annotationType = folderParts(end);
    end
    Type(k) = upper(regexprep(annotationType,'\s+','_'));
    RelativeFolder(k) = relativeFolder;
    FilenamePattern(k) = regexprep(string(name) + string(ext),'(?i)IDRiD_\d{1,3}(test)?','IDRiD_###$1');
end
records = table(maskPaths,ID,ImageSet,Type,RelativeFolder,FilenamePattern, ...
    'VariableNames',{'MaskFile','ID','ImageSet','Type','RelativeFolder','FilenamePattern'});
end

function imageSet = inferSplit(folder,isExplicitTest)
folder = lower(folder);
if contains(folder,"training")
    imageSet = "training";
elseif contains(folder,"testing")
    imageSet = "testing";
elseif isExplicitTest
    imageSet = "testing";
else
    imageSet = "unspecified";
end
end

function records = validatePixelsAndDimensions(records,images)
n = height(records); MaskSize = strings(n,1); MaskClass = strings(n,1); UniqueValues = cell(n,1); Representation = strings(n,1);
imageKey = images.ImageSet + "|" + images.ID;
for k = 1:n
    key = records.ImageSet(k) + "|" + records.ID(k);
    imagePath = images.ImageFile(find(imageKey == key,1));
    imageInfo = imfinfo(imagePath);
    [mask,maskInfo] = readCanonicalMask(records.MaskFile(k));
    assert(size(mask,1) == imageInfo.Height && size(mask,2) == imageInfo.Width, ...
        'DrishtiMitra:IDRiDMaskDimensions', ...
        'Dimension mismatch: %s is %s but its image is [%d %d].',records.MaskFile(k), ...
        mat2str(size(mask)),imageInfo.Height,imageInfo.Width);
    MaskSize(k) = maskInfo.SourceSize;
    MaskClass(k) = maskInfo.SourceClass;
    UniqueValues{k} = unique(mask(:))';
    Representation(k) = maskInfo.Representation;
end
records.MaskSize = MaskSize; records.MaskClass = MaskClass; records.UniqueValues = UniqueValues; records.Representation = Representation;
end

function [coverage,missingPairs] = summarizeCoverage(records,images)
types = unique(records.Type,'stable');
imageSets = unique(images.ImageSet,'stable');
coverageType = strings(0,1); coverageSet = strings(0,1); coverageMaskCount = zeros(0,1); coverageImageCount = zeros(0,1);
missingKeys = strings(0,1); missingTypes = strings(0,1);
for s = 1:numel(imageSets)
    imageKey = images.ID(images.ImageSet == imageSets(s));
    for k = 1:numel(types)
        present = unique(records.ID(records.ImageSet == imageSets(s) & records.Type == types(k)));
        absent = setdiff(imageKey,present);
        coverageType(end+1,1) = types(k); %#ok<AGROW>
        coverageSet(end+1,1) = imageSets(s); %#ok<AGROW>
        coverageMaskCount(end+1,1) = numel(present); %#ok<AGROW>
        coverageImageCount(end+1,1) = numel(imageKey); %#ok<AGROW>
        missingKeys = [missingKeys; imageSets(s) + "|" + absent]; %#ok<AGROW>
        missingTypes = [missingTypes; repmat(types(k),numel(absent),1)]; %#ok<AGROW>
    end
end
coverage = table(coverageSet,coverageType,coverageMaskCount,coverageImageCount, ...
    'VariableNames',{'ImageSet','Type','MaskCount','ImageCount'});
coverage.MissingCount = coverage.ImageCount - coverage.MaskCount;
missingPairs = table(missingKeys,missingTypes,'VariableNames',{'ImageKey','Type'});
end

function structure = inferTargetStructure(records)
allBinary = all(cellfun(@numel,records.UniqueValues) <= 2);
types = unique(records.Type);
if ~allBinary && numel(types) == 1
    structure = "single multiclass mask";
    return
end
assert(allBinary,'DrishtiMitra:IDRiDMaskStructure', ...
    'Mixed binary/nonbinary mask values prevent an unambiguous segmentation target.');
if numel(types) == 1
    structure = "single binary mask";
    return
end
hasOverlap = false;
for imageSet = unique(records.ImageSet)'
    ids = unique(records.ID(records.ImageSet == imageSet));
    for id = ids'
        rows = records(records.ImageSet == imageSet & records.ID == id,:);
        foreground = cell(height(rows),1);
        for k = 1:height(rows)
            M = readCanonicalMask(rows.MaskFile(k)); foreground{k} = M;
        end
        for a = 1:numel(foreground)-1
            for b = a+1:numel(foreground)
                hasOverlap = hasOverlap || any(foreground{a} & foreground{b},'all');
            end
        end
    end
end
if hasOverlap
    structure = "separate binary masks with overlapping multilabel foreground";
else
    structure = "separate binary masks with mutually exclusive observed foreground";
end
end

function [mask,details] = readCanonicalMask(maskFile)
% Read official TIFF encodings without changing source data. A 2-D binary
% raster is accepted directly. RGB/RGBA is accepted only when inspection
% proves a two-colour binary encoding with a constant alpha channel.
info = imfinfo(maskFile);
assert(numel(info) == 1,'DrishtiMitra:IDRiDMaskPages', ...
    'TIFF has %d pages; page-to-mask interpretation is ambiguous: %s',numel(info),maskFile);
raw = imread(maskFile);
details = struct('SourceSize',string(mat2str(size(raw))), ...
    'SourceClass',string(class(raw)),'Representation',"");

if ismatrix(raw)
    mask = logicalFromBinaryValues(raw,maskFile);
    details.Representation = sprintf('%s TIFF; imread %s %s', ...
        describeTiff(info),class(raw),mat2str(size(raw)));
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
    alphaDescription = sprintf(', constant alpha=%g',double(alphaValues));
else
    alphaDescription = '';
end

pixelRows = reshape(permute(rgb,[3 1 2]),3,[])';
[colours,~,colourIndex] = unique(pixelRows,'rows','stable');
assert(size(colours,1) <= 2,'DrishtiMitra:IDRiDMaskColourEncoding', ...
    'RGB/RGBA TIFF has %d colours, not an unambiguous binary encoding: %s',size(colours,1),maskFile);
counts = accumarray(colourIndex,1);
[~,backgroundIndex] = max(counts);
mask = reshape(colourIndex ~= backgroundIndex,size(raw,1),size(raw,2));
colourText = strjoin(compose('[%g %g %g]',double(colours(:,1)),double(colours(:,2)),double(colours(:,3))),', ');
details.Representation = sprintf('%s TIFF; imread %s %s; %d-colour RGB encoding {%s}%s; modal colour is background', ...
    describeTiff(info),class(raw),mat2str(size(raw)),size(colours,1),colourText,alphaDescription);
end

function mask = logicalFromBinaryValues(raw,maskFile)
values = unique(raw(:));
assert(numel(values) <= 2,'DrishtiMitra:IDRiDMaskValues', ...
    'Single-channel TIFF has %d values, not a binary mask: %s',numel(values),maskFile);
background = mode(raw(:));
mask = raw ~= background;
end

function description = describeTiff(info)
compression = "unknown"; colourType = "unknown"; bitDepth = NaN;
if isfield(info,'Compression'), compression = string(info.Compression); end
if isfield(info,'ColorType'), colourType = string(info.ColorType); end
if isfield(info,'BitDepth'), bitDepth = double(info.BitDepth); end
description = sprintf('TIFF pages=%d ColorType=%s BitDepth=%g Compression=%s', ...
    numel(info),colourType,bitDepth,compression);
end

function J = enhanceImageV2(I, opts)
arguments
    I {mustBeNumeric}
    opts.ClipLimit (1,1) double = 0.01
end

I = im2uint8(I);

if ndims(I) == 2
    I = cat(3,I,I,I);
end

lab = rgb2lab(I);
L = lab(:,:,1) / 100;

% Local contrast enhancement only — preserve retinal structures.
L = adapthisteq(L, 'ClipLimit', opts.ClipLimit, ...
    'NumTiles', [8 8]);

lab(:,:,1) = 100 * L;
J = lab2rgb(lab, 'OutputType', 'uint8');
end

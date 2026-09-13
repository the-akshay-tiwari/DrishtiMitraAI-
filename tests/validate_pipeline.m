%% VALIDATE_PIPELINE Synthetic smoke checks only — never a clinical benchmark.
% This generated image confirms interfaces, rejection boundaries, and report output.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(fullfile(root,'src')));
n=512; [X,Y]=meshgrid(1:n,1:n); field=(X-n/2).^2+(Y-n/2).^2<(n*.43)^2;
I=zeros(n,n,3,'uint8'); I(:,:,1)=uint8(35+80*field); I(:,:,2)=uint8(25+110*field); I(:,:,3)=uint8(20+55*field);
% Synthetic vessel-like lines and bright optic disc; not an anatomical phantom.
I(:,:,2)=max(I(:,:,2),uint8(190*(((Y-n/2)-.25*(X-n/2)).^2<4 & field)));
disc=(X-n*.36).^2+(Y-n*.47).^2<28^2; I(:,:,1)=I(:,:,1)+uint8(45*disc); I(:,:,2)=I(:,:,2)+uint8(45*disc);
q=assessQuality(I); assert(isfield(q,'gradeabilityScore'));
J=enhanceImage(I); assert(isequal(size(J),size(I)));
v=segmentVessels(J,'ClassicalOnly',true); assert(isequal(size(v.mask),[n n]));
d=localizeOpticDiscFovea(J); assert(isfield(d,'foveaCenter'));
ma=detectMicroaneurysms(J); ex=segmentExudates(J,d); he=classifyHemorrhages(J,d); nv=detectNeovascularization(J,d);
r=gradeImage(I); assert(isfield(r,'status'));
if strcmp(r.status,'graded')
    out=fullfile(tempdir,'drishtimitra_smoke_report.html'); generateClinicalReport(r,struct('id','SYNTHETIC-SMOKE'),out); assert(isfile(out));
end
% Evaluation utilities only validate arithmetic here; values are not published metrics.
m=computeSensSpec([.1;.2;.8;.9],[false;false;true;true]); assert(m.auc==1);
assert(abs(computeQWK([0;1;2],[0;1;2])-1)<eps);
assert(isstruct(ma) && isstruct(ex) && isstruct(he) && isstruct(nv));
disp('PASS: synthetic smoke test completed. No real-dataset metrics were produced.');

function result = gradeImage(I, classifier, calibration, threshold)
%GRADEIMAGE Fuse classifier result with independently measured lesion evidence.
if nargin<2, classifier=[]; end; if nargin<3, calibration=[]; end; if nargin<4, threshold=.5; end
q=assessQuality(I); if ~q.isGradeable, result=struct('status','recapture','quality',q); return; end
J=enhanceImage(I); disc=localizeOpticDiscFovea(J); ma=detectMicroaneurysms(J); ex=segmentExudates(J,disc); he=classifyHemorrhages(J,disc); nv=detectNeovascularization(J,disc);
deepScore=NaN; deepGrade=NaN;
if ~isempty(classifier)
    try
        [~,score]=classify(classifier,J); deepScore=max(score(2:end)); deepGrade=find(score==max(score),1)-1;
    catch, warning('DrishtiMitra:ClassifierUnavailable','Classifier could not infer this image.'); end
end
lesionScore=min(1,.02*ma.count + .00001*ex.areaPixels + .12*he.affectedQuadrants + .35*(nv.nvdFlag||nv.nveFlag));
if isnan(deepScore), fused=lesionScore; else, fused=.75*deepScore+.25*lesionScore; end
if ~isempty(calibration), confidence=calibration.predict(fused); else, confidence=fused; end
grade=max([double(ma.count>0), 2*double(ex.areaPixels>100 || he.affectedQuadrants>=2), 4*double(nv.nvdFlag||nv.nveFlag)]);
if ~isnan(deepGrade), grade=max(grade,deepGrade); end
result=struct('status','graded','quality',q,'enhancedImage',J,'opticDisc',disc,'microaneurysms',ma, ...
 'exudates',ex,'hemorrhages',he,'neovascularization',nv,'deepScore',deepScore, ...
 'lesionScore',lesionScore,'fusedScore',fused,'calibratedConfidence',confidence, ...
 'grade',grade,'referableDR',fused>=threshold || grade>=2,'threshold',threshold);
end

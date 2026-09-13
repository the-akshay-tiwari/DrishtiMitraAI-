function capacity = ProcessingThroughput(secondsPerImage, parallelWorkers, operatingHours)
%PROCESSINGTHROUGHPUT Convert measured MATLAB inference latency to capacity.
assert(secondsPerImage>0,'DrishtiMitra:Latency','Use a measured positive MATLAB latency.');
capacity=parallelWorkers*operatingHours*3600/secondsPerImage;
end

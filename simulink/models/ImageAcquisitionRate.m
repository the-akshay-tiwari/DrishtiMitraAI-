function rate = ImageAcquisitionRate(choCenters, stationsPerCenter, imagesPerStationHour, hoursPerDay, workDays)
%IMAGEACQUISITIONRATE Annual fundus-image acquisition capacity.
rate=choCenters*stationsPerCenter*imagesPerStationHour*hoursPerDay*workDays;
end

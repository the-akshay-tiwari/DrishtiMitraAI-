function hours = BandwidthConstraint(images, imageMB, uplinkMbps, storeForwardFraction)
%BANDWIDTHCONSTRAINT Transfer hours; store-forward data does not occupy live uplink.
hours=(images*imageMB*8*(1-storeForwardFraction))/max(eps,uplinkMbps)/3600;
end

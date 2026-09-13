function capacity = ReviewCapacity(specialists, secondsPerReview, hoursPerDay, workDays)
%REVIEWCAPACITY Annual ophthalmologist review capacity.
capacity=specialists*hoursPerDay*3600*workDays/secondsPerReview;
end

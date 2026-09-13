%% TESTSMOKEATTENTIONUNET Smoke check for Attention U-Net pipeline
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));

fprintf('Running Attention U-Net Smoke Test Unit Check...\n');
result = runIDRiDSmokeTest(4);

assert(strcmp(result.status, "SUCCESS"), 'Smoke test status must be SUCCESS');
assert(isfinite(result.loss), 'Smoke test loss must be finite');
assert(result.loss > 0, 'Smoke test loss must be positive');

fprintf('PASS: Attention U-Net smoke check passed cleanly.\n');

% BUILDSCREENINGWORKFLOW Generate the documented discrete workflow in Simulink.
% Run this file from MATLAB/Simulink; it intentionally does not manufacture an SLX without Simulink.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(fullfile(root,'models')));
scenarioFile=fullfile(root,'scenarios','districtLevel_100k.mat');
if ~isfile(scenarioFile), createDistrictLevelScenario(scenarioFile); end
assert(exist('simulink','file')==4,'DrishtiMitra:SimulinkMissing','Simulink is required to create ScreeningWorkflow.slx.');
model='ScreeningWorkflow'; if bdIsLoaded(model), close_system(model,0); end
new_system(model); open_system(model);
add_block('simulink/Sources/Constant',[model '/Acquisition rate'],'Value','50*1*5');
add_block('simulink/Math Operations/Gain',[model '/Bandwidth service'],'Gain','5');
add_block('simulink/Math Operations/Gain',[model '/Processing service'],'Gain','1');
add_block('simulink/Math Operations/Gain',[model '/Review capacity'],'Gain','1');
add_block('simulink/Sinks/Scope',[model '/Workflow capacity']);
add_line(model,'Acquisition rate/1','Bandwidth service/1');
add_line(model,'Bandwidth service/1','Processing service/1');
add_line(model,'Processing service/1','Review capacity/1');
add_line(model,'Review capacity/1','Workflow capacity/1');
save_system(model,fullfile(root,'ScreeningWorkflow.slx')); close_system(model);

function outputFile = generateClinicalReport(result, patient, outputFile)
%GENERATECLINICALREPORT Create an HTML decision-support report, printable to PDF.
arguments; result struct; patient struct; outputFile string = "clinical_report.html"; end
fid=fopen(outputFile,'w'); assert(fid>0,'DrishtiMitra:ReportWrite','Cannot write report.'); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'<html><body><h1>DrishtiMitra screening decision-support report</h1>');
if isfield(patient,'id'), patientId=patient.id; else, patientId='Not supplied'; end
fprintf(fid,'<p>Patient: %s</p>',patientId);
if isfield(result,'status') && strcmp(result.status,'recapture'), fprintf(fid,'<h2>Recapture required</h2><p>%s</p>',strjoin(result.quality.recaptureReasons,' ')); return, end
fprintf(fid,'<p>ICDR grade: %d | Calibrated confidence: %.1f%% | Referral: %s</p>',result.grade,100*result.calibratedConfidence,string(result.referableDR));
fprintf(fid,'<table border="1"><tr><th>Evidence</th><th>Value</th></tr><tr><td>Microaneurysms</td><td>%d</td></tr><tr><td>Exudate area (px)</td><td>%d</td></tr><tr><td>Hemorrhage quadrants</td><td>%d</td></tr><tr><td>NVD/NVE flags</td><td>%d / %d</td></tr></table>',result.microaneurysms.count,result.exudates.areaPixels,result.hemorrhages.affectedQuadrants,result.neovascularization.nvdFlag,result.neovascularization.nveFlag);
fprintf(fid,'<p><strong>Safety:</strong> This is screening decision support. Lesion segmentations are supporting evidence, not standalone diagnostic ground truth. A qualified clinician must review every referral and make the final diagnosis.</p></body></html>');
end

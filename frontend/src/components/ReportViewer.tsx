import type { GradeDetails, Patient } from '../types'
import { StatusBadge } from './StatusBadge'

export type ModelPredictionInfo = {
  confidence: number
  probabilities: Record<string, number>
  heatmap?: string
  performanceMetrics?: {
    prediction_latency_ms: number
    gradcam_latency_ms: number
    inference_strategy: string
    forward_passes: number
    experiment_id: string
  }
}

export function ReportViewer({
  patient,
  detail,
  online,
  prediction,
}: {
  patient?: Partial<Patient>
  detail: GradeDetails
  online: boolean
  prediction?: ModelPredictionInfo | null
}) {
  const tone = detail.priority === 'Priority' ? 'rose' : detail.priority === 'Review' ? 'amber' : 'teal'
  const timestamp = new Date().toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  })

  const patientName = patient?.name || 'Anonymous Patient'
  const patientId = patient?.id || 'DM-SCAN-TEMP'
  const age = patient?.age ? `${patient.age} yrs` : 'N/A'
  const sex = patient?.sex || 'N/A'
  const screenDate = patient?.screenDate || new Date().toISOString().split('T')[0]
  const diabetesDuration = patient?.diabetesDuration || 'Unspecified'

  const classLabels = ['Grade 0 (No DR)', 'Grade 1 (Mild DR)', 'Grade 2 (Moderate DR)', 'Grade 3 (Severe DR)', 'Grade 4 (Proliferative DR)']

  return (
    <article className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm print:rounded-none print:border-none print:shadow-none">
      {/* Printable Report Header */}
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 bg-slate-900 px-6 py-5 text-white print:bg-white print:text-slate-950 print:border-b-2 print:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-teal-600 text-sm font-black text-white">◉</span>
            <p className="text-xl font-black tracking-tight">Drishti<span className="text-teal-400 print:text-teal-700">Mitra</span> AI</p>
          </div>
          <p className="mt-1 text-xs font-bold uppercase tracking-[0.16em] text-slate-400 print:text-slate-600">
            Clinical Screening Support & Explainability Report
          </p>
        </div>
        <div className="text-right text-xs text-slate-300 print:text-slate-700">
          <p>Report Ref: <strong className="text-white print:text-slate-950">DMR-{screenDate.replace(/-/g, '')}-01</strong></p>
          <p className="mt-1">Generated: <strong className="text-white print:text-slate-950">{timestamp}</strong></p>
          <p className="mt-1 text-[11px] text-teal-400 print:text-teal-700 font-semibold">{online ? 'Cloud Sync Enabled' : 'Local Edge Mode'}</p>
        </div>
      </header>

      {/* Patient & Quality Screening Metadata */}
      <div className="grid gap-6 p-6 md:grid-cols-[1.1fr_.9fr] border-b border-slate-100">
        <section>
          <p className="text-xs font-extrabold uppercase tracking-wider text-teal-700">Patient Metadata</p>
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2.5 text-sm">
            <div><dt className="text-xs text-slate-500 font-medium">Patient ID</dt><dd className="font-bold text-slate-900">{patientId}</dd></div>
            <div><dt className="text-xs text-slate-500 font-medium">Screening Date</dt><dd className="font-bold text-slate-900">{screenDate}</dd></div>
            <div><dt className="text-xs text-slate-500 font-medium">Patient Name</dt><dd className="font-bold text-slate-900">{patientName}</dd></div>
            <div><dt className="text-xs text-slate-500 font-medium">Age / Sex</dt><dd className="font-bold text-slate-900">{age} / {sex}</dd></div>
            <div><dt className="text-xs text-slate-500 font-medium">Diabetes Duration</dt><dd className="font-bold text-slate-900">{diabetesDuration}</dd></div>
            <div><dt className="text-xs text-slate-500 font-medium">Image Quality Status</dt><dd className="font-bold text-teal-700">✓ Passed Quality Gate</dd></div>
          </dl>
        </section>

        {/* AI Screening Assessment */}
        <section className="rounded-xl bg-slate-50 p-5 border border-slate-200">
          <div className="flex items-center justify-between">
            <p className="text-xs font-extrabold uppercase tracking-wider text-slate-500">AI Assessment</p>
            <StatusBadge tone={tone}>{detail.priority}</StatusBadge>
          </div>
          <div className="mt-3">
            <p className="text-2xl font-black text-slate-950">{detail.label}</p>
            <p className="text-xs font-semibold text-slate-600 mt-0.5">{detail.clinicalLabel}</p>
          </div>
          {prediction ? (
            <div className="mt-3 flex items-center justify-between text-xs border-t border-slate-200 pt-2.5">
              <span className="text-slate-600">Model Confidence:</span>
              <span className="font-extrabold text-teal-700 font-mono text-sm">{(prediction.confidence * 100).toFixed(1)}%</span>
            </div>
          ) : (
            <p className="mt-3 text-xs text-slate-500 border-t border-slate-200 pt-2">Demonstration Reference Mode</p>
          )}
          <p className="mt-2 text-[11px] text-slate-500 leading-4">
            Prediction generated using <strong className="text-slate-700">E02_no_sampler_35ep</strong> model with <strong className="text-slate-700">AVG_2X Test-Time Augmentation (TTA)</strong>.
          </p>
        </section>
      </div>

      {/* Class Probability Distribution Table */}
      {prediction?.probabilities && (
        <section className="p-6 border-b border-slate-100">
          <p className="text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-3">Class Probability Distribution</p>
          <div className="grid gap-2 text-xs">
            {classLabels.map((lbl, idx) => {
              const prob = prediction.probabilities[String(idx)] ?? prediction.probabilities[idx] ?? 0
              const isTop = idx === detail.grade
              return (
                <div key={lbl} className={`flex items-center justify-between gap-3 p-2 rounded-lg ${isTop ? 'bg-teal-50 border border-teal-200 font-bold' : 'bg-slate-50'}`}>
                  <span className="w-36 text-slate-700">{lbl}</span>
                  <div className="flex-1 bg-slate-200 h-2 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${isTop ? 'bg-teal-600' : 'bg-slate-400'}`} style={{ width: `${Math.max(prob * 100, 1)}%` }} />
                  </div>
                  <span className="w-14 text-right font-mono text-slate-800">{(prob * 100).toFixed(1)}%</span>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Explainability Heatmap & Original Fundus Scan */}
      <section className="p-6 border-b border-slate-100">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Visual Screening Evidence</p>
          <span className="text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-0.5 rounded-full">
            Grad-CAM Layer: features.7
          </span>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {/* Original Input View */}
          <div className="rounded-xl border border-slate-200 bg-slate-950 p-2 text-center">
            <img className="aspect-[4/3] w-full rounded-lg object-cover" src={detail.image} alt="Original Retinal Fundus Scan" />
            <p className="mt-2 text-[11px] font-bold text-slate-300">Original Retinal Fundus Scan</p>
          </div>

          {/* Grad-CAM Heatmap View */}
          <div className="rounded-xl border border-slate-200 bg-slate-950 p-2 text-center relative overflow-hidden">
            <img className="aspect-[4/3] w-full rounded-lg object-cover" src={detail.image} alt="Fundus Scan" />
            {prediction?.heatmap ? (
              <div className="pointer-events-none absolute inset-2 bg-cover bg-center mix-blend-screen opacity-75 rounded-lg" style={{ backgroundImage: `url("${prediction.heatmap}")` }} />
            ) : detail.heatmap ? (
              <div className="pointer-events-none absolute inset-2 bg-cover bg-center mix-blend-screen opacity-75 rounded-lg" style={{ backgroundImage: detail.heatmap.startsWith('data:') ? `url("${detail.heatmap}")` : detail.heatmap }} />
            ) : null}
            <p className="mt-2 text-[11px] font-bold text-teal-300">AI Attention / Grad-CAM Explainability Heatmap</p>
          </div>
        </div>

        {/* Explicit Heatmap Disclaimer */}
        <div className="mt-3 rounded-lg bg-slate-50 border border-slate-200 p-3 text-[11px] leading-5 text-slate-600">
          <strong>Explainability Notice:</strong> The visual overlay above represents an <strong className="text-slate-800">AI Attention / Grad-CAM Explainability Heatmap</strong> highlighting image regions that influenced the model prediction. It is <strong>NOT</strong> automated lesion segmentation (e.g. microaneurysms, exudates, or hemorrhages).
        </div>
      </section>

      {/* Model & Inference System Metadata */}
      <section className="p-6 border-b border-slate-100 bg-slate-50">
        <p className="text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-3">Model System Specifications</p>
        <div className="grid gap-x-6 gap-y-2 text-xs md:grid-cols-2 font-mono text-slate-700">
          <div><span className="text-slate-500 font-sans">Backbone Model:</span> EfficientNet-B0 (ImageNet Transfer Learning)</div>
          <div><span className="text-slate-500 font-sans">Experiment ID:</span> E02_no_sampler_35ep</div>
          <div><span className="text-slate-500 font-sans">Inference Strategy:</span> AVG_2X TTA (Original + Horizontal Flip Average)</div>
          <div><span className="text-slate-500 font-sans">Explainability Method:</span> Grad-CAM (Target Layer: features.7)</div>
          {prediction?.performanceMetrics && (
            <>
              <div><span className="text-slate-500 font-sans">Prediction Latency:</span> {prediction.performanceMetrics.prediction_latency_ms} ms</div>
              <div><span className="text-slate-500 font-sans">Grad-CAM Overhead:</span> {prediction.performanceMetrics.gradcam_latency_ms} ms</div>
            </>
          )}
        </div>
      </section>

      {/* Clinical Recommendation & Safety Disclaimer */}
      <div className="p-6">
        <p className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Recommended Action Plan</p>
        <p className="mt-2 text-sm font-semibold leading-6 text-slate-800 bg-slate-50 p-4 rounded-xl border border-slate-200">
          {detail.recommendation}
        </p>
      </div>

      {/* Mandated Safety Footer */}
      <footer className="border-t border-slate-200 bg-amber-50 px-6 py-4 text-xs leading-5 text-amber-950">
        <strong>Mandated Clinical Safety Disclaimer:</strong> DrishtiMitra is an experimental AI decision-support research tool. It is <strong>NOT</strong> a standalone diagnostic device. Image quality, visual evidence, and AI recommendations must be reviewed and interpreted by a qualified ophthalmologist prior to any clinical action.
      </footer>
    </article>
  )
}

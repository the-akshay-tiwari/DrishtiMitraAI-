import type { GradeDetails } from '../types'
import { StatusBadge } from './StatusBadge'

export function DRGradeCard({ detail, prominent = false, prediction }: { detail: GradeDetails; prominent?: boolean; prediction?: number }) {
  const tone = detail.priority === 'Priority' ? 'rose' : detail.priority === 'Review' ? 'amber' : 'teal'
  return (
    <div className={`rounded-2xl border bg-white ${prominent ? 'border-teal-200 p-6 shadow-[0_18px_50px_-28px_rgba(13,148,136,.7)]' : 'border-slate-200 p-5 shadow-sm'}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">AI Screening Result</p>
          <div className="mt-2 flex items-end gap-3"><span className="text-4xl font-black tracking-tight text-slate-950">{detail.label}</span><span className="mb-1 text-sm font-semibold text-slate-500">{detail.clinicalLabel}</span></div>
        </div>
        <StatusBadge tone={tone}>{detail.priority} {detail.priority === 'Priority' ? 'referral' : detail.priority === 'Review' ? 'review' : 'follow-up'}</StatusBadge>
      </div>
      <div className="mt-5 rounded-xl bg-slate-50 p-4">
        <p className="text-xs font-bold uppercase tracking-[0.12em] text-slate-500">Screening recommendation</p>
        <p className="mt-1.5 text-sm font-semibold leading-6 text-slate-800">{detail.recommendation}</p>
      </div>
      <p className="mt-4 text-xs leading-5 text-slate-500">{prediction === undefined ? 'Prototype demonstration — deterministic result mapped to the selected supplied IDRiD demo image.' : `Experimental local-model output · ${(prediction * 100).toFixed(1)}% top-class confidence.`} AI-assisted screening, not autonomous diagnosis.</p>
    </div>
  )
}

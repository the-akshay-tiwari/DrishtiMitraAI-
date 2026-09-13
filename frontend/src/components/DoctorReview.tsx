import { useState } from 'react'
import type { GradeDetails, Patient } from '../types'
import { FundusViewer } from './FundusViewer'

export function DoctorReview({ patient, detail }: { patient: Patient; detail: GradeDetails }) {
  const [reviewed, setReviewed] = useState(patient.reviewStatus === 'Signed off')
  const [signedOff, setSignedOff] = useState(patient.reviewStatus === 'Signed off')
  return (
    <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Referral case</p><h2 className="mt-1 text-xl font-bold text-slate-950">{patient.name} <span className="text-sm font-semibold text-slate-500">· {patient.id}</span></h2></div><span className="rounded-full bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-800">Clinician review required</span></div><FundusViewer detail={detail} mode="overlay" opacity={75} className="mt-5" label="Fundus + prototype evidence" /><div className="mt-5 grid gap-3 sm:grid-cols-3"><div className="mini-stat"><span>AI screening grade</span><strong>{detail.label}</strong></div><div className="mini-stat"><span>Evidence</span><strong>Available</strong></div><div className="mini-stat"><span>Referral</span><strong>{detail.priority}</strong></div></div></div>
      <aside className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Ophthalmologist review</p><h3 className="mt-1 text-lg font-bold text-slate-900">AI assists. Clinician decides.</h3><div className="mt-5 space-y-3 rounded-xl bg-slate-50 p-4 text-sm"><p><span className="font-semibold text-slate-500">Patient:</span> {patient.age} years · {patient.sex}</p><p><span className="font-semibold text-slate-500">Diabetes duration:</span> {patient.diabetesDuration}</p><p><span className="font-semibold text-slate-500">Recommendation:</span> {detail.recommendation}</p></div><label className="field-label mt-5">Clinical review note<textarea defaultValue="Review demonstration — no clinical finding entered." rows={4} /></label><button onClick={() => setReviewed(true)} className="btn-secondary mt-4 w-full">{reviewed ? 'Reviewed by Ophthalmologist ✓' : 'Mark as reviewed'}</button><button onClick={() => { setReviewed(true); setSignedOff(true) }} className="btn-primary mt-3 w-full">{signedOff ? 'Clinical sign-off recorded ✓' : 'Clinical sign-off (demonstration)'}</button><p className="mt-3 text-xs leading-5 text-slate-500">Demonstration control only. Final clinical interpretation remains with an ophthalmologist.</p></aside>
    </div>
  )
}

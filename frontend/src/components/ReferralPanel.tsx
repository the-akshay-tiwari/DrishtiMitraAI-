import { useState } from 'react'
import type { GradeDetails, Patient } from '../types'

export function ReferralPanel({ patient, detail, onCreated }: { patient: Patient; detail: GradeDetails; onCreated: (id: string) => void }) {
  const [priority, setPriority] = useState(detail.priority === 'Priority' ? 'Priority' : 'Standard')
  const [notes, setNotes] = useState('Please review the screening report and visual evidence.')
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Specialist referral</p>
      <h3 className="mt-1 text-lg font-bold text-slate-900">Refer through telemedicine workflow</h3>
      <div className="mt-4 rounded-xl bg-slate-50 p-4 text-sm text-slate-700"><p><strong>Patient:</strong> {patient.name} · {patient.id}</p><p className="mt-1"><strong>Screening grade:</strong> {detail.label} — {detail.clinicalLabel}</p><p className="mt-1"><strong>Evidence:</strong> Grad-CAM and lesion visualizations attached</p></div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2"><label className="field-label">Referral priority<select value={priority} onChange={(event) => setPriority(event.target.value)}><option>Priority</option><option>Standard</option></select></label><label className="field-label">Preferred workflow<select defaultValue="e-Sanjeevani"><option>e-Sanjeevani / telemedicine</option><option>District ophthalmology clinic</option><option>Manual referral export</option></select></label></div>
      <label className="field-label mt-4">CHO note<textarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={3} /></label>
      <button onClick={() => onCreated(`REF-DM-${Math.floor(1000 + Math.random() * 8999)}`)} className="btn-primary mt-5 w-full">Submit referral</button>
      <p className="mt-3 text-xs leading-5 text-slate-500">Prototype integration / workflow demonstration. No live e-Sanjeevani API connection is claimed.</p>
    </div>
  )
}

import type { Patient } from '../types'
import { gradeDetails } from '../data/demo'
import { StatusBadge } from './StatusBadge'

export function PatientCard({ patient, onOpen }: { patient: Patient; onOpen: () => void }) {
  const grade = gradeDetails[patient.grade]
  const priorityTone = grade.priority === 'Priority' ? 'rose' : grade.priority === 'Review' ? 'amber' : 'teal'
  return (
    <button onClick={onOpen} className="group w-full rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-teal-300 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-teal-50 text-sm font-bold text-teal-700">{patient.name.split(' ').map((part) => part[0]).join('').slice(0, 2)}</div>
          <div className="min-w-0">
            <p className="truncate font-bold text-slate-900">{patient.name}</p>
            <p className="mt-0.5 text-xs text-slate-500">{patient.id} · {patient.age} years · {patient.sex}</p>
          </div>
        </div>
        <StatusBadge tone={priorityTone}>{grade.label}</StatusBadge>
      </div>
      <div className="mt-3 flex items-center justify-between border-t border-slate-100 pt-3 text-xs">
        <span className="text-slate-500">{patient.screenDate}</span>
        <span className="font-semibold text-slate-700 group-hover:text-teal-700">Open record →</span>
      </div>
    </button>
  )
}

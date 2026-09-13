import type { ReactNode } from 'react'

type Props = {
  children: ReactNode
  tone?: 'teal' | 'sky' | 'amber' | 'orange' | 'rose' | 'slate'
}

const tones = {
  teal: 'border-teal-200 bg-teal-50 text-teal-800',
  sky: 'border-sky-200 bg-sky-50 text-sky-800',
  amber: 'border-amber-200 bg-amber-50 text-amber-800',
  orange: 'border-orange-200 bg-orange-50 text-orange-800',
  rose: 'border-rose-200 bg-rose-50 text-rose-800',
  slate: 'border-slate-200 bg-slate-50 text-slate-700',
}

export function StatusBadge({ children, tone = 'slate' }: Props) {
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-bold tracking-wide ${tones[tone]}`}>{children}</span>
}

export function OfflineStatus({ online, compact = false }: { online: boolean; compact?: boolean }) {
  return (
    <StatusBadge tone={online ? 'teal' : 'amber'}>
      <span className={`h-1.5 w-1.5 rounded-full ${online ? 'bg-teal-500' : 'bg-amber-500'}`} />
      {compact ? (online ? 'ONLINE' : 'EDGE MODE') : (online ? 'ONLINE — CLOUD SYNC READY' : 'OFFLINE — EDGE MODE ACTIVE')}
    </StatusBadge>
  )
}

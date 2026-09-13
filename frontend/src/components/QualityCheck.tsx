export function QualityCheck({ accepted = true }: { accepted?: boolean }) {
  const checks = [
    ['Image clarity', accepted ? 'Clear' : 'Blur detected'],
    ['Field of view', accepted ? 'Sufficient' : 'Insufficient'],
    ['Illumination', accepted ? 'Balanced' : 'Uneven'],
    ['Retina visibility', accepted ? 'Visible' : 'Partially obscured'],
  ]
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Image-quality gate</p>
      <div className="mt-4 space-y-3">
        {checks.map(([label, result]) => (
          <div className="flex items-center justify-between gap-3" key={label}>
            <span className="text-sm text-slate-600">{label}</span>
            <span className={`inline-flex items-center gap-1.5 text-sm font-bold ${accepted ? 'text-teal-700' : 'text-rose-700'}`}><span>{accepted ? '✓' : '!'}</span>{result}</span>
          </div>
        ))}
      </div>
      <div className={`mt-5 rounded-xl border px-3 py-3 text-sm font-bold ${accepted ? 'border-teal-200 bg-teal-50 text-teal-800' : 'border-rose-200 bg-rose-50 text-rose-800'}`}>
        {accepted ? 'Image Quality: Acceptable' : 'Image Quality: Poor — Please recapture'}
      </div>
      <p className="mt-3 text-xs leading-5 text-slate-500">Prototype quality check. Images that fail this gate should be recaptured before screening.</p>
    </div>
  )
}

import type { GradeDetails } from '../types'

export type ViewerMode = 'original' | 'heatmap' | 'segmentation' | 'overlay'

export function FundusViewer({
  detail,
  mode = 'original',
  opacity = 68,
  label = 'Fundus image',
  className = '',
}: {
  detail: GradeDetails
  mode?: ViewerMode
  opacity?: number
  label?: string
  className?: string
}) {
  const showHeatmap = mode === 'heatmap' || mode === 'overlay'
  const showSegmentation = mode === 'segmentation' || mode === 'overlay'
  const bgStyle = detail.heatmap.startsWith('data:') ? `url("${detail.heatmap}")` : detail.heatmap
  return (
    <div className={`relative overflow-hidden rounded-2xl bg-slate-950 ${className}`}>
      <img className="aspect-[4/3] h-full w-full object-cover" src={detail.image} alt={`${detail.label} fundus image`} />
      {showHeatmap && <div className="pointer-events-none absolute inset-0 bg-cover bg-center mix-blend-screen" style={{ backgroundImage: bgStyle, opacity: opacity / 100 }} aria-label="Grad-CAM overlay" />}

      {showSegmentation && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Prototype lesion segmentation overlay">
          {detail.lesions.length === 0 ? <circle cx="59" cy="50" r="8" fill="#2dd4bf" fillOpacity="0.08" stroke="#5eead4" strokeOpacity="0.65" strokeWidth="0.45" strokeDasharray="1.2 1.2" /> : detail.lesions.map((lesion, i) => (
            <ellipse key={i} cx={parseFloat(lesion.cx)} cy={parseFloat(lesion.cy)} rx={parseFloat(lesion.rx)} ry={parseFloat(lesion.ry)} transform={lesion.rotate ? `rotate(${lesion.rotate} ${parseFloat(lesion.cx)} ${parseFloat(lesion.cy)})` : undefined} fill="#ef4444" fillOpacity={0.24 * (opacity / 100)} stroke="#fbbf24" strokeWidth="0.65" strokeOpacity="0.95" />
          ))}
        </svg>
      )}
      <div className="absolute bottom-3 left-3 rounded-md bg-slate-950/75 px-2 py-1 text-[10px] font-bold tracking-wider text-white backdrop-blur">{label.toUpperCase()}</div>
    </div>
  )
}

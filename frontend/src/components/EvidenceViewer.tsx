import { useState } from 'react'
import type { GradeDetails } from '../types'
import { FundusViewer, type ViewerMode } from './FundusViewer'

type EvidenceKind = 'heatmap' | 'segmentation'

export function EvidenceViewer({ detail, kind }: { detail: GradeDetails; kind: EvidenceKind }) {
  const [mode, setMode] = useState<ViewerMode>(kind === 'heatmap' ? 'heatmap' : 'segmentation')
  const [opacity, setOpacity] = useState(68)
  const isHeatmap = kind === 'heatmap'
  const modes: Array<[ViewerMode, string]> = isHeatmap ? [['original', 'Original'], ['heatmap', 'Heatmap'], ['overlay', 'Overlay']] : [['original', 'Original'], ['segmentation', 'Segmentation'], ['overlay', 'Overlay']]
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">{isHeatmap ? 'Grad-CAM evidence' : 'Attention-gated U-Net evidence'}</p>
          <h3 className="mt-1 text-lg font-bold text-slate-900">{isHeatmap ? 'Model attention regions' : 'Localized lesion evidence'}</h3>
        </div>
        <span className="rounded-full bg-violet-50 px-2.5 py-1 text-xs font-bold text-violet-700">Prototype visualization</span>
      </div>
      <div className="mt-4 flex flex-wrap gap-2" role="tablist" aria-label="Evidence viewer options">
        {modes.map(([value, label]) => <button key={value} role="tab" aria-selected={mode === value} onClick={() => setMode(value)} className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${mode === value ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>{label}</button>)}
      </div>
      <FundusViewer detail={detail} mode={mode} opacity={opacity} className="mt-4" label={mode === 'original' ? 'Original supplied sample' : mode === 'heatmap' ? 'Grad-CAM overlay' : mode === 'segmentation' ? 'Lesion mask' : 'Combined overlay'} />
      {mode !== 'original' && <label className="mt-4 flex items-center gap-3 text-xs font-semibold text-slate-600">Overlay opacity <input aria-label="Overlay opacity" className="h-1.5 flex-1 accent-teal-600" type="range" min="25" max="100" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /><span className="w-9 text-right">{opacity}%</span></label>}
      <p className="mt-4 text-sm leading-6 text-slate-600">{isHeatmap ? 'Highlighted regions indicate areas that contributed strongly to the model’s screening prediction. This AI-generated visual evidence does not itself establish a clinical conclusion.' : 'Lesion segmentation provides localized visual evidence that complements the DR grade. This is a non-clinical prototype visualization, not an inference result.'}</p>
    </div>
  )
}

export function HeatmapViewer({ detail }: { detail: GradeDetails }) {
  return <EvidenceViewer detail={detail} kind="heatmap" />
}

export function LesionMaskViewer({ detail }: { detail: GradeDetails }) {
  return <EvidenceViewer detail={detail} kind="segmentation" />
}

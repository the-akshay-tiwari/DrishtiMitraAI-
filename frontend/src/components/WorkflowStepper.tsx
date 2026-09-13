const steps = ['Capture', 'Quality check', 'Analyze', 'Explain', 'Refer']

export function WorkflowStepper({ active }: { active: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-y-2 text-xs font-bold tracking-wide text-slate-400">
      {steps.map((step, index) => <li key={step} className="flex items-center"><span className={`grid h-6 w-6 place-items-center rounded-full ${index <= active ? 'bg-teal-600 text-white' : 'bg-slate-200 text-slate-500'}`}>{index + 1}</span><span className={`ml-2 ${index === active ? 'text-teal-700' : ''}`}>{step}</span>{index < steps.length - 1 && <span className={`mx-3 h-px w-6 sm:w-10 ${index < active ? 'bg-teal-300' : 'bg-slate-200'}`} />}</li>)}
    </ol>
  )
}

import { useMemo, useState } from 'react'
import { demoPatients, gradeDetails, gradeOptions, today } from './data/demo'
import type { Grade, GradeDetails, Patient } from './types'
import { DoctorReview } from './components/DoctorReview'
import { DRGradeCard } from './components/DRGradeCard'
import { HeatmapViewer, LesionMaskViewer } from './components/EvidenceViewer'
import { FundusViewer } from './components/FundusViewer'
import { OfflineStatus, StatusBadge } from './components/StatusBadge'
import { PatientCard } from './components/PatientCard'
import { QualityCheck } from './components/QualityCheck'
import { ReferralPanel } from './components/ReferralPanel'
import { ReportViewer, type ModelPredictionInfo } from './components/ReportViewer'
import { WorkflowStepper } from './components/WorkflowStepper'

type Page = 'dashboard' | 'workflow' | 'history' | 'report' | 'review' | 'settings'
type ModelPrediction = ModelPredictionInfo


const navItems: Array<{ id: Page; label: string; glyph: string }> = [
  { id: 'dashboard', label: 'Dashboard', glyph: '⌂' },
  { id: 'workflow', label: 'New Screening', glyph: '⊕' },
  { id: 'history', label: 'Patient History', glyph: '▤' },
  { id: 'report', label: 'Screening Report', glyph: '▧' },
  { id: 'review', label: 'Specialist Review', glyph: '◉' },
  { id: 'settings', label: 'System Status', glyph: '⚙' },
]

const newPatient = (): Patient => ({
  id: 'DM-26006', name: 'Sunita Verma', age: 52, sex: 'Female', diabetesDuration: '8 years', lastScreening: 'First screening', contact: '+91 9••• ••• 210', grade: 3, status: 'Priority referral', reviewStatus: 'Not reviewed', screenDate: today,
})

function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [online, setOnline] = useState(true)
  const [patients, setPatients] = useState<Patient[]>(demoPatients)
  const [selectedPatient, setSelectedPatient] = useState<Patient>(demoPatients[0])
  const [selectedGrade, setSelectedGrade] = useState<Grade>(3)
  const [step, setStep] = useState(0)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [uploadName, setUploadName] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [prediction, setPrediction] = useState<ModelPrediction | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [showReferral, setShowReferral] = useState(false)
  const [referralId, setReferralId] = useState<string | null>(null)
  const detail = gradeDetails[selectedGrade]

  const chosenImage = imagePreview ?? detail.image
  const openedPatientDetail = useMemo(() => ({ ...selectedPatient, grade: selectedGrade, status: detail.priority === 'Priority' ? 'Priority referral' as const : detail.priority === 'Review' ? 'Review needed' as const : 'No referral' as const }), [detail.priority, selectedGrade, selectedPatient])

  const openPatient = (patient: Patient, destination: Page = 'workflow', startStep = 3) => {
    setSelectedPatient(patient)
    setSelectedGrade(patient.grade)
    setImagePreview(null)
    setUploadName(null)
    setUploadedFile(null)
    setStep(startStep)
    setAnalysisComplete(true)
    setPrediction(null)
    setAnalysisError(null)
    setShowReferral(false)
    setReferralId(patient.status === 'Priority referral' ? 'REF-DM-4821' : null)
    setPage(destination)
  }

  const beginScreening = () => {
    const patient = newPatient()
    setSelectedPatient(patient)
    setSelectedGrade(3)
    setImagePreview(null)
    setUploadName(null)
    setUploadedFile(null)
    setStep(0)
    setAnalysisComplete(false)
    setPrediction(null)
    setAnalysisError(null)
    setShowReferral(false)
    setReferralId(null)
    setPage('workflow')
  }

  const setDemoCase = (grade: Grade) => {
    setSelectedGrade(grade)
    setImagePreview(null)
    setUploadName(null)
    setUploadedFile(null)
    setAnalysisComplete(false)
    setPrediction(null)
    setAnalysisError(null)
  }

  const previewFile = (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => {
      setImagePreview(String(reader.result))
      setUploadName(file.name)
      setUploadedFile(file)
      setAnalysisComplete(false)
      setPrediction(null)
      setAnalysisError(null)
    }
    reader.readAsDataURL(file)
  }

  const runAnalysis = async () => {
    if (!uploadedFile) {
      setAnalysisError('Upload a retinal image before running the trained local model. The supplied grade buttons are reference demonstrations only.')
      return
    }
    setIsAnalyzing(true)
    setAnalysisError(null)
    try {
      const body = new FormData()
      body.append('image', uploadedFile)
      const isLocalHost = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
      const apiBase = import.meta.env.VITE_API_BASE_URL !== undefined ? import.meta.env.VITE_API_BASE_URL : (isLocalHost ? 'http://127.0.0.1:8000' : '')
      const response = await fetch(`${apiBase}/predict`, { method: 'POST', body })
      
      const rawText = await response.text()
      let result: { detail?: unknown; grade?: unknown; confidence?: unknown; probabilities?: unknown; heatmap?: string } = {}
      if (rawText) {
        try {
          result = JSON.parse(rawText)
        } catch {
          // Response body was not valid JSON (e.g. HTML error page or proxy error)
        }
      }

      if (!response.ok) {
        const detailMsg = typeof result.detail === 'string' ? result.detail : `Server error (${response.status} ${response.statusText || 'Inference failed'}). Please try again.`
        throw new Error(detailMsg)
      }

      if (typeof result.grade !== 'number' || !Number.isInteger(result.grade) || result.grade < 0 || result.grade > 4 || typeof result.confidence !== 'number' || typeof result.probabilities !== 'object' || result.probabilities === null) {
        throw new Error('The inference service returned an invalid prediction structure.')
      }

      setSelectedGrade(result.grade as Grade)
      setPrediction({
        confidence: result.confidence,
        probabilities: result.probabilities as Record<string, number>,
        heatmap: typeof result.heatmap === 'string' ? result.heatmap : undefined,
        performanceMetrics: (result as { performance_metrics?: ModelPredictionInfo['performanceMetrics'] }).performance_metrics,
      })
      setAnalysisComplete(true)
    } catch (error) {
      setAnalysisComplete(false)
      setPrediction(null)
      setAnalysisError(error instanceof Error ? error.message : 'Unable to contact the inference service.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const completeRegistration = (form: HTMLFormElement) => {
    const data = new FormData(form)
    const name = String(data.get('name') || 'Sunita Verma')
    const age = Number(data.get('age') || 52)
    const sex = String(data.get('sex') || 'Female') as Patient['sex']
    const patient: Patient = { ...newPatient(), name, age, sex, diabetesDuration: String(data.get('duration') || '8 years'), contact: String(data.get('contact') || '+91 9••• ••• 210') }
    setSelectedPatient(patient)
    setPatients((existing) => [patient, ...existing.filter((entry) => entry.id !== patient.id)])
    setStep(0)
  }

  const switchPage = (target: Page) => {
    if (target === 'workflow' && page !== 'workflow') beginScreening()
    else setPage(target)
  }

  return (
    <div className="min-h-screen bg-[#f7faf9] text-slate-900">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-[252px] flex-col border-r border-slate-200 bg-white px-4 py-5 lg:flex">
        <Brand />
        <nav className="mt-9 space-y-1" aria-label="Main navigation">
          {navItems.map((item) => <button key={item.id} onClick={() => switchPage(item.id)} className={`nav-link ${page === item.id ? 'nav-link-active' : ''}`}><span className="grid h-7 w-7 place-items-center text-base">{item.glyph}</span>{item.label}</button>)}
        </nav>
        <div className="mt-auto rounded-2xl border border-teal-100 bg-gradient-to-br from-teal-50 to-cyan-50 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.14em] text-teal-700">Safety principle</p>
          <p className="mt-2 text-sm font-semibold leading-5 text-slate-800">AI-assisted screening — not autonomous diagnosis.</p>
          <p className="mt-2 text-xs leading-5 text-slate-600">Human review and ophthalmologist sign-off remain essential.</p>
        </div>
        <p className="mt-4 px-2 text-[11px] font-medium text-slate-400">Prototype Demonstration · SIH26038</p>
      </aside>

      <main className="lg:pl-[252px]">
        <header className="sticky top-0 z-10 flex h-[76px] items-center justify-between gap-4 border-b border-slate-200 bg-white/90 px-5 backdrop-blur lg:px-9">
          <div className="flex min-w-0 items-center gap-3 lg:hidden"><Brand compact /></div>
          <div className="hidden lg:block"><p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Rural Health Centre</p><p className="mt-0.5 text-sm font-bold text-slate-700">CHO Screening Workspace</p></div>
          <div className="ml-auto flex items-center gap-3"><button onClick={() => setOnline((current) => !current)} className="rounded-lg p-1 outline-none transition hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-teal-600" aria-label="Switch online or offline mode"><OfflineStatus online={online} /></button><div className="hidden h-8 w-px bg-slate-200 sm:block" /><div className="hidden text-right sm:block"><p className="text-sm font-bold text-slate-800">Anjali Sharma</p><p className="text-xs text-slate-500">Community Health Officer</p></div><div className="grid h-9 w-9 place-items-center rounded-full bg-slate-900 text-xs font-black text-white">AS</div></div>
        </header>
        <div className="mx-auto max-w-[1500px] px-5 py-7 lg:px-9 lg:py-9">
          {page === 'dashboard' && <Dashboard patients={patients} online={online} onNew={beginScreening} onHistory={() => setPage('history')} onOpen={openPatient} />}
          {page === 'workflow' && <Workflow patient={openedPatientDetail} detail={detail} image={chosenImage} uploadName={uploadName} step={step} online={online} analysisComplete={analysisComplete} isAnalyzing={isAnalyzing} prediction={prediction} analysisError={analysisError} showReferral={showReferral} referralId={referralId} onRegister={completeRegistration} onGrade={setDemoCase} onUpload={previewFile} onDrop={previewFile} onNext={() => setStep((current) => Math.min(current + 1, 4))} onBack={() => setStep((current) => Math.max(current - 1, 0))} onAnalyze={runAnalysis} onShowReferral={() => setShowReferral(true)} onReferralCreated={(id) => { setReferralId(id); setShowReferral(false) }} onReview={() => setPage('review')} onReport={() => setPage('report')} onCamera={() => setDemoCase(((selectedGrade + 1) % 5) as Grade)} />}
          {page === 'history' && <History patients={patients} onOpen={openPatient} />}
          {page === 'report' && <ReportPage patient={openedPatientDetail} detail={detail} online={online} prediction={prediction} onRefer={() => { setPage('workflow'); setStep(4); setShowReferral(true) }} />}
          {page === 'review' && <ReviewPage patient={openedPatientDetail} detail={detail} />}
          {page === 'settings' && <Settings online={online} onToggle={() => setOnline((current) => !current)} />}
        </div>
      </main>
    </div>
  )
}

function Brand({ compact = false }: { compact?: boolean }) {
  return <div className="flex min-w-0 items-center gap-3"><div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-teal-700 text-xl text-white shadow-lg shadow-teal-700/20">◉</div><div className={compact ? 'hidden sm:block' : ''}><p className="text-lg font-black tracking-tight text-slate-950">Drishti<span className="text-teal-700">Mitra</span></p><p className="-mt-0.5 text-[10px] font-extrabold uppercase tracking-[0.18em] text-slate-500">Retina-XAI</p></div></div>
}

function Dashboard({ patients, online, onNew, onHistory, onOpen }: { patients: Patient[]; online: boolean; onNew: () => void; onHistory: () => void; onOpen: (patient: Patient) => void }) {
  const pending = patients.filter((p) => p.reviewStatus !== 'Signed off').length
  const referrals = patients.filter((p) => p.status === 'Priority referral').length
  return <>
    <section className="flex flex-col justify-between gap-6 rounded-3xl bg-slate-950 px-6 py-7 text-white shadow-xl shadow-slate-900/10 md:flex-row md:items-end lg:px-8">
      <div className="max-w-2xl"><p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-300">Good morning, Anjali</p><h1 className="mt-2 text-3xl font-black tracking-tight md:text-4xl">Screen patients with clarity.<br /><span className="text-teal-300">Escalate with context.</span></h1><p className="mt-4 max-w-xl text-sm leading-6 text-slate-300">A guided rural point-of-care workflow for diabetic-retinopathy screening support, visual evidence, and specialist review.</p></div>
      <div className="flex flex-wrap gap-3"><button onClick={onNew} className="btn-light">＋ New Screening</button><button onClick={onHistory} className="btn-dark-outline">Patient History</button></div>
    </section>
    <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Stat title="Today's screenings" value="12" note="3 completed today" glyph="▣" tone="teal" />
      <Stat title="Pending reviews" value={String(pending)} note="Ophthalmologist queue" glyph="◉" tone="amber" />
      <Stat title="Referral cases" value={String(referrals)} note="Priority referrals" glyph="↗" tone="rose" />
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-sm font-semibold text-slate-600">System status</p><div className="mt-4"><OfflineStatus online={online} compact /></div><p className="mt-4 text-xs leading-5 text-slate-500">{online ? 'Cloud sync available. Prototype workflow remains local.' : 'Local screening and storage available. Pending items sync when connected.'}</p></div>
    </section>
    <section className="mt-8 grid gap-6 xl:grid-cols-[1.05fr_.95fr]">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Today’s queue</p><h2 className="mt-1 text-xl font-bold text-slate-950">Recent patients</h2></div><button onClick={onHistory} className="text-sm font-bold text-teal-700 hover:text-teal-800">View history →</button></div><div className="mt-5 grid gap-3 md:grid-cols-2">{patients.slice(0, 4).map((patient) => <PatientCard key={patient.id} patient={patient} onOpen={() => onOpen(patient)} />)}</div></div>
      <div className="rounded-2xl border border-teal-100 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[0.14em] text-teal-700">Guided care pathway</p><h2 className="mt-1 text-xl font-bold text-slate-950">Capture → Analyze → Explain → Refer</h2><div className="mt-5 space-y-0">{[['1', 'Capture', 'Select a supplied IDRiD demonstration image or upload an image.'], ['2', 'Analyze', 'Quality-gated deterministic prototype screening & grading.'], ['3', 'Explain', 'Review Grad-CAM attention and lesion visual evidence.'], ['4', 'Refer', 'Generate a FHIR/ABDM-ready report and refer for sign-off.']].map(([number, title, copy], index) => <div className="relative flex gap-4 pb-5 last:pb-0" key={title}>{index < 3 && <span className="absolute left-4 top-9 h-[calc(100%-22px)] w-px bg-teal-100" />}<span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-teal-700 text-xs font-black text-white">{number}</span><div><p className="font-bold text-slate-900">{title}</p><p className="mt-0.5 text-sm leading-5 text-slate-600">{copy}</p></div></div>)}</div></div>
    </section>
    <section className="mt-6 grid gap-4 md:grid-cols-3"><InfoPanel title="Image-quality gate" icon="✓" text="Guide recapture before any screening workflow continues." /><InfoPanel title="Human-in-the-loop" icon="◌" text="Visual evidence supports—not replaces—clinical review." /><InfoPanel title="FHIR / ABDM-ready" icon="⇄" text="Prototype records are structured for future interoperability." /></section>
  </>
}

function Stat({ title, value, note, glyph, tone }: { title: string; value: string; note: string; glyph: string; tone: 'teal' | 'amber' | 'rose' }) {
  const style = tone === 'teal' ? 'bg-teal-50 text-teal-700' : tone === 'amber' ? 'bg-amber-50 text-amber-700' : 'bg-rose-50 text-rose-700'
  return <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-start justify-between"><p className="text-sm font-semibold text-slate-600">{title}</p><span className={`grid h-9 w-9 place-items-center rounded-xl text-lg ${style}`}>{glyph}</span></div><p className="mt-4 text-3xl font-black tracking-tight text-slate-950">{value}</p><p className="mt-1 text-xs text-slate-500">{note}</p></div>
}

function InfoPanel({ title, icon, text }: { title: string; icon: string; text: string }) {
  return <div className="flex gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-slate-100 font-bold text-slate-700">{icon}</span><div><p className="font-bold text-slate-900">{title}</p><p className="mt-1 text-sm leading-5 text-slate-600">{text}</p></div></div>
}

type WorkflowProps = {
  patient: Patient; detail: GradeDetails; image: string; uploadName: string | null; step: number; online: boolean; analysisComplete: boolean; isAnalyzing: boolean; prediction: ModelPrediction | null; analysisError: string | null; showReferral: boolean; referralId: string | null;
  onRegister: (form: HTMLFormElement) => void; onGrade: (grade: Grade) => void; onUpload: (file?: File) => void; onDrop: (file?: File) => void; onNext: () => void; onBack: () => void; onAnalyze: () => void; onShowReferral: () => void; onReferralCreated: (id: string) => void; onReview: () => void; onReport: () => void; onCamera: () => void
}

function Workflow(props: WorkflowProps) {
  const { patient, detail, image, uploadName, step, online, analysisComplete, isAnalyzing, prediction, analysisError, showReferral, referralId, onRegister, onGrade, onUpload, onDrop, onNext, onBack, onAnalyze, onShowReferral, onReferralCreated, onReview, onReport, onCamera } = props
  const customDetail = { ...detail, image }
  return <>
    <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end"><div><p className="eyebrow">New screening</p><h1 className="page-title">{step === 0 ? 'Register & capture' : step === 1 ? 'Image quality check' : step === 2 ? 'AI screening & grading' : step === 3 ? 'Explainable AI evidence' : 'Report & referral'}</h1><p className="mt-2 text-sm text-slate-600">{patient.name} · {patient.id} · <span className="font-semibold">Prototype Demonstration</span></p></div><OfflineStatus online={online} /></div>
    <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"><WorkflowStepper active={step} /></div>
    <div className="mt-6">
      {step === 0 && <CaptureStep patient={patient} detail={customDetail} uploadName={uploadName} onRegister={onRegister} onGrade={onGrade} onUpload={onUpload} onDrop={onDrop} onNext={onNext} onCamera={onCamera} />}
      {step === 1 && <QualityStep detail={customDetail} onBack={onBack} onNext={onNext} />}
      {step === 2 && <AnalysisStep detail={customDetail} online={online} complete={analysisComplete} isAnalyzing={isAnalyzing} prediction={prediction} error={analysisError} onBack={onBack} onAnalyze={onAnalyze} onNext={onNext} />}
      {step === 3 && <ExplainStep detail={customDetail} prediction={prediction} onBack={onBack} onNext={onNext} />}
      {step === 4 && <ReportStep patient={patient} detail={customDetail} online={online} prediction={prediction} showReferral={showReferral} referralId={referralId} onBack={onBack} onRefer={onShowReferral} onReferralCreated={onReferralCreated} onReview={onReview} onReport={onReport} />}
    </div>
  </>
}

function CaptureStep({ patient, detail, uploadName, onRegister, onGrade, onUpload, onDrop, onNext, onCamera }: { patient: Patient; detail: GradeDetails; uploadName: string | null; onRegister: (form: HTMLFormElement) => void; onGrade: (grade: Grade) => void; onUpload: (file?: File) => void; onDrop: (file?: File) => void; onNext: () => void; onCamera: () => void }) {
  const [editing, setEditing] = useState(false)
  return <div className="grid gap-6 xl:grid-cols-[.85fr_1.15fr]">
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-start justify-between gap-4"><div><p className="eyebrow">Patient registration</p><h2 className="mt-1 text-xl font-bold text-slate-950">{editing ? 'Register patient' : 'Patient selected'}</h2></div><button onClick={() => setEditing((current) => !current)} className="text-sm font-bold text-teal-700">{editing ? 'Use selected patient' : 'Edit / register'}</button></div>{editing ? <PatientForm patient={patient} onSave={(form) => { onRegister(form); setEditing(false) }} /> : <div className="mt-5 space-y-0 rounded-xl bg-slate-50 p-4"><DataLine label="Patient ID" value={patient.id} /><DataLine label="Name" value={patient.name} /><DataLine label="Age / sex" value={`${patient.age} years · ${patient.sex}`} /><DataLine label="Diabetes duration" value={patient.diabetesDuration} /><DataLine label="Previous eye screening" value={patient.lastScreening} /><DataLine label="Contact" value={patient.contact} /></div>}<div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950"><strong>Consent reminder:</strong> confirm patient consent before capturing or storing a retinal image.</div></section>
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="eyebrow">Fundus capture / upload</p><h2 className="mt-1 text-xl font-bold text-slate-950">Upload a fundus image for model screening</h2></div><span className="rounded-full bg-violet-50 px-3 py-1.5 text-xs font-bold text-violet-700">Research-model input</span></div><div className="mt-5 grid gap-5 lg:grid-cols-[1fr_.86fr]"><div><FundusViewer detail={detail} label={uploadName ? 'Uploaded image — ready for local model' : 'Supplied reference image'} /><p className="mt-3 text-xs leading-5 text-slate-500">{uploadName ? `Uploaded: ${uploadName}. This image will be sent only to the local model when you run screening.` : `Selected ${detail.label} supplied reference case. Upload an image to use the trained model.`}</p></div><div><div onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); onDrop(event.dataTransfer.files[0]) }} className="rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 p-4 text-center"><p className="font-bold text-slate-800">Upload image</p><p className="mt-1 text-xs leading-5 text-slate-500">Drag and drop a retina image, or choose a file for local model screening.</p><label className="btn-secondary mt-3 inline-flex cursor-pointer"><input type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(event) => onUpload(event.target.files?.[0])} />Choose file</label></div><button onClick={onCamera} className="btn-secondary mt-3 w-full">▣ Simulate camera capture</button><p className="mt-5 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Known grade reference cases</p><div className="mt-3 grid grid-cols-5 gap-2">{gradeOptions.map((option: GradeDetails) => <button onClick={() => onGrade(option.grade)} key={option.grade} className={`rounded-xl border p-2 text-center transition ${detail.grade === option.grade ? 'border-teal-500 bg-teal-50 text-teal-800' : 'border-slate-200 text-slate-600 hover:border-teal-300'}`}><span className="block text-sm font-black">{option.grade}</span><span className="mt-0.5 block text-[10px] font-semibold">G{option.grade}</span></button>)}</div><p className="mt-2 text-[11px] leading-4 text-slate-500">Reference cases are UI examples only and cannot be submitted to the local model.</p></div></div><div className="mt-6 flex flex-wrap justify-end gap-3 border-t border-slate-100 pt-5"><button onClick={onNext} className="btn-primary">Continue to quality check →</button></div></section>
  </div>
}

function PatientForm({ patient, onSave }: { patient: Patient; onSave: (form: HTMLFormElement) => void }) {
  return <form className="mt-5 space-y-4" onSubmit={(event) => { event.preventDefault(); onSave(event.currentTarget) }}><div className="grid gap-3 sm:grid-cols-2"><label className="field-label">Patient ID<input name="id" defaultValue={patient.id} /></label><label className="field-label">Name<input name="name" defaultValue={patient.name} required /></label><label className="field-label">Age<input name="age" type="number" min="18" max="110" defaultValue={patient.age} required /></label><label className="field-label">Sex<select name="sex" defaultValue={patient.sex}><option>Female</option><option>Male</option><option>Other</option></select></label><label className="field-label">Diabetes duration<input name="duration" defaultValue={patient.diabetesDuration} /></label><label className="field-label">Previous eye screening<select name="previous" defaultValue={patient.lastScreening}><option>First screening</option><option>Within last 12 months</option><option>More than 12 months ago</option></select></label></div><label className="field-label">Contact information<input name="contact" defaultValue={patient.contact} /></label><button type="submit" className="btn-primary w-full">Save patient & continue</button></form>
}

function DataLine({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-4 border-b border-slate-200 py-3 last:border-0 last:pb-0 first:pt-0"><span className="text-sm text-slate-500">{label}</span><span className="text-right text-sm font-bold text-slate-800">{value}</span></div>
}

function QualityStep({ detail, onBack, onNext }: { detail: GradeDetails; onBack: () => void; onNext: () => void }) {
  return <div className="grid gap-6 xl:grid-cols-[1.1fr_.9fr]"><div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="eyebrow">Capture review</p><h2 className="mt-1 text-xl font-bold text-slate-950">Is the retina image ready for screening?</h2><FundusViewer detail={detail} className="mt-5" label="Quality check input" /><div className="mt-5 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">The quality gate occurs before preprocessing and AI screening. It visually demonstrates a safeguard against processing poor-quality captures.</div></div><div><QualityCheck /><div className="mt-5 flex justify-between gap-3"><button onClick={onBack} className="btn-secondary">← Retake / change</button><button onClick={onNext} className="btn-primary">Start AI screening →</button></div></div></div>
}

function AnalysisStep({ detail, online, complete, isAnalyzing, prediction, error, onBack, onAnalyze, onNext }: { detail: GradeDetails; online: boolean; complete: boolean; isAnalyzing: boolean; prediction: ModelPrediction | null; error: string | null; onBack: () => void; onAnalyze: () => void; onNext: () => void }) {
  const isLocalHost = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  return <div className="grid gap-6 xl:grid-cols-[1fr_.95fr]"><div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="eyebrow">{isLocalHost ? 'Local experimental inference' : 'Experimental inference'}</p><h2 className="mt-1 text-xl font-bold text-slate-950">AI screening & grading</h2><FundusViewer detail={detail} className="mt-5" label="Preprocessed model input" /><div className="mt-5 grid gap-3 sm:grid-cols-3">{[['1', 'Preprocess', 'Resize · normalize'], ['2', 'Deep learning model', 'EfficientNet-B0 transfer learning'], ['3', 'Screening grade', 'Grade 0–4 classification']].map(([num, title, copy], index) => <div key={title} className={`rounded-xl border p-3 ${complete && index === 2 ? 'border-teal-200 bg-teal-50' : 'border-slate-200 bg-slate-50'}`}><span className="text-xs font-black text-teal-700">0{num}</span><p className="mt-2 text-sm font-bold text-slate-800">{title}</p><p className="mt-1 text-xs leading-5 text-slate-500">{copy}</p></div>)}</div></div><div>{complete ? <><DRGradeCard detail={detail} prediction={prediction?.confidence} prominent />{prediction && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950"><strong>Experimental result:</strong> {(prediction.confidence * 100).toFixed(1)}% top-class confidence. This is not calibrated clinical certainty and must be reviewed by an ophthalmologist.</div>}</> : <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Ready to run</p><h3 className="mt-1 text-xl font-bold text-slate-950">{isLocalHost ? 'Run the trained local model' : 'Run the screening model'}</h3><p className="mt-4 text-sm leading-6 text-slate-600">{isLocalHost ? 'The uploaded image is sent to the local FastAPI service at 127.0.0.1. It is not uploaded to a third-party cloud service.' : 'The uploaded image is sent to the DrishtiMitra inference service for automated screening.'}</p>{error && <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-900"><strong>Screening unavailable:</strong> {error}</div>}<div className="mt-5 rounded-xl bg-slate-50 p-4"><OfflineStatus online={online} compact /><p className="mt-3 text-sm text-slate-700">{isLocalHost ? 'Start the local model service after training, then run screening on an uploaded retinal image.' : 'Ensure the backend service is reachable, then run screening on an uploaded retinal image.'}</p></div><button disabled={isAnalyzing} onClick={onAnalyze} className="btn-primary mt-5 w-full">{isAnalyzing ? (isLocalHost ? 'Running local model…' : 'Running model…') : (isLocalHost ? 'Run trained model' : 'Run screening model')}</button></div>}<div className="mt-5 flex justify-between gap-3"><button onClick={onBack} className="btn-secondary">← Quality check</button>{complete && <button onClick={onNext} className="btn-primary">View evidence status →</button>}</div></div></div>
}

function ExplainStep({ detail, prediction, onBack, onNext }: { detail: GradeDetails; prediction: ModelPrediction | null; onBack: () => void; onNext: () => void }) {
  const activeDetail = prediction?.heatmap ? { ...detail, heatmap: prediction.heatmap } : detail
  return <><div className="rounded-2xl border border-teal-100 bg-teal-50 px-5 py-4 text-sm leading-6 text-teal-950"><strong>Evidence summary:</strong> {detail.label} {prediction?.heatmap ? '· Live PyTorch Grad-CAM Heatmap Generated' : '· Prototype Visuals'} · <strong>{detail.recommendation}</strong></div><div className="mt-6 grid gap-6 xl:grid-cols-2"><HeatmapViewer detail={activeDetail} /><LesionMaskViewer detail={detail} /></div><div className="mt-6 flex justify-between gap-3"><button onClick={onBack} className="btn-secondary">← Screening result</button><button onClick={onNext} className="btn-primary">Generate research report →</button></div></>
}

function ReportStep({ patient, detail, online, prediction, showReferral, referralId, onBack, onRefer, onReferralCreated, onReview, onReport }: { patient: Patient; detail: GradeDetails; online: boolean; prediction: ModelPrediction | null; showReferral: boolean; referralId: string | null; onBack: () => void; onRefer: () => void; onReferralCreated: (id: string) => void; onReview: () => void; onReport: () => void }) {
  return <div className="grid gap-6 xl:grid-cols-[1.15fr_.85fr]"><div><ReportViewer patient={patient} detail={detail} online={online} prediction={prediction} /><div className="mt-5 flex flex-wrap gap-3"><button onClick={() => window.print()} className="btn-primary">📄 Generate Patient Report (PDF)</button><button onClick={onReport} className="btn-secondary">Open full report page</button><button onClick={onReview} className="btn-secondary">Open specialist review →</button></div></div><aside>{referralId ? <div className="rounded-2xl border border-teal-200 bg-teal-50 p-6 shadow-sm"><span className="grid h-11 w-11 place-items-center rounded-full bg-teal-700 text-lg font-black text-white">✓</span><p className="mt-4 text-xs font-bold uppercase tracking-[0.14em] text-teal-700">Referral Created</p><h3 className="mt-1 text-xl font-black text-slate-950">{referralId}</h3><p className="mt-3 text-sm leading-6 text-slate-700">Referral workflow is queued for specialist review. This is a prototype integration / workflow demonstration.</p><button onClick={onReview} className="btn-primary mt-5 w-full">Open ophthalmologist review</button></div> : showReferral ? <ReferralPanel patient={patient} detail={detail} onCreated={onReferralCreated} /> : <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Next step</p><h3 className="mt-1 text-xl font-bold text-slate-950">Refer with evidence</h3><p className="mt-4 text-sm leading-6 text-slate-600">{prediction ? 'The research report includes the uploaded image and experimental classifier output. It does not include patient-specific heatmaps or lesion masks.' : 'The demonstration report includes prototype evidence visualizations and a recommendation for specialist consideration.'}</p><button onClick={onRefer} className="btn-primary mt-5 w-full">Refer to specialist</button><div className="mt-5 rounded-xl bg-slate-50 p-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-slate-500">Referral flow</p><p className="mt-2 text-sm leading-6 text-slate-700">Rural Health Centre → DrishtiMitra → Screening Report → e-Sanjeevani / Telemedicine → Ophthalmologist → Final clinical review / sign-off</p></div></div>}<button onClick={onBack} className="btn-secondary mt-5">← Back to evidence</button></aside></div>
}

function History({ patients, onOpen }: { patients: Patient[]; onOpen: (patient: Patient, destination?: Page, startStep?: number) => void }) {
  const [query, setQuery] = useState('')
  const visible = patients.filter((patient) => `${patient.name} ${patient.id}`.toLowerCase().includes(query.toLowerCase()))
  return <><div className="flex flex-col justify-between gap-4 md:flex-row md:items-end"><div><p className="eyebrow">Longitudinal care</p><h1 className="page-title">Patient history</h1><p className="mt-2 text-sm text-slate-600">Demonstration records with screening grade, evidence, referral, and review status.</p></div><label className="relative"><span className="sr-only">Search patients</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search patient or ID" className="min-w-[245px]" /></label></div><div className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"><div className="hidden grid-cols-[1.25fr_.7fr_.7fr_1fr_1fr_.65fr] gap-4 border-b border-slate-200 bg-slate-50 px-5 py-3 text-xs font-bold uppercase tracking-[0.12em] text-slate-500 lg:grid"><span>Patient</span><span>Screening</span><span>DR grade</span><span>Recommendation</span><span>Doctor review</span><span></span></div>{visible.map((patient) => { const detail = gradeDetails[patient.grade]; const tone = detail.priority === 'Priority' ? 'rose' : detail.priority === 'Review' ? 'amber' : 'teal'; return <div key={patient.id} className="grid gap-3 border-b border-slate-100 px-5 py-4 last:border-0 lg:grid-cols-[1.25fr_.7fr_.7fr_1fr_1fr_.65fr] lg:items-center lg:gap-4"><div><p className="font-bold text-slate-900">{patient.name}</p><p className="mt-0.5 text-xs text-slate-500">{patient.id} · {patient.age} years</p></div><p className="text-sm text-slate-600">{patient.screenDate}</p><StatusBadge tone={tone}>{detail.label}</StatusBadge><p className="text-sm font-semibold text-slate-700">{detail.priority === 'Priority' ? 'Priority referral' : detail.priority === 'Review' ? 'Specialist review' : 'Routine follow-up'}</p><p className="text-sm text-slate-600">{patient.reviewStatus}</p><button onClick={() => onOpen(patient)} className="text-left text-sm font-bold text-teal-700 hover:text-teal-800 lg:text-right">View →</button></div> })}</div><p className="mt-4 text-xs text-slate-500">All records are controlled demonstration data. Click a record to open its fundus image, prototype evidence, and screening report workflow.</p></>
}

function ReportPage({ patient, detail, online, prediction, onRefer }: { patient: Patient; detail: GradeDetails; online: boolean; prediction: ModelPrediction | null; onRefer: () => void }) {
  const [printed, setPrinted] = useState(false)
  return <><div className="flex flex-col justify-between gap-4 md:flex-row md:items-end"><div><p className="eyebrow">FHIR / ABDM-ready prototype</p><h1 className="page-title">Screening report</h1><p className="mt-2 text-sm text-slate-600">Structured as an interoperable screening record — no live ABDM, ABHA, or FHIR server integration.</p></div><div className="flex gap-3"><button onClick={() => { window.print(); setPrinted(true) }} className="btn-primary">📄 Generate Patient Report (PDF)</button><button onClick={onRefer} className="btn-primary">Refer to specialist</button></div></div>{printed && <div className="mt-5 rounded-xl border border-teal-200 bg-teal-50 px-4 py-3 text-sm font-semibold text-teal-900">Print dialog requested. In a production integration, this would export a signed FHIR-compatible report.</div>}<div className="mt-6"><ReportViewer patient={patient} detail={detail} online={online} prediction={prediction} /></div><div className="mt-6 grid gap-4 md:grid-cols-3"><InfoPanel title="Patient" icon="P" text={`${patient.id} · screening date ${patient.screenDate}`} /><InfoPanel title="Observation" icon="O" text={`${detail.label} evidence and recommendation represented as prototype record fields.`} /><InfoPanel title="Service request" icon="R" text="Referral status can be carried to a specialist workflow when integrated." /></div></>
}

function ReviewPage({ patient, detail }: { patient: Patient; detail: GradeDetails }) {
  return <><div><p className="eyebrow">Human-in-the-loop review</p><h1 className="page-title">Ophthalmologist review</h1><p className="mt-2 text-sm text-slate-600">Review the screening support record, visual evidence, referral context, and enter a clinical decision.</p></div><div className="mt-6"><DoctorReview patient={patient} detail={detail} /></div></>
}

function Settings({ online, onToggle }: { online: boolean; onToggle: () => void }) {
  return <><div className="flex flex-col justify-between gap-4 md:flex-row md:items-end"><div><p className="eyebrow">Prototype architecture</p><h1 className="page-title">System status & settings</h1><p className="mt-2 text-sm text-slate-600">Toggle connectivity to demonstrate the same CHO screening workflow in cloud or local edge mode.</p></div><OfflineStatus online={online} /></div><section className="mt-6 grid gap-6 xl:grid-cols-[.95fr_1.05fr]"><div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Connectivity simulation</p><h2 className="mt-1 text-xl font-bold text-slate-950">{online ? 'Cloud mode available' : 'Edge mode active'}</h2><div className="mt-6 flex items-center justify-between rounded-2xl bg-slate-50 p-5"><div><p className="font-bold text-slate-900">Online / Offline</p><p className="mt-1 text-sm leading-5 text-slate-600">{online ? 'Cloud processing and synchronization are conceptually available.' : 'Local workflow and result storage remain available; synchronization is queued.'}</p></div><button onClick={onToggle} className={`relative h-8 w-14 rounded-full transition ${online ? 'bg-teal-600' : 'bg-slate-300'}`} aria-label="Toggle connection"><span className={`absolute top-1 h-6 w-6 rounded-full bg-white shadow transition ${online ? 'left-7' : 'left-1'}`} /></button></div><div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950"><strong>Prototype note:</strong> offline screening available in prototype demonstration. No actual cloud inference, device sync, or local ML runtime is connected.</div></div><div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Conceptual processing path</p><h2 className="mt-1 text-xl font-bold text-slate-950">{online ? 'Fundus image → Cloud / production server → AI screening → Report → Specialist referral' : 'Fundus image → Edge workflow → Local result storage → Pending sync when connected'}</h2><div className="mt-5 grid grid-cols-4 gap-2">{(online ? ['Fundus image', 'Cloud server', 'Screening', 'Report / referral'] : ['Fundus image', 'Edge mode', 'Local record', 'Pending sync']).map((item, index) => <div key={item} className="relative rounded-xl bg-slate-50 p-3 text-center text-xs font-bold leading-5 text-slate-700"><span className="mb-2 grid h-6 w-6 place-items-center rounded-full bg-teal-700 text-[10px] text-white">{index + 1}</span>{item}{index < 3 && <span className="absolute -right-2 top-6 z-10 text-teal-500">→</span>}</div>)}</div></div></section><section className="mt-6 grid gap-6 lg:grid-cols-2"><div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Digital health record</p><h2 className="mt-1 text-xl font-bold text-slate-950">FHIR / ABDM-ready prototype</h2><div className="mt-5 grid gap-3 sm:grid-cols-2">{[['FHIR R4', 'Patient, Observation, DiagnosticReport and ServiceRequest-ready field structure'], ['ABDM / ABHA', 'Interoperability-ready wording only; no live government integration'], ['Screening record', 'Date, grade, evidence, recommendation, referral status'], ['Privacy path', 'Production roadmap: consent, encrypted storage, role-based access']].map(([title, copy]) => <div className="rounded-xl bg-slate-50 p-4" key={title}><p className="font-bold text-slate-900">{title}</p><p className="mt-1 text-sm leading-5 text-slate-600">{copy}</p></div>)}</div></div><div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Proposed technical approach</p><h2 className="mt-1 text-xl font-bold text-slate-950">Ready for ML integration</h2><div className="mt-5 flex flex-wrap gap-2">{['MATLAB', 'Deep Learning Toolbox', 'Medical Imaging Toolbox', 'Computer Vision Toolbox', 'ResNet-50', 'EfficientNet-B0', 'Attention-gated U-Net', 'Grad-CAM', 'GPU Coder'].map((tech) => <span key={tech} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-bold text-slate-700">{tech}</span>)}</div><p className="mt-5 text-sm leading-6 text-slate-600">Proposed data role: IDRiD, EyePACS, and APTOS for training, evaluation, and cross-dataset validation. No dataset statistics or trained-model performance is claimed in this prototype.</p></div></section><section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><p className="eyebrow">Proposed validation targets</p><h2 className="mt-1 text-xl font-bold text-slate-950">Targets to evaluate in a future clinical ML programme</h2><div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{[['QWK', '> 0.90'], ['Sensitivity', '> 90–93%'], ['Specificity', '> 90–93%'], ['Lesion DSC', '> 0.50 MAs · > 0.70 larger exudates']].map(([metric, target]) => <div key={metric} className="rounded-xl bg-slate-50 p-4"><p className="text-sm font-bold text-slate-600">{metric}</p><p className="mt-1 text-xl font-black text-slate-950">{target}</p></div>)}</div><p className="mt-4 text-xs font-semibold text-rose-700">These are proposed validation targets from the project material — not achieved model results and not clinical performance claims.</p></section></>
}

export default App

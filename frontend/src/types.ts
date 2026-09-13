export type Grade = 0 | 1 | 2 | 3 | 4

export type Patient = {
  id: string
  name: string
  age: number
  sex: 'Female' | 'Male' | 'Other'
  diabetesDuration: string
  lastScreening: string
  contact: string
  grade: Grade
  status: 'No referral' | 'Review needed' | 'Priority referral'
  reviewStatus: 'Not reviewed' | 'Awaiting review' | 'Signed off'
  screenDate: string
}

export type GradeDetails = {
  grade: Grade
  label: string
  clinicalLabel: string
  recommendation: string
  priority: 'Routine' | 'Review' | 'Priority'
  image: string
  tone: string
  heatmap: string
  lesions: Array<{ cx: string; cy: string; rx: string; ry: string; rotate?: number }>
}

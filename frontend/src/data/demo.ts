import type { Grade, GradeDetails, Patient } from '../types'

export const gradeDetails: Record<Grade, GradeDetails> = {
  0: {
    grade: 0,
    label: 'Grade 0',
    clinicalLabel: 'No apparent DR',
    recommendation: 'Routine diabetes eye-screening follow-up.',
    priority: 'Routine',
    image: '/assets/fundus/grade_0_normal.jpg',
    tone: 'teal',
    heatmap: 'radial-gradient(ellipse at 60% 48%, rgba(59,130,246,.25) 0%, rgba(59,130,246,.08) 18%, transparent 42%)',
    lesions: [],
  },
  1: {
    grade: 1,
    label: 'Grade 1',
    clinicalLabel: 'Mild DR',
    recommendation: 'Schedule a follow-up eye review as per local protocol.',
    priority: 'Routine',
    image: '/assets/fundus/grade_1_mild.jpg',
    tone: 'sky',
    heatmap: 'radial-gradient(circle at 59% 61%, rgba(250,204,21,.8) 0%, rgba(249,115,22,.5) 12%, transparent 29%), radial-gradient(circle at 50% 44%, rgba(239,68,68,.42) 0%, transparent 18%)',
    lesions: [{ cx: '59%', cy: '61%', rx: '4%', ry: '3%' }, { cx: '51%', cy: '45%', rx: '2%', ry: '2%' }],
  },
  2: {
    grade: 2,
    label: 'Grade 2',
    clinicalLabel: 'Moderate DR',
    recommendation: 'Specialist review recommended; send screening report.',
    priority: 'Review',
    image: '/assets/fundus/grade_2_moderate.jpg',
    tone: 'amber',
    heatmap: 'radial-gradient(circle at 49% 46%, rgba(250,204,21,.95) 0%, rgba(249,115,22,.62) 10%, transparent 26%), radial-gradient(circle at 63% 26%, rgba(239,68,68,.55) 0%, transparent 16%)',
    lesions: [{ cx: '49%', cy: '46%', rx: '6%', ry: '5%', rotate: -12 }, { cx: '62%', cy: '27%', rx: '3%', ry: '2%' }],
  },
  3: {
    grade: 3,
    label: 'Grade 3',
    clinicalLabel: 'Severe DR',
    recommendation: 'Priority specialist referral recommended.',
    priority: 'Priority',
    image: '/assets/fundus/grade_3_severe.jpg',
    tone: 'orange',
    heatmap: 'radial-gradient(ellipse at 38% 58%, rgba(250,204,21,.95) 0%, rgba(239,68,68,.68) 16%, transparent 36%), radial-gradient(circle at 48% 32%, rgba(239,68,68,.65) 0%, transparent 19%)',
    lesions: [{ cx: '38%', cy: '58%', rx: '13%', ry: '19%', rotate: 20 }, { cx: '47%', cy: '31%', rx: '10%', ry: '8%', rotate: -14 }, { cx: '25%', cy: '29%', rx: '5%', ry: '5%' }],
  },
  4: {
    grade: 4,
    label: 'Grade 4',
    clinicalLabel: 'Proliferative DR',
    recommendation: 'Urgent ophthalmologist review and priority referral recommended.',
    priority: 'Priority',
    image: '/assets/fundus/grade_4_proliferative.jpg',
    tone: 'rose',
    heatmap: 'radial-gradient(ellipse at 57% 52%, rgba(239,68,68,.8) 0%, rgba(185,28,28,.55) 17%, transparent 37%), radial-gradient(circle at 38% 47%, rgba(250,204,21,.82) 0%, transparent 19%)',
    lesions: [{ cx: '57%', cy: '52%', rx: '16%', ry: '18%', rotate: -10 }, { cx: '40%', cy: '47%', rx: '9%', ry: '8%', rotate: 20 }, { cx: '50%', cy: '62%', rx: '7%', ry: '11%' }],
  },
}

export const demoPatients: Patient[] = [
  { id: 'DM-26001', name: 'Meera Devi', age: 58, sex: 'Female', diabetesDuration: '11 years', lastScreening: 'First screening', contact: '+91 98••• 48211', grade: 3, status: 'Priority referral', reviewStatus: 'Awaiting review', screenDate: '01 Sep 2026' },
  { id: 'DM-26002', name: 'Ramesh Kumar', age: 63, sex: 'Male', diabetesDuration: '14 years', lastScreening: '12 Sep 2025', contact: '+91 97••• 11740', grade: 4, status: 'Priority referral', reviewStatus: 'Not reviewed', screenDate: '01 Sep 2026' },
  { id: 'DM-26003', name: 'Asha Patel', age: 49, sex: 'Female', diabetesDuration: '7 years', lastScreening: '10 Aug 2025', contact: '+91 99••• 26910', grade: 2, status: 'Review needed', reviewStatus: 'Signed off', screenDate: '15 Aug 2026' },
  { id: 'DM-26004', name: 'Suresh Yadav', age: 55, sex: 'Male', diabetesDuration: '6 years', lastScreening: 'First screening', contact: '+91 96••• 54188', grade: 0, status: 'No referral', reviewStatus: 'Signed off', screenDate: '09 Aug 2026' },
  { id: 'DM-26005', name: 'Kavita Singh', age: 45, sex: 'Female', diabetesDuration: '5 years', lastScreening: '20 Aug 2025', contact: '+91 98••• 81326', grade: 1, status: 'No referral', reviewStatus: 'Signed off', screenDate: '02 Aug 2026' },
]

export const gradeOptions = Object.values(gradeDetails)

export const today = '01 Sep 2026'

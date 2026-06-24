import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import api from '../../lib/api'

export default function DoctorPatient() {
    const { id } = useParams()
    const [data, setData] = useState(null)

    const load = () => api.get(`/doctor/patients/${id}/reports`).then((r) => setData(r.data)).catch(() => setData({ reports: [] }))
    useEffect(() => { load() }, [id])

    if (!data) return <div className="text-slate-400 italic">Loading…</div>

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            <Link to="/" className="inline-flex items-center gap-1 text-slate-500 hover:text-slate-700 text-sm">
                <ArrowLeft size={16} /> Back to patients
            </Link>

            <div className="glass p-6 rounded-3xl border border-slate-200">
                <h2 className="text-2xl font-extrabold text-slate-800">{data.patient?.name || 'Patient'}</h2>
                <p className="text-slate-500 mt-1">
                    {data.patient?.age || '—'} · {data.patient?.gender || '—'} · {data.patient?.phone || 'no phone'}
                </p>
                {data.patient?.medical_history && (
                    <p className="text-sm text-slate-600 mt-3 bg-slate-50 rounded-lg px-3 py-2">
                        <strong>History:</strong> {data.patient.medical_history}
                    </p>
                )}
            </div>

            <h3 className="font-bold text-slate-700">Screening reports</h3>
            {data.reports.length === 0 ? (
                <p className="text-slate-400 italic">No reports for this patient.</p>
            ) : (
                data.reports.map((r) => <ReportCard key={r.id} report={r} onSaved={load} />)
            )}
        </div>
    )
}

function ReportCard({ report, onSaved }) {
    const [notes, setNotes] = useState(report.doctor_notes || '')
    const [saving, setSaving] = useState(false)

    const save = async () => {
        setSaving(true)
        try {
            await api.post(`/doctor/reports/${report.id}/notes`, { notes })
            onSaved()
        } finally {
            setSaving(false)
        }
    }

    return (
        <div className="glass p-6 rounded-2xl border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${report.label === "Parkinson's" ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
                    {report.label} · {Math.round(report.pd_probability * 100)}%
                </span>
                <span className="text-slate-400 text-sm">{new Date(report.created_at).toLocaleString()}</span>
            </div>
            <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add diagnosis notes / recommendations…"
                rows={3}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none text-sm"
            />
            <div className="flex items-center gap-3">
                <button onClick={save} disabled={saving}
                    className="bg-blue-600 text-white font-semibold px-5 py-2 rounded-xl hover:bg-blue-700 disabled:opacity-50 text-sm">
                    {saving ? 'Saving…' : report.reviewed ? 'Update note' : 'Save note'}
                </button>
                {report.reviewed && <span className="text-green-600 text-xs font-semibold">✓ Reviewed</span>}
            </div>
        </div>
    )
}

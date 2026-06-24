import React, { useEffect, useState } from 'react'
import api from '../../lib/api'

const STATUS_STYLES = {
    Pending: 'bg-amber-100 text-amber-700',
    Approved: 'bg-blue-100 text-blue-700',
    Completed: 'bg-green-100 text-green-700',
    Cancelled: 'bg-slate-200 text-slate-500',
}

// Allowed transitions surfaced as buttons per current status.
const ACTIONS = {
    Pending: [['Approved', 'Approve', 'bg-blue-600'], ['Cancelled', 'Decline', 'bg-slate-400']],
    Approved: [['Completed', 'Mark completed', 'bg-green-600'], ['Cancelled', 'Cancel', 'bg-slate-400']],
    Completed: [],
    Cancelled: [],
}

export default function DoctorAppointments() {
    const [appts, setAppts] = useState(null)

    const load = () => api.get('/doctor/appointments').then((r) => setAppts(r.data.appointments)).catch(() => setAppts([]))
    useEffect(() => { load() }, [])

    const setStatus = async (id, status) => {
        await api.post(`/doctor/appointments/${id}/status`, { status })
        load()
    }

    if (appts === null) return <div className="text-slate-400 italic">Loading appointments…</div>

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Appointments</h2>
                <p className="text-slate-500 mt-2">Manage patient consultation requests</p>
            </header>

            {appts.length === 0 ? (
                <div className="glass p-10 rounded-3xl text-center text-slate-400 border border-slate-200">
                    No appointment requests yet.
                </div>
            ) : (
                <div className="space-y-3">
                    {appts.map((a) => (
                        <div key={a.id} className="glass p-5 rounded-2xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                            <div>
                                <div className="font-semibold text-slate-800">{a.patient_name}</div>
                                <div className="text-sm text-slate-500 mt-1">{a.date} at {a.time}</div>
                                {a.reason && <div className="text-sm text-slate-400 mt-1 italic">{a.reason}</div>}
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`px-3 py-1 rounded-full text-xs font-bold ${STATUS_STYLES[a.status]}`}>{a.status}</span>
                                {ACTIONS[a.status].map(([status, label, color]) => (
                                    <button key={status} onClick={() => setStatus(a.id, status)}
                                        className={`${color} text-white text-xs font-semibold px-3 py-1.5 rounded-lg hover:opacity-90`}>
                                        {label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

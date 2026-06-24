import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, Users } from 'lucide-react'
import api from '../../lib/api'
import { useAuth } from '../../context/AuthContext'

export default function DoctorDashboard() {
    const { user } = useAuth()
    const navigate = useNavigate()
    const [patients, setPatients] = useState(null)
    const verified = user?.profile?.verified

    useEffect(() => {
        if (!verified) return
        api.get('/doctor/patients').then((r) => setPatients(r.data.patients)).catch(() => setPatients([]))
    }, [verified])

    if (!verified) {
        return (
            <div className="max-w-xl mx-auto mt-16 glass border border-slate-200 rounded-3xl p-10 text-center">
                <div className="w-16 h-16 bg-amber-50 rounded-full flex items-center justify-center mx-auto">
                    <Clock className="text-amber-500" size={28} />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mt-6">Awaiting approval</h2>
                <p className="text-slate-500 mt-3">
                    Your doctor account is pending verification by an administrator. You'll be able to
                    review patients and reports once approved.
                </p>
            </div>
        )
    }

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Patients</h2>
                <p className="text-slate-500 mt-2">Review patient voice screenings and add notes</p>
            </header>

            {patients === null ? (
                <div className="text-slate-400 italic">Loading patients…</div>
            ) : patients.length === 0 ? (
                <div className="glass p-10 rounded-3xl text-center text-slate-400 border border-slate-200">
                    <Users className="mx-auto mb-3 text-slate-300" size={32} /> No patients have registered yet.
                </div>
            ) : (
                <div className="glass rounded-3xl border border-slate-200 overflow-hidden">
                    <table className="w-full text-sm">
                        <thead className="bg-slate-100 text-slate-500 uppercase text-xs">
                            <tr>
                                <th className="text-left px-5 py-3">Patient</th>
                                <th className="text-left px-5 py-3">Age / Gender</th>
                                <th className="text-left px-5 py-3">Reports</th>
                                <th className="text-left px-5 py-3">Latest</th>
                                <th className="px-5 py-3"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {patients.map((p) => (
                                <tr key={p.patient_id} className="border-t border-slate-100 hover:bg-slate-50">
                                    <td className="px-5 py-3 font-medium text-slate-700">{p.name || '—'}</td>
                                    <td className="px-5 py-3 text-slate-500">{p.age || '—'} / {p.gender || '—'}</td>
                                    <td className="px-5 py-3 text-slate-500">{p.report_count}</td>
                                    <td className="px-5 py-3">
                                        {p.latest_label ? (
                                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${p.latest_label === "Parkinson's" ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
                                                {p.latest_label} ({Math.round((p.latest_pd_probability || 0) * 100)}%)
                                            </span>
                                        ) : <span className="text-slate-300">—</span>}
                                    </td>
                                    <td className="px-5 py-3 text-right">
                                        <button onClick={() => navigate(`/patients/${p.patient_id}`)}
                                            className="text-blue-600 font-semibold hover:underline">Open</button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}

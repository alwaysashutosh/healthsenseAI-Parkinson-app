import React, { useEffect, useState } from 'react'
import { Users, Stethoscope, Activity, AlertTriangle } from 'lucide-react'
import api from '../../lib/api'

export default function AdminDashboard() {
    const [stats, setStats] = useState(null)
    const [doctors, setDoctors] = useState([])

    const load = () => {
        api.get('/admin/stats').then((r) => setStats(r.data)).catch(() => { })
        api.get('/admin/doctors').then((r) => setDoctors(r.data.doctors)).catch(() => { })
    }
    useEffect(() => { load() }, [])

    const setVerified = async (id, approve) => {
        await api.post(`/admin/doctors/${id}/${approve ? 'approve' : 'revoke'}`)
        load()
    }

    const cards = stats ? [
        { label: 'Patients', value: stats.patients, icon: <Users className="text-blue-500" /> },
        { label: 'Doctors', value: stats.doctors, icon: <Stethoscope className="text-green-500" /> },
        { label: 'Predictions', value: stats.predictions, icon: <Activity className="text-purple-500" /> },
        { label: 'PD flagged', value: stats.parkinsons_flagged, icon: <AlertTriangle className="text-red-500" /> },
    ] : []

    return (
        <div className="max-w-5xl mx-auto space-y-8">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Admin Console</h2>
                <p className="text-slate-500 mt-2">Approve doctors and monitor the platform</p>
            </header>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                {cards.map((c, i) => (
                    <div key={i} className="glass p-6 rounded-2xl border border-slate-200 flex items-center gap-4">
                        <div className="p-3 bg-slate-100 rounded-xl">{c.icon}</div>
                        <div>
                            <p className="text-sm text-slate-500 font-medium">{c.label}</p>
                            <p className="text-2xl font-bold text-slate-900">{c.value}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="glass rounded-3xl border border-slate-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 font-bold text-slate-700">
                    Doctor verification {stats?.doctors_pending ? <span className="ml-2 text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-full">{stats.doctors_pending} pending</span> : null}
                </div>
                {doctors.length === 0 ? (
                    <div className="p-8 text-center text-slate-400">No doctors registered yet.</div>
                ) : (
                    <table className="w-full text-sm">
                        <thead className="bg-slate-50 text-slate-500 uppercase text-xs">
                            <tr>
                                <th className="text-left px-6 py-3">Name</th>
                                <th className="text-left px-6 py-3">Reg. No.</th>
                                <th className="text-left px-6 py-3">Specialization</th>
                                <th className="text-left px-6 py-3">Status</th>
                                <th className="px-6 py-3"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {doctors.map((d) => (
                                <tr key={d.doctor_id} className="border-t border-slate-100">
                                    <td className="px-6 py-3 font-medium text-slate-700">{d.name}</td>
                                    <td className="px-6 py-3 text-slate-500">{d.registration_number || '—'}</td>
                                    <td className="px-6 py-3 text-slate-500">{d.specialization || '—'}</td>
                                    <td className="px-6 py-3">
                                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${d.verified ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-700'}`}>
                                            {d.verified ? 'Verified' : 'Pending'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-3 text-right">
                                        {d.verified ? (
                                            <button onClick={() => setVerified(d.doctor_id, false)} className="text-red-500 font-semibold hover:underline">Revoke</button>
                                        ) : (
                                            <button onClick={() => setVerified(d.doctor_id, true)} className="text-blue-600 font-semibold hover:underline">Approve</button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}

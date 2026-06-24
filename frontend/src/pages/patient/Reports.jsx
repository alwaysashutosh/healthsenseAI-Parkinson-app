import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../../lib/api'

export default function Reports() {
    const [rows, setRows] = useState(null)

    useEffect(() => {
        api.get('/patient/reports').then((r) => setRows(r.data.reports)).catch(() => setRows([]))
    }, [])

    if (rows === null) return <div className="text-slate-400 italic">Loading reports…</div>

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">My Reports</h2>
                <p className="text-slate-500 mt-2">Your voice screening history</p>
            </header>

            {rows.length === 0 ? (
                <div className="glass p-10 rounded-3xl text-center text-slate-400 border border-slate-200">
                    No screenings yet. <Link to="/test" className="text-blue-600 font-semibold">Take your first test →</Link>
                </div>
            ) : (
                <div className="space-y-4">
                    {rows.map((r) => (
                        <div key={r.id} className="glass p-6 rounded-2xl border border-slate-200 flex items-center justify-between">
                            <div>
                                <span className={`px-3 py-1 rounded-full text-xs font-bold ${r.label === "Parkinson's" ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
                                    {r.label}
                                </span>
                                <p className="text-slate-400 text-sm mt-2">{new Date(r.created_at).toLocaleString()} · {r.source}</p>
                                {r.doctor_notes && (
                                    <p className="text-sm text-slate-600 mt-2 bg-blue-50 rounded-lg px-3 py-2">
                                        <strong>Doctor's note:</strong> {r.doctor_notes}
                                    </p>
                                )}
                            </div>
                            <div className="text-right">
                                <div className="text-2xl font-black text-slate-700">{Math.round(r.pd_probability * 100)}%</div>
                                <div className="text-xs text-slate-400">PD probability</div>
                                {r.reviewed && <div className="text-xs text-green-600 font-semibold mt-1">✓ Reviewed</div>}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

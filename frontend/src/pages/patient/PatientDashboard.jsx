import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Mic, FileText, ArrowRight, Activity } from 'lucide-react'
import api from '../../lib/api'
import { useAuth } from '../../context/AuthContext'

export default function PatientDashboard() {
    const { user } = useAuth()
    const [reports, setReports] = useState([])

    useEffect(() => {
        api.get('/patient/reports').then((r) => setReports(r.data.reports)).catch(() => { })
    }, [])

    const latest = reports[0]

    return (
        <div className="max-w-5xl mx-auto space-y-8">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Welcome, {user?.name || user?.username}</h2>
                <p className="text-slate-500 mt-2">Run a voice screening or review your past results.</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Link to="/test" className="glass p-8 rounded-3xl border border-slate-200 hover:shadow-md transition group">
                    <div className="p-3 bg-blue-50 rounded-xl w-fit"><Mic className="text-blue-600" /></div>
                    <h3 className="font-bold text-slate-800 text-lg mt-4">New Voice Test</h3>
                    <p className="text-slate-500 text-sm mt-1">Record or upload a sustained "ahhh" for analysis.</p>
                    <span className="inline-flex items-center gap-1 text-blue-600 font-semibold text-sm mt-4 group-hover:gap-2 transition-all">
                        Start <ArrowRight size={16} />
                    </span>
                </Link>

                <Link to="/reports" className="glass p-8 rounded-3xl border border-slate-200 hover:shadow-md transition group">
                    <div className="p-3 bg-green-50 rounded-xl w-fit"><FileText className="text-green-600" /></div>
                    <h3 className="font-bold text-slate-800 text-lg mt-4">My Reports</h3>
                    <p className="text-slate-500 text-sm mt-1">{reports.length} screening{reports.length !== 1 ? 's' : ''} on record.</p>
                    <span className="inline-flex items-center gap-1 text-green-600 font-semibold text-sm mt-4 group-hover:gap-2 transition-all">
                        View <ArrowRight size={16} />
                    </span>
                </Link>
            </div>

            {latest && (
                <div className="glass p-6 rounded-3xl border border-slate-200">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wide flex items-center gap-2">
                        <Activity size={16} /> Latest result
                    </h3>
                    <div className="flex items-center justify-between mt-4">
                        <div>
                            <span className={`text-lg font-bold ${latest.label === "Parkinson's" ? 'text-red-500' : 'text-green-500'}`}>
                                {latest.label}
                            </span>
                            <p className="text-slate-400 text-sm">{new Date(latest.created_at).toLocaleString()}</p>
                        </div>
                        <div className="text-3xl font-black text-slate-700">{Math.round(latest.pd_probability * 100)}%</div>
                    </div>
                </div>
            )}
        </div>
    )
}

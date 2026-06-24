import React, { useEffect, useState } from 'react'
import { CalendarPlus, X } from 'lucide-react'
import api from '../../lib/api'

const STATUS_STYLES = {
    Pending: 'bg-amber-100 text-amber-700',
    Approved: 'bg-blue-100 text-blue-700',
    Completed: 'bg-green-100 text-green-700',
    Cancelled: 'bg-slate-200 text-slate-500',
}

const field = "w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none"

export default function PatientAppointments() {
    const [doctors, setDoctors] = useState([])
    const [appts, setAppts] = useState([])
    const [form, setForm] = useState({ doctor_id: '', date: '', time: '', reason: '' })
    const [error, setError] = useState('')
    const [saving, setSaving] = useState(false)

    const loadAppts = () => api.get('/patient/appointments').then((r) => setAppts(r.data.appointments)).catch(() => { })

    useEffect(() => {
        api.get('/patient/doctors').then((r) => setDoctors(r.data.doctors)).catch(() => { })
        loadAppts()
    }, [])

    const change = (e) => setForm({ ...form, [e.target.name]: e.target.value })

    const book = async (e) => {
        e.preventDefault()
        setError(''); setSaving(true)
        try {
            await api.post('/patient/appointments', { ...form, doctor_id: Number(form.doctor_id) })
            setForm({ doctor_id: '', date: '', time: '', reason: '' })
            loadAppts()
        } catch (err) {
            setError(err.response?.data?.error || 'Could not book appointment.')
        } finally {
            setSaving(false)
        }
    }

    const cancel = async (id) => {
        await api.post(`/patient/appointments/${id}/cancel`)
        loadAppts()
    }

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Appointments</h2>
                <p className="text-slate-500 mt-2">Book a consultation with a verified neurologist</p>
            </header>

            <form onSubmit={book} className="glass p-6 rounded-3xl border border-slate-200 space-y-4">
                <h3 className="font-bold text-slate-700 flex items-center gap-2"><CalendarPlus size={18} /> Book an appointment</h3>
                {error && <div className="bg-red-50 text-red-600 text-sm rounded-xl px-4 py-3">{error}</div>}
                {doctors.length === 0 ? (
                    <p className="text-sm text-slate-400 italic">No approved doctors are available to book yet.</p>
                ) : (
                    <>
                        <select name="doctor_id" value={form.doctor_id} onChange={change} required className={field}>
                            <option value="">Select a doctor…</option>
                            {doctors.map((d) => (
                                <option key={d.doctor_id} value={d.doctor_id}>
                                    {d.name} — {d.specialization || 'Neurology'}{d.hospital ? ` (${d.hospital})` : ''}
                                </option>
                            ))}
                        </select>
                        <div className="grid grid-cols-2 gap-4">
                            <input name="date" type="date" value={form.date} onChange={change} required className={field} />
                            <input name="time" type="time" value={form.time} onChange={change} required className={field} />
                        </div>
                        <textarea name="reason" value={form.reason} onChange={change} rows={2} placeholder="Reason (optional)" className={field} />
                        <button disabled={saving} className="bg-blue-600 text-white font-bold px-6 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50">
                            {saving ? 'Booking…' : 'Request appointment'}
                        </button>
                    </>
                )}
            </form>

            <div>
                <h3 className="font-bold text-slate-700 mb-4">My appointments</h3>
                {appts.length === 0 ? (
                    <p className="text-slate-400 italic">No appointments yet.</p>
                ) : (
                    <div className="space-y-3">
                        {appts.map((a) => (
                            <div key={a.id} className="glass p-5 rounded-2xl border border-slate-200 flex items-center justify-between">
                                <div>
                                    <div className="font-semibold text-slate-800">{a.doctor_name} <span className="text-slate-400 font-normal">· {a.doctor_specialization || 'Neurology'}</span></div>
                                    <div className="text-sm text-slate-500 mt-1">{a.date} at {a.time}</div>
                                    {a.reason && <div className="text-sm text-slate-400 mt-1 italic">{a.reason}</div>}
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${STATUS_STYLES[a.status]}`}>{a.status}</span>
                                    {['Pending', 'Approved'].includes(a.status) && (
                                        <button onClick={() => cancel(a.id)} title="Cancel" className="text-slate-400 hover:text-red-500"><X size={18} /></button>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}

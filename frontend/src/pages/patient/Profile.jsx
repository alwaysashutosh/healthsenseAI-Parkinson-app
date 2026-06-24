import React, { useState } from 'react'
import api from '../../lib/api'
import { useAuth } from '../../context/AuthContext'

const field = "w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none"

export default function Profile() {
    const { user, refresh } = useAuth()
    const p = user?.profile || {}
    const [form, setForm] = useState({
        name: user?.name || '',
        age: p.age || '',
        gender: p.gender || '',
        phone: p.phone || '',
        address: p.address || '',
        medical_history: p.medical_history || '',
    })
    const [saved, setSaved] = useState(false)
    const [loading, setLoading] = useState(false)

    const change = (e) => { setForm({ ...form, [e.target.name]: e.target.value }); setSaved(false) }

    const save = async (e) => {
        e.preventDefault()
        setLoading(true)
        try {
            await api.put('/patient/profile', form)
            await refresh()
            setSaved(true)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-2xl mx-auto space-y-6">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">My Profile</h2>
                <p className="text-slate-500 mt-2">Keep your details up to date for your care team</p>
            </header>

            <form onSubmit={save} className="glass p-8 rounded-3xl border border-slate-200 space-y-4">
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Full name</label>
                    <input name="name" value={form.name} onChange={change} className={field} />
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-600 mb-1">Age</label>
                        <input name="age" type="number" value={form.age} onChange={change} className={field} />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-600 mb-1">Gender</label>
                        <select name="gender" value={form.gender} onChange={change} className={field}>
                            <option value="">Select</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Phone</label>
                    <input name="phone" value={form.phone} onChange={change} className={field} />
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Address</label>
                    <input name="address" value={form.address} onChange={change} className={field} />
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Medical history</label>
                    <textarea name="medical_history" rows={3} value={form.medical_history} onChange={change} className={field} />
                </div>
                <div className="flex items-center gap-4">
                    <button disabled={loading} className="bg-blue-600 text-white font-bold px-6 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50">
                        {loading ? 'Saving…' : 'Save changes'}
                    </button>
                    {saved && <span className="text-green-600 text-sm font-semibold">✓ Saved</span>}
                </div>
            </form>
        </div>
    )
}

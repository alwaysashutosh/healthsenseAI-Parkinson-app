import React, { useState } from 'react'
import { Search, MapPin, Phone, ExternalLink, Loader2 } from 'lucide-react'
import api from '../../lib/api'
import { useAuth } from '../../context/AuthContext'

export default function FindCare() {
    const { user } = useAuth()
    const [city, setCity] = useState(user?.profile?.address || '')
    const [radius, setRadius] = useState(10)
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const search = async (e) => {
        e?.preventDefault()
        if (!city.trim()) return
        setLoading(true); setError(''); setData(null)
        try {
            const res = await api.get('/hospitals', { params: { city, radius } })
            setData(res.data)
        } catch (err) {
            setError(err.response?.data?.error || 'Could not search. Try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <header>
                <h2 className="text-3xl font-extrabold text-slate-800">Find Care</h2>
                <p className="text-slate-500 mt-2">Locate nearby hospitals & clinics (powered by OpenStreetMap)</p>
            </header>

            <form onSubmit={search} className="glass p-6 rounded-3xl border border-slate-200 flex flex-col sm:flex-row gap-3">
                <div className="flex-1 flex items-center gap-2 px-4 rounded-xl border border-slate-200 bg-white">
                    <MapPin size={18} className="text-slate-400" />
                    <input value={city} onChange={(e) => setCity(e.target.value)} placeholder="Enter your city (e.g. Pune)"
                        className="flex-1 py-3 outline-none bg-transparent" />
                </div>
                <select value={radius} onChange={(e) => setRadius(Number(e.target.value))}
                    className="px-4 py-3 rounded-xl border border-slate-200 bg-white">
                    {[5, 10, 20, 30].map((r) => <option key={r} value={r}>{r} km</option>)}
                </select>
                <button disabled={loading}
                    className="flex items-center justify-center gap-2 bg-blue-600 text-white font-bold px-6 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50">
                    {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />} Search
                </button>
            </form>

            {error && <div className="bg-red-50 text-red-600 text-sm rounded-xl px-4 py-3">{error}</div>}

            {data && (
                <>
                    <p className="text-sm text-slate-500">
                        {data.count} result{data.count !== 1 ? 's' : ''} near <strong>{data.location.resolved}</strong>
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {data.hospitals.map((h, i) => (
                            <div key={i} className="glass p-5 rounded-2xl border border-slate-200">
                                <div className="flex justify-between items-start gap-2">
                                    <h3 className="font-bold text-slate-800">{h.name}</h3>
                                    <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2 py-1 rounded-full whitespace-nowrap">{h.distance_km} km</span>
                                </div>
                                <p className="text-xs text-slate-400 capitalize mt-1">{h.type}</p>
                                {h.address && <p className="text-sm text-slate-500 mt-2">{h.address}</p>}
                                <div className="flex items-center gap-4 mt-3 text-sm">
                                    {h.phone && (
                                        <a href={`tel:${h.phone}`} className="flex items-center gap-1 text-green-600 font-semibold">
                                            <Phone size={14} /> Call
                                        </a>
                                    )}
                                    <a href={h.map_url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-blue-600 font-semibold">
                                        <ExternalLink size={14} /> Directions
                                    </a>
                                </div>
                            </div>
                        ))}
                    </div>
                    {data.count === 0 && <p className="text-slate-400 italic">No facilities found — try a larger radius.</p>}
                </>
            )}

            <p className="text-xs text-slate-400">
                To book with a neurologist on this platform, use the <strong>Appointments</strong> page.
            </p>
        </div>
    )
}

import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const field = "w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none"

export default function Register() {
    const { register } = useAuth()
    const navigate = useNavigate()
    const [role, setRole] = useState('patient')
    const [form, setForm] = useState({})
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const change = (e) => setForm({ ...form, [e.target.name]: e.target.value })

    const submit = async (e) => {
        e.preventDefault()
        setError(''); setLoading(true)
        try {
            const user = await register({ ...form, role })
            // Doctors land on a pending-approval screen via the dashboard.
            navigate('/')
            if (user.role === 'doctor') { /* dashboard shows pending state */ }
        } catch (err) {
            setError(err.response?.data?.error || 'Registration failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-900 p-4 py-10">
            <form onSubmit={submit} className="bg-white rounded-3xl shadow-2xl p-10 w-full max-w-md space-y-5">
                <div className="text-center">
                    <div className="text-4xl mb-2">🧠</div>
                    <h1 className="text-2xl font-extrabold text-slate-800">Create account</h1>
                </div>

                {/* Role toggle */}
                <div className="grid grid-cols-2 gap-2 bg-slate-100 p-1 rounded-xl">
                    {['patient', 'doctor'].map((r) => (
                        <button key={r} type="button" onClick={() => setRole(r)}
                            className={`py-2 rounded-lg font-semibold capitalize transition ${role === r ? 'bg-white shadow text-blue-600' : 'text-slate-500'}`}>
                            {r}
                        </button>
                    ))}
                </div>

                {error && <div className="bg-red-50 text-red-600 text-sm rounded-xl px-4 py-3">{error}</div>}

                <input name="name" placeholder="Full name" onChange={change} className={field} />
                <input name="username" placeholder="Username" required minLength={3} onChange={change} className={field} />
                <input name="email" type="email" placeholder="Email (optional)" onChange={change} className={field} />
                <input name="password" type="password" placeholder="Password (min 6)" required minLength={6} onChange={change} className={field} />

                {role === 'patient' ? (
                    <div className="grid grid-cols-2 gap-3">
                        <input name="age" type="number" placeholder="Age" onChange={change} className={field} />
                        <select name="gender" onChange={change} className={field} defaultValue="">
                            <option value="" disabled>Gender</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                ) : (
                    <>
                        <input name="registration_number" placeholder="Medical registration no." onChange={change} className={field} />
                        <input name="specialization" placeholder="Specialization (e.g. Neurology)" onChange={change} className={field} />
                        <div className="grid grid-cols-2 gap-3">
                            <input name="hospital" placeholder="Hospital" onChange={change} className={field} />
                            <input name="experience" type="number" placeholder="Years exp." onChange={change} className={field} />
                        </div>
                        <p className="text-xs text-amber-600 bg-amber-50 rounded-lg px-3 py-2">
                            Doctor accounts require admin approval before access is granted.
                        </p>
                    </>
                )}

                <button disabled={loading}
                    className="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 transition disabled:opacity-50">
                    {loading ? 'Creating…' : 'Register'}
                </button>
                <p className="text-center text-sm text-slate-500">
                    Have an account? <Link to="/login" className="text-blue-600 font-semibold">Sign in</Link>
                </p>
            </form>
        </div>
    )
}

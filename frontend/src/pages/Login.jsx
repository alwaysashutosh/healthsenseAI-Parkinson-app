import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
    const { login } = useAuth()
    const navigate = useNavigate()
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const submit = async (e) => {
        e.preventDefault()
        setError(''); setLoading(true)
        try {
            await login(username, password)
            navigate('/')
        } catch (err) {
            setError(err.response?.data?.error || 'Login failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-900 p-4">
            <form onSubmit={submit} className="bg-white rounded-3xl shadow-2xl p-10 w-full max-w-md space-y-6">
                <div className="text-center">
                    <div className="text-4xl mb-2">🧠</div>
                    <h1 className="text-2xl font-extrabold text-slate-800">NeuroVoice</h1>
                    <p className="text-slate-500 text-sm mt-1">Sign in to your account</p>
                </div>
                {error && <div className="bg-red-50 text-red-600 text-sm rounded-xl px-4 py-3">{error}</div>}
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Username</label>
                    <input value={username} onChange={(e) => setUsername(e.target.value)} required
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1">Password</label>
                    <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
                <button disabled={loading}
                    className="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 transition disabled:opacity-50">
                    {loading ? 'Signing in…' : 'Sign In'}
                </button>
                <p className="text-center text-sm text-slate-500">
                    No account? <Link to="/register" className="text-blue-600 font-semibold">Register</Link>
                </p>
            </form>
        </div>
    )
}

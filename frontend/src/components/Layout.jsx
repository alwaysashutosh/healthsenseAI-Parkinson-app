import React from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { LayoutDashboard, Mic, FileText, User, LogOut, Stethoscope, Shield, MapPin, CalendarDays } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const NAV = {
    patient: [
        { to: '/', label: 'Dashboard', icon: <LayoutDashboard size={20} />, end: true },
        { to: '/test', label: 'New Test', icon: <Mic size={20} /> },
        { to: '/reports', label: 'My Reports', icon: <FileText size={20} /> },
        { to: '/appointments', label: 'Appointments', icon: <CalendarDays size={20} /> },
        { to: '/find-care', label: 'Find Care', icon: <MapPin size={20} /> },
        { to: '/profile', label: 'Profile', icon: <User size={20} /> },
    ],
    doctor: [
        { to: '/', label: 'Patients', icon: <Stethoscope size={20} />, end: true },
        { to: '/appointments', label: 'Appointments', icon: <CalendarDays size={20} /> },
    ],
    admin: [
        { to: '/', label: 'Admin Console', icon: <Shield size={20} />, end: true },
    ],
}

const ROLE_LABEL = { patient: 'Patient', doctor: 'Doctor', admin: 'Administrator' }

export default function Layout() {
    const { user, logout } = useAuth()
    const navigate = useNavigate()
    const items = NAV[user?.role] || []

    const handleLogout = () => { logout(); navigate('/login') }

    return (
        <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden">
            <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col shadow-xl">
                <div className="p-6 border-b border-slate-800">
                    <h1 className="text-xl font-bold text-white flex items-center gap-2">
                        <span className="p-2 bg-blue-600 rounded-lg">🧠</span> NeuroVoice
                    </h1>
                    <p className="text-xs text-slate-500 mt-2">{ROLE_LABEL[user?.role]} portal</p>
                </div>
                <nav className="flex-1 p-4 space-y-2">
                    {items.map((l) => (
                        <NavLink key={l.to} to={l.to} end={l.end}
                            className={({ isActive }) =>
                                `w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${isActive
                                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                                    : 'hover:bg-slate-800 hover:text-white'}`}>
                            {l.icon}<span className="font-medium">{l.label}</span>
                        </NavLink>
                    ))}
                </nav>
                <div className="p-4 border-t border-slate-800">
                    <div className="px-2 mb-3 text-sm text-slate-400">
                        <div className="text-white font-semibold">{user?.name || user?.username}</div>
                        <div className="text-xs capitalize">{user?.role}</div>
                    </div>
                    <button onClick={handleLogout}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-300 hover:bg-slate-800 hover:text-white transition-all">
                        <LogOut size={20} /> <span className="font-medium">Log out</span>
                    </button>
                </div>
            </aside>
            <main className="flex-1 overflow-y-auto p-4 md:p-8">
                <Outlet />
            </main>
        </div>
    )
}

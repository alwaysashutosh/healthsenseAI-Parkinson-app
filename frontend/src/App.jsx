import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import Login from './pages/Login'
import Register from './pages/Register'

import PatientDashboard from './pages/patient/PatientDashboard'
import Test from './pages/patient/Test'
import Reports from './pages/patient/Reports'
import Profile from './pages/patient/Profile'
import FindCare from './pages/patient/FindCare'
import PatientAppointments from './pages/patient/Appointments'

import DoctorDashboard from './pages/doctor/DoctorDashboard'
import DoctorPatient from './pages/doctor/DoctorPatient'
import DoctorAppointments from './pages/doctor/Appointments'

import AdminDashboard from './pages/admin/AdminDashboard'

function RequireRole({ roles, children }) {
    const { user, loading } = useAuth()
    if (loading) return <div className="h-screen flex items-center justify-center text-slate-400">Loading…</div>
    if (!user) return <Navigate to="/login" replace />
    if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />
    return children
}

function RoleHome() {
    const { user } = useAuth()
    if (user?.role === 'doctor') return <DoctorDashboard />
    if (user?.role === 'admin') return <AdminDashboard />
    return <PatientDashboard />
}

function AppointmentsHome() {
    const { user } = useAuth()
    return user?.role === 'doctor' ? <DoctorAppointments /> : <PatientAppointments />
}

export default function App() {
    return (
        <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            <Route element={<RequireRole><Layout /></RequireRole>}>
                <Route path="/" element={<RoleHome />} />
                {/* Patient */}
                <Route path="/test" element={<RequireRole roles={['patient']}><Test /></RequireRole>} />
                <Route path="/reports" element={<RequireRole roles={['patient']}><Reports /></RequireRole>} />
                <Route path="/profile" element={<RequireRole roles={['patient']}><Profile /></RequireRole>} />
                <Route path="/find-care" element={<RequireRole roles={['patient']}><FindCare /></RequireRole>} />
                {/* Shared: appointments (patient books, doctor manages) */}
                <Route path="/appointments" element={<RequireRole roles={['patient', 'doctor']}><AppointmentsHome /></RequireRole>} />
                {/* Doctor */}
                <Route path="/patients/:id" element={<RequireRole roles={['doctor']}><DoctorPatient /></RequireRole>} />
            </Route>

            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    )
}

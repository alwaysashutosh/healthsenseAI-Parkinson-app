import React, { createContext, useContext, useState, useEffect } from 'react'
import api from '../lib/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const token = localStorage.getItem('token')
        if (!token) { setLoading(false); return }
        api.get('/auth/me')
            .then((res) => setUser(res.data.user))
            .catch(() => localStorage.removeItem('token'))
            .finally(() => setLoading(false))
    }, [])

    const persist = (data) => {
        localStorage.setItem('token', data.token)
        setUser(data.user)
        return data.user
    }

    const login = async (username, password) => {
        const res = await api.post('/auth/login', { username, password })
        return persist(res.data)
    }

    const register = async (payload) => {
        const res = await api.post('/auth/register', payload)
        return persist(res.data)
    }

    const logout = () => {
        localStorage.removeItem('token')
        setUser(null)
    }

    // Refresh current user (e.g. after profile update or doctor approval).
    const refresh = async () => {
        const res = await api.get('/auth/me')
        setUser(res.data.user)
        return res.data.user
    }

    return (
        <AuthContext.Provider value={{ user, loading, login, register, logout, refresh }}>
            {children}
        </AuthContext.Provider>
    )
}

export const useAuth = () => useContext(AuthContext)

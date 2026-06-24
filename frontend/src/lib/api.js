import axios from 'axios'

// All requests go to the Flask API. In dev, Vite proxies /api -> localhost:5000.
const api = axios.create({ baseURL: '/api' })

// Attach the JWT (if any) to every request.
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token')
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
})

// On 401, clear the stale token so the app redirects to login.
api.interceptors.response.use(
    (res) => res,
    (err) => {
        if (err.response?.status === 401) {
            localStorage.removeItem('token')
        }
        return Promise.reject(err)
    }
)

export default api

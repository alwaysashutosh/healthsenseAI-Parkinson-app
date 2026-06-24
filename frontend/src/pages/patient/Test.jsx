import React, { useState, useRef, useEffect } from 'react'
import { Mic, Square, Upload, RotateCcw, Stethoscope, ArrowLeft } from 'lucide-react'
import api from '../../lib/api'
import { toWavFile } from '../../lib/wav'

export default function Test() {
    const [recordedBlob, setRecordedBlob] = useState(null)
    const [uploadedFile, setUploadedFile] = useState(null)
    const [isRecording, setIsRecording] = useState(false)
    const [seconds, setSeconds] = useState(0)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [result, setResult] = useState(null)

    const mediaRecorderRef = useRef(null)
    const chunksRef = useRef([])
    const timerRef = useRef(null)

    useEffect(() => () => clearInterval(timerRef.current), [])

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            const mr = new MediaRecorder(stream)
            mediaRecorderRef.current = mr
            chunksRef.current = []
            mr.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data)
            mr.onstop = () => {
                setRecordedBlob(new Blob(chunksRef.current, { type: 'audio/wav' }))
                setUploadedFile(null)
                stream.getTracks().forEach((t) => t.stop())
            }
            mr.start()
            setIsRecording(true); setError(''); setSeconds(0)
            timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000)
        } catch {
            setError('Could not access your microphone. Allow permission or upload a file instead.')
        }
    }

    const stopRecording = () => {
        mediaRecorderRef.current?.stop()
        setIsRecording(false)
        clearInterval(timerRef.current)
    }

    const onUpload = (e) => {
        const f = e.target.files[0]
        if (f) { setUploadedFile(f); setRecordedBlob(null); setError('') }
    }

    const reset = () => { setRecordedBlob(null); setUploadedFile(null); setSeconds(0) }
    const hasAudio = recordedBlob || uploadedFile

    const analyze = async () => {
        setLoading(true); setError('')
        const source = recordedBlob ? 'record' : 'upload'
        const raw = recordedBlob || uploadedFile
        const fd = new FormData()
        try {
            // Normalize whatever the browser captured/loaded into a real PCM WAV,
            // since MediaRecorder produces WebM/Opus that Praat cannot read.
            let file
            try {
                file = await toWavFile(raw, 'sample.wav')
            } catch {
                file = raw instanceof File ? raw : new File([raw], 'sample.wav')
            }
            fd.append('file', file, file.name || 'sample.wav')
            fd.append('source', source)
            const res = await api.post('/patient/predict', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
            setResult(res.data)
        } catch (err) {
            setError(err.response?.data?.error || 'Something went wrong. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    if (result) return <Result result={result} onAgain={() => { reset(); setResult(null) }} />

    return (
        <div className="max-w-2xl mx-auto">
            <h1 className="text-3xl font-extrabold text-slate-900 text-center">Voice Test</h1>
            <p className="text-slate-500 text-center mt-3">
                Take a breath, then say <strong>"ahhh"</strong> steadily for about 5 seconds.
            </p>

            {error && <div className="mt-6 bg-red-50 text-red-600 text-sm rounded-xl px-4 py-3">{error}</div>}

            <div className="mt-8 glass rounded-3xl border border-slate-200 p-8 space-y-6">
                <div className="flex flex-col items-center gap-4">
                    {!isRecording ? (
                        <button onClick={startRecording}
                            className="flex items-center gap-3 bg-blue-600 text-white px-10 py-5 rounded-2xl font-bold text-lg hover:bg-blue-700 transition active:scale-95 shadow-lg shadow-blue-200">
                            <Mic size={24} /> Start Recording
                        </button>
                    ) : (
                        <button onClick={stopRecording}
                            className="flex items-center gap-3 bg-red-500 text-white px-10 py-5 rounded-2xl font-bold text-lg animate-pulse">
                            <Square size={22} /> Stop ({seconds}s)
                        </button>
                    )}
                    <div className="text-slate-300 text-sm font-medium">— or —</div>
                    <label className="flex items-center gap-2 cursor-pointer text-slate-600 border-2 border-dashed border-slate-300 px-6 py-3 rounded-2xl hover:border-blue-400 hover:text-blue-600 transition">
                        <Upload size={18} /> Upload an audio file
                        <input type="file" accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*" onChange={onUpload} className="hidden" />
                    </label>
                </div>

                {hasAudio && (
                    <div className="bg-slate-50 rounded-2xl p-4 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm text-slate-600 font-medium">
                                {recordedBlob ? 'Recorded sample ready' : uploadedFile.name}
                            </span>
                            <button onClick={reset} className="text-slate-400 hover:text-red-500" title="Clear"><RotateCcw size={18} /></button>
                        </div>
                        {recordedBlob && <audio controls src={URL.createObjectURL(recordedBlob)} className="w-full" />}
                    </div>
                )}

                <button onClick={analyze} disabled={!hasAudio || loading}
                    className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white font-bold py-4 rounded-2xl hover:bg-blue-700 transition disabled:opacity-40 disabled:cursor-not-allowed">
                    <Stethoscope size={20} /> {loading ? 'Analyzing your voice…' : 'Analyze My Voice'}
                </button>
            </div>
            <p className="text-center text-xs text-slate-400 mt-6">Your audio is analyzed and not stored after processing.</p>
        </div>
    )
}

function Result({ result, onAgain }) {
    const prob = result.pd_probability
    const highRisk = result.prediction === 1
    const pct = Math.round(prob * 100)

    return (
        <div className="max-w-2xl mx-auto">
            <div className="glass rounded-3xl border border-slate-200 p-10 flex flex-col items-center gap-6 text-center">
                <div className={`w-36 h-36 rounded-full flex flex-col items-center justify-center ${highRisk ? 'bg-red-50' : 'bg-green-50'}`}>
                    <span className={`text-5xl font-black ${highRisk ? 'text-red-500' : 'text-green-500'}`}>{pct}%</span>
                    <span className="text-xs font-semibold text-slate-400 uppercase mt-1">PD markers</span>
                </div>
                <div className={`text-2xl font-bold px-6 py-2 rounded-xl text-white ${highRisk ? 'bg-red-500' : 'bg-green-500'}`}>
                    {highRisk ? "Parkinson's markers detected" : 'Low likelihood'}
                </div>
                <p className="text-slate-600 max-w-md leading-relaxed">
                    {highRisk
                        ? 'Your voice sample shows patterns associated with Parkinson\'s. This is not a diagnosis — please consult a neurologist. Your report is saved and visible to reviewing doctors.'
                        : 'Few markers associated with Parkinson\'s were found. Keep up regular check-ups, and re-test anytime.'}
                </p>
                <div className="bg-amber-50 border border-amber-100 text-amber-800 text-sm rounded-2xl px-5 py-4">
                    {result.disclaimer || 'This is a screening aid, not a medical diagnosis.'}
                </div>
                <button onClick={onAgain} className="flex items-center gap-2 text-blue-600 font-semibold hover:underline">
                    <ArrowLeft size={18} /> Test again
                </button>
            </div>
        </div>
    )
}

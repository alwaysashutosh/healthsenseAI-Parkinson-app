// Convert any browser-decodable audio (MediaRecorder WebM/Opus, mp3, m4a, ogg,
// wav...) into a real mono 16-bit PCM WAV blob. The MediaRecorder API does NOT
// produce WAV even when you label the blob "audio/wav" — it is usually WebM/Opus,
// which Praat/parselmouth cannot read. Decoding via the Web Audio API and
// re-encoding here guarantees the backend receives a genuine WAV.

function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(buffer)
    const writeStr = (off, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i))
    }
    writeStr(0, 'RIFF')
    view.setUint32(4, 36 + samples.length * 2, true)
    writeStr(8, 'WAVE')
    writeStr(12, 'fmt ')
    view.setUint32(16, 16, true)        // PCM chunk size
    view.setUint16(20, 1, true)         // PCM format
    view.setUint16(22, 1, true)         // mono
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true) // byte rate (mono, 16-bit)
    view.setUint16(32, 2, true)         // block align
    view.setUint16(34, 16, true)        // bits per sample
    writeStr(36, 'data')
    view.setUint32(40, samples.length * 2, true)
    let off = 44
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]))
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true)
        off += 2
    }
    return new Blob([view], { type: 'audio/wav' })
}

export async function toWavFile(input, filename = 'sample.wav') {
    const arrayBuf = await input.arrayBuffer()
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    const ctx = new AudioCtx()
    try {
        const audioBuf = await ctx.decodeAudioData(arrayBuf)
        const n = audioBuf.length
        const mono = new Float32Array(n)
        for (let c = 0; c < audioBuf.numberOfChannels; c++) {
            const data = audioBuf.getChannelData(c)
            for (let i = 0; i < n; i++) mono[i] += data[i] / audioBuf.numberOfChannels
        }
        const wavBlob = encodeWav(mono, audioBuf.sampleRate)
        return new File([wavBlob], filename, { type: 'audio/wav' })
    } finally {
        ctx.close()
    }
}

import { useRef, useState } from "react";
import { Link } from "react-router-dom";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export default function DetectVideo() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const onPick = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const onAnalyze = async () => {
    if (!file) {
      setError("Please choose a video file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setError(data.message || data.error || "Prediction failed.");
      } else if (data.error === "no_faces") {
        setError(data.message || "No faces detected in the video.");
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(
        "Could not reach the backend. Make sure it is running (python -m app.server)."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const isFake = result?.label === "FAKE";
  const reasons = result?.forgery_reasons?.reasons ?? [];

  return (
    <section className="max-w-5xl mx-auto px-6 py-10">
      <h2 className="text-3xl md:text-4xl font-extrabold text-center mb-8 drop-shadow-[0_0_10px_#00f5d4]">
        Detect DeepFake Video
      </h2>

      <div className="bg-[#0f182c]/70 border border-gray-700 rounded-xl p-6 shadow-[0_0_20px_#00f5d455]">
        <label className="block text-sm font-semibold text-teal-300 mb-2">
          Upload a video (.mp4, .avi, .mov, .mkv)
        </label>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={onPick}
          className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-teal-400 file:text-black file:font-semibold hover:file:bg-teal-300 cursor-pointer"
        />

        {preview && (
          <video
            src={preview}
            controls
            className="mt-4 w-full max-h-80 rounded-lg border border-gray-700"
          />
        )}

        <div className="flex flex-wrap gap-3 mt-5">
          <button
            onClick={onAnalyze}
            disabled={loading || !file}
            className="bg-teal-400 text-black font-semibold px-6 py-2.5 rounded-lg hover:bg-teal-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing..." : "Analyze Video"}
          </button>
          <button
            onClick={reset}
            disabled={loading}
            className="px-6 py-2.5 rounded-lg border border-gray-600 text-gray-300 hover:bg-gray-800 transition-all disabled:opacity-50"
          >
            Reset
          </button>
          <Link
            to="/"
            className="px-6 py-2.5 rounded-lg border border-gray-600 text-gray-300 hover:bg-gray-800 transition-all"
          >
            Back
          </Link>
        </div>

        {loading && (
          <p className="mt-4 text-sm text-gray-400 animate-pulse">
            Extracting frames and running the model…
          </p>
        )}
        {error && (
          <p className="mt-4 text-sm text-red-400 bg-red-950/40 border border-red-800 rounded-lg p-3">
            {error}
          </p>
        )}
      </div>

      {result && (
        <div className="mt-8 space-y-6">
          {/* Verdict */}
          <div
            className={`rounded-xl p-6 border text-center ${
              isFake
                ? "border-red-600 bg-red-950/40"
                : "border-green-600 bg-green-950/30"
            }`}
          >
            <p className="text-sm uppercase tracking-widest text-gray-400">
              Verdict
            </p>
            <p
              className={`text-4xl font-extrabold ${
                isFake ? "text-red-400" : "text-green-400"
              }`}
            >
              {result.label}
            </p>
            <p className="text-gray-300 mt-1">
              Confidence: <span className="font-semibold text-white">{result.confidence}%</span>
            </p>
          </div>

          {/* Top suspicious frames */}
          {result.frames?.length > 0 && (
            <div>
              <h3 className="text-xl font-semibold text-teal-300 mb-3">
                Top Suspicious Frames
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {result.frames.map((b64, i) => (
                  <div key={i} className="rounded-lg overflow-hidden border border-gray-700">
                    <img
                      src={`data:image/jpeg;base64,${b64}`}
                      alt={`Frame ${i + 1}`}
                      className="w-full h-auto"
                    />
                    {result.frame_scores?.[i] !== undefined && (
                      <p className="text-xs text-center text-gray-400 py-1">
                        score {result.frame_scores[i]}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Forgery reasoning */}
          {reasons.length > 0 && (
            <div>
              <h3 className="text-xl font-semibold text-teal-300 mb-3">
                Forgery Reasoning
              </h3>
              <ul className="space-y-3">
                {reasons.map((r, i) => (
                  <li
                    key={i}
                    className="bg-[#0f182c]/70 border border-gray-700 rounded-lg p-4"
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-semibold text-white capitalize">
                        {r.type?.replace(/_/g, " ")}
                      </span>
                      {typeof r.score === "number" && (
                        <span className="text-xs text-gray-400">
                          score {r.score.toFixed?.(3) ?? r.score}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-300">{r.reason}</p>
                    {r.evidence_frames?.length > 0 && (
                      <p className="text-xs text-gray-500 mt-1">
                        evidence frames: {r.evidence_frames.join(", ")}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

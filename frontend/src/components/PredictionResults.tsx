// PredictionResults.tsx — v2 Enhanced
import { useEffect, useState, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import {
  getLatestPredictionV2,
  type PredictionResponseV2,
  type LatencyBreakdown,
} from "../api/patient";
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// ── Constants ─────────────────────────────────────────────────────────────────

const DISEASE_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cvd: "Cardiovascular Disease",
  ckd: "Chronic Kidney Disease",
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function riskLevelColor(level: string): string {
  switch (level) {
    case "Critical": return "text-red-700";
    case "High":     return "text-orange-600";
    case "Moderate": return "text-yellow-600";
    default:         return "text-green-600";
  }
}

function riskLevelBg(level: string): string {
  switch (level) {
    case "Critical": return "bg-red-50 border-red-300";
    case "High":     return "bg-orange-50 border-orange-300";
    case "Moderate": return "bg-yellow-50 border-yellow-300";
    default:         return "bg-green-50 border-green-300";
  }
}

function riskLevelBadge(level: string): string {
  switch (level) {
    case "Critical": return "bg-red-100 text-red-800";
    case "High":     return "bg-orange-100 text-orange-800";
    case "Moderate": return "bg-yellow-100 text-yellow-800";
    default:         return "bg-green-100 text-green-800";
  }
}

function riskBarColor(level: string): string {
  switch (level) {
    case "Critical": return "bg-red-500";
    case "High":     return "bg-orange-500";
    case "Moderate": return "bg-yellow-400";
    default:         return "bg-green-500";
  }
}

function robustnessColor(score: number): string {
  if (score >= 0.85) return "text-green-700";
  if (score >= 0.65) return "text-yellow-700";
  return "text-red-700";
}

// ── Chevron icon ──────────────────────────────────────────────────────────────

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      className={`h-4 w-4 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

// ── V2 DiseasePredictionCard ──────────────────────────────────────────────────

interface DiseasePredictionCardProps {
  pred: PredictionResponseV2;
}
function DiseasePredictionCard({ pred }: DiseasePredictionCardProps) {
  const [modelDetailsOpen, setModelDetailsOpen] = useState(false);
  const [perfOpen, setPerfOpen] = useState(false);

  // SHAP bar chart data
  const shapLabels = pred.shap_features.map((f) => f.feature_name);
  const shapValues = pred.shap_features.map((f) => f.shap_value);
  const shapColors = pred.shap_features.map((f) =>
    f.direction === "increases risk"
      ? "rgba(239,68,68,0.7)"
      : "rgba(34,197,94,0.7)"
  );
  const shapBorderColors = pred.shap_features.map((f) =>
    f.direction === "increases risk" ? "rgb(239,68,68)" : "rgb(34,197,94)"
  );

  const shapChartData = {
    labels: shapLabels,
    datasets: [
      {
        label: "SHAP Value",
        data: shapValues,
        backgroundColor: shapColors,
        borderColor: shapBorderColors,
        borderWidth: 1,
      },
    ],
  };

  const shapChartOptions = {
    indexAxis: "y" as const,
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Top-3 Feature Contributions (SHAP)",
        font: { size: 12 },
      },
    },
    scales: {
      x: { title: { display: true, text: "SHAP Value" } },
    },
  };

  return (
    <div className={`rounded-lg border p-5 space-y-4 ${riskLevelBg(pred.risk_level)}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-gray-800">
          {DISEASE_LABELS[pred.disease] ?? pred.disease}
        </h3>
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${riskLevelBadge(pred.risk_level)}`}
        >
          {pred.risk_level}
        </span>
      </div>

      {/* Risk score + confidence */}
      <div className="flex items-end gap-4">
        <div>
          <p className={`text-4xl font-bold ${riskLevelColor(pred.risk_level)}`}>
            {pred.risk_score.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">Risk Score</p>
        </div>
        <div className="pb-1">
          <p className="text-lg font-semibold text-gray-700">
            {(pred.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500">Confidence</p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-2 w-full rounded-full bg-gray-200">
        <div
          className={`h-2 rounded-full ${riskBarColor(pred.risk_level)}`}
          style={{ width: `${Math.min(pred.risk_score, 100)}%` }}
          role="progressbar"
          aria-valuenow={pred.risk_score}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${DISEASE_LABELS[pred.disease] ?? pred.disease} risk score ${pred.risk_score.toFixed(1)} out of 100`}
        />
      </div>

      {/* SHAP Features */}
      <section aria-label={`SHAP features for ${DISEASE_LABELS[pred.disease] ?? pred.disease}`}>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Top Feature Contributions</h4>
        <div className="space-y-1 mb-3">
          {pred.shap_features.map((feat) => (
            <div key={feat.feature_name} className="flex items-center gap-2 text-sm">
              <span className="text-gray-700 font-medium">{feat.feature_name}</span>
              {pred.missing_flags[feat.feature_name] && (
                <span
                  className="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-800"
                  title="This value was imputed from population median"
                >
                  imputed
                </span>
              )}
              <span
                className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ${
                  feat.direction === "increases risk"
                    ? "bg-red-100 text-red-700"
                    : "bg-green-100 text-green-700"
                }`}
              >
                {feat.direction}
              </span>
              <span className="text-gray-500 text-xs ml-auto">
                {feat.shap_value >= 0 ? "+" : ""}
                {feat.shap_value.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
        <div className="rounded bg-white border border-gray-200 p-2">
          <Bar
            data={shapChartData}
            options={shapChartOptions}
            aria-label={`Horizontal bar chart of top SHAP features for ${DISEASE_LABELS[pred.disease] ?? pred.disease}`}
          />
        </div>
      </section>

      {/* Explanation */}
      {pred.explanation && (
        <section aria-label={`Explanation for ${DISEASE_LABELS[pred.disease] ?? pred.disease}`}>
          <h4 className="text-sm font-semibold text-gray-700 mb-1">Explanation</h4>
          <p className="text-sm text-gray-600 leading-relaxed">{pred.explanation}</p>
        </section>
      )}

      {/* Triggered Rules */}
      {pred.triggered_rules.length > 0 && (
        <section aria-label={`Triggered clinical rules for ${DISEASE_LABELS[pred.disease] ?? pred.disease}`}>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Triggered Clinical Rules</h4>
          <div className="flex flex-wrap gap-2">
            {pred.triggered_rules.map((rule, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-amber-100 text-amber-800 border border-amber-300"
              >
                ⚠ {rule}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Model Details — expandable */}
      <section>
        <button
          type="button"
          onClick={() => setModelDetailsOpen((o) => !o)}
          aria-expanded={modelDetailsOpen}
          aria-controls={`model-details-${pred.disease}`}
          className="flex w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <span>Model Details</span>
          <ChevronIcon open={modelDetailsOpen} />
        </button>
        {modelDetailsOpen && (
          <div
            id={`model-details-${pred.disease}`}
            className="mt-2 rounded-md border border-gray-200 bg-white p-4 space-y-3 text-sm"
          >
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Threshold Used</span>
              <span className="font-semibold text-gray-800">
                {(pred.threshold_used * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Robustness Score</span>
              <span className={`font-semibold ${robustnessColor(pred.robustness_score)}`}>
                {(pred.robustness_score * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}
      </section>

      {/* Performance — expandable */}
      <section>
        <button
          type="button"
          onClick={() => setPerfOpen((o) => !o)}
          aria-expanded={perfOpen}
          aria-controls={`perf-${pred.disease}`}
          className="flex w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <span>Performance</span>
          <ChevronIcon open={perfOpen} />
        </button>
        {perfOpen && (
          <div
            id={`perf-${pred.disease}`}
            className="mt-2 rounded-md border border-gray-200 bg-white p-4 text-sm"
          >
            <LatencyTable latency={pred.latency} />
          </div>
        )}
      </section>
    </div>
  );
}

// ── Latency Table ─────────────────────────────────────────────────────────────

interface LatencyTableProps {
  latency: LatencyBreakdown;
}

function LatencyTable({ latency }: LatencyTableProps) {
  const stages: Array<{ label: string; value: number | null | undefined }> = [
    { label: "OCR", value: latency.ocr_ms },
    { label: "Feature Extraction", value: latency.feature_extraction_ms },
    { label: "Scaler Normalization", value: latency.scaler_normalization_ms },
    { label: "Classical ML", value: latency.classical_ml_ms },
    { label: "VQC (Quantum)", value: latency.vqc_ms },
    { label: "Hybrid Fusion", value: latency.hybrid_fusion_ms },
    { label: "Clinical Rules", value: latency.clinical_rules_ms },
    { label: "XAI / SHAP", value: latency.xai_layer_ms },
  ];

  const isSlowTotal = latency.total_ms > 1000;

  return (
    <div className="space-y-2">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-1 text-gray-500 font-medium">Stage</th>
            <th className="text-right py-1 text-gray-500 font-medium">Duration (ms)</th>
          </tr>
        </thead>
        <tbody>
          {stages.map(({ label, value }) =>
            value != null ? (
              <tr key={label} className="border-b border-gray-100">
                <td className="py-1 text-gray-600">{label}</td>
                <td className="py-1 text-right text-gray-700">{value.toFixed(1)}</td>
              </tr>
            ) : null
          )}
        </tbody>
        <tfoot>
          <tr>
            <td className="pt-2 font-semibold text-gray-800">Total</td>
            <td
              className={`pt-2 text-right font-bold ${
                isSlowTotal ? "text-red-600" : "text-gray-800"
              }`}
            >
              {latency.total_ms.toFixed(1)} ms
            </td>
          </tr>
        </tfoot>
      </table>
      {isSlowTotal && (
        <p className="text-xs text-red-600 font-medium">
          ⚠ Total inference time exceeds 1000 ms budget.
        </p>
      )}
    </div>
  );
}

// ── V2 Limitations Section ────────────────────────────────────────────────────

function LimitationsSection({ limitations }: { limitations: string[] }) {
  if (limitations.length === 0) return null;
  return (
    <section
      aria-label="Model limitations"
      className="rounded-lg border-2 border-blue-200 bg-blue-50 p-4"
    >
      <div className="flex items-start gap-2 mb-3">
        <span className="text-blue-600 text-lg leading-none" aria-hidden="true">ℹ️</span>
        <h2 className="text-sm font-semibold text-blue-800">Model Limitations</h2>
      </div>
      <ul className="space-y-1.5 pl-2">
        {limitations.map((lim, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-blue-700">
            <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-blue-400" aria-hidden="true" />
            {lim}
          </li>
        ))}
      </ul>
    </section>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function PredictionResults({
  predictionId,
  documentId,
}: {
  predictionId?: string;
  documentId?: string | null;
}) {
  const [v2Predictions, setV2Predictions] = useState<PredictionResponseV2[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Track the predictionId we last fetched for, so a new upload always re-fetches
  const lastFetchedIdRef = useRef<string | undefined>(undefined);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function stopPolling() {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  async function load(isPolling = false) {
    if (!isPolling) setLoading(true);
    setError(null);
    try {
      const v2 = await getLatestPredictionV2(documentId ?? undefined);
      if (v2 && v2.length > 0) {
        setV2Predictions(v2);
        setPolling(false);
        stopPolling();
      } else {
        setV2Predictions(null);
        if (isPolling) {
          pollTimerRef.current = setTimeout(() => load(true), 3000);
        }
      }
    } catch {
      setError("Failed to load prediction results.");
      setPolling(false);
      stopPolling();
    } finally {
      if (!isPolling) setLoading(false);
    }
  }

  useEffect(() => {
    // Re-fetch whenever predictionId changes (new document uploaded)
    if (lastFetchedIdRef.current === predictionId && v2Predictions !== null) {
      return;
    }
    const isNewUpload = lastFetchedIdRef.current !== undefined && lastFetchedIdRef.current !== predictionId;
    lastFetchedIdRef.current = predictionId;
    stopPolling();

    if (isNewUpload) {
      // New upload — clear stale results and start polling for the new prediction
      setV2Predictions(null);
      setPolling(true);
      setLoading(false);
      // Start polling after a short delay to give the background task time to start
      pollTimerRef.current = setTimeout(() => load(true), 3000);
    } else {
      load(false);
    }

    return () => stopPolling();
  }, [predictionId]);

  // ── Loading state ──────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div role="status" aria-live="polite" className="flex items-center gap-2 py-8 text-sm text-gray-500">
        <svg className="h-5 w-5 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24" aria-hidden="true">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        Loading results…
      </div>
    );
  }

  if (error) {
    return <p role="alert" className="text-sm text-red-600">{error}</p>;
  }

  // ── No results / polling state ─────────────────────────────────────────────

  if (!v2Predictions) {
    if (polling) {
      return (
        <div role="status" aria-live="polite" className="flex flex-col items-center gap-3 py-10">
          <svg className="h-7 w-7 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24" aria-hidden="true">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <p className="text-sm text-gray-600 font-medium">Analysing your report…</p>
          <p className="text-xs text-gray-400">This usually takes 5–15 seconds. Results will appear automatically.</p>
        </div>
      );
    }

    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center space-y-3">
        <svg
          className="mx-auto h-10 w-10 text-gray-300"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z"
          />
        </svg>
        <p className="text-sm font-medium text-gray-600">No predictions yet</p>
        <p className="text-xs text-gray-400">
          Complete your health profile and upload a medical report to get your risk assessment.
        </p>
        <button
          onClick={() => { lastFetchedIdRef.current = undefined; load(false); }}
          className="mt-2 inline-flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 underline"
          aria-label="Refresh prediction results"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Check again
        </button>
      </div>
    );
  }

  // ── V2 Render ──────────────────────────────────────────────────────────────

  if (v2Predictions && v2Predictions.length > 0) {
    // Collect shared disclaimer and limitations from first prediction
    const disclaimer = v2Predictions[0].disclaimer;
    const limitations = v2Predictions[0].limitations;

    return (
      <div className="space-y-6">
        {/* Non-dismissible disclaimer banner — Req 22.2 */}
        <div
          role="alert"
          aria-live="assertive"
          className="flex items-start gap-3 rounded-lg border border-yellow-400 bg-yellow-50 px-4 py-3"
        >
          <svg
            className="h-5 w-5 flex-shrink-0 text-yellow-600 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
            />
          </svg>
          <p className="text-sm font-medium text-yellow-800">{disclaimer}</p>
        </div>

        {/* Risk Score Cards — one per disease */}
        <section aria-label="Risk scores per disease">
          <h2 className="text-base font-semibold text-gray-800 mb-3">Risk Assessment</h2>
          <div className="space-y-6">
            {v2Predictions.map((pred) => (
              <DiseasePredictionCard key={pred.disease} pred={pred} />
            ))}
          </div>
        </section>

        {/* Limitations — Req 11.4 */}
        <LimitationsSection limitations={limitations} />
      </div>
    );
  }

  return null;
}

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
  getLatestPrediction,
  getExplanation,
  getRecommendations,
  type PredictionResult,
  type SHAPExplanation,
  type Recommendation,
} from "../api/patient";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const DISEASE_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cvd: "Cardiovascular Disease",
  ckd: "Chronic Kidney Disease",
};

function riskColor(score: number): string {
  if (score >= 75) return "text-red-600";
  if (score >= 30) return "text-yellow-600";
  return "text-green-600";
}

function riskBg(score: number): string {
  if (score >= 75) return "bg-red-50 border-red-200";
  if (score >= 30) return "bg-yellow-50 border-yellow-200";
  return "bg-green-50 border-green-200";
}

interface ScoreCardProps {
  disease: string;
  quantum: number | null;
  classical: number | null;
}

function ScoreCard({ disease, quantum, classical }: ScoreCardProps) {
  const primary = quantum ?? classical ?? 0;
  return (
    <div className={`rounded-lg border p-4 ${riskBg(primary)}`}>
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        {DISEASE_LABELS[disease] ?? disease}
      </h3>
      <div className="flex gap-6">
        {quantum !== null && (
          <div className="text-center">
            <p className={`text-3xl font-bold ${riskColor(quantum)}`}>
              {quantum.toFixed(1)}
            </p>
            <p className="text-xs text-gray-500 mt-1">Quantum</p>
          </div>
        )}
        {classical !== null && (
          <div className="text-center">
            <p className={`text-3xl font-bold ${riskColor(classical)}`}>
              {classical.toFixed(1)}
            </p>
            <p className="text-xs text-gray-500 mt-1">Classical</p>
          </div>
        )}
      </div>
      <div className="mt-3 h-2 w-full rounded-full bg-gray-200">
        <div
          className={`h-2 rounded-full ${
            primary >= 75 ? "bg-red-500" : primary >= 30 ? "bg-yellow-400" : "bg-green-500"
          }`}
          style={{ width: `${Math.min(primary, 100)}%` }}
          role="progressbar"
          aria-valuenow={primary}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${DISEASE_LABELS[disease] ?? disease} risk ${primary.toFixed(1)}%`}
        />
      </div>
    </div>
  );
}

export default function PredictionResults() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [explanation, setExplanation] = useState<SHAPExplanation | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    async function load() {
      try {
        const pred = await getLatestPrediction();
        if (!pred) {
          setLoading(false);
          return;
        }
        setPrediction(pred);

        const [expl, recs] = await Promise.allSettled([
          getExplanation(pred.id),
          getRecommendations(pred.id),
        ]);
        if (expl.status === "fulfilled") setExplanation(expl.value);
        if (recs.status === "fulfilled") setRecommendations(recs.value);
      } catch {
        setError("Failed to load prediction results.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

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

  if (!prediction) {
    return (
      <p className="text-sm text-gray-500">
        No prediction results yet. Upload a medical report to get started.
      </p>
    );
  }

  const diseases = Object.keys(prediction.risk_scores) as Array<
    keyof typeof prediction.risk_scores
  >;

  // SHAP waterfall chart data
  const shapChartData = explanation
    ? {
        labels: explanation.waterfall_data.labels,
        datasets: [
          {
            label: "SHAP Contribution",
            data: explanation.waterfall_data.values,
            backgroundColor: explanation.waterfall_data.values.map((v) =>
              v >= 0 ? "rgba(239,68,68,0.7)" : "rgba(34,197,94,0.7)"
            ),
            borderColor: explanation.waterfall_data.values.map((v) =>
              v >= 0 ? "rgb(239,68,68)" : "rgb(34,197,94)"
            ),
            borderWidth: 1,
          },
        ],
      }
    : null;

  const shapChartOptions = {
    indexAxis: "y" as const,
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Feature Contributions (SHAP Waterfall)",
        font: { size: 13 },
      },
    },
    scales: {
      x: { title: { display: true, text: "SHAP Value" } },
    },
  };

  return (
    <div className="space-y-6">
      {/* Fallback notice */}
      {prediction.model_used === "classical" && (
        <div
          role="status"
          className="rounded-md bg-yellow-50 border border-yellow-200 px-4 py-3 text-sm text-yellow-800"
        >
          Classical prediction used — quantum engine was unavailable.
        </div>
      )}

      {/* Risk Score Cards */}
      <section aria-label="Risk scores">
        <h2 className="text-base font-semibold text-gray-800 mb-3">Risk Scores</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {diseases.map((disease) => (
            <ScoreCard
              key={disease}
              disease={disease}
              quantum={prediction.quantum_scores?.[disease] ?? null}
              classical={prediction.classical_scores?.[disease] ?? null}
            />
          ))}
        </div>
      </section>

      {/* SHAP Waterfall Chart */}
      {shapChartData && (
        <section aria-label="SHAP feature importance chart">
          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <Bar data={shapChartData} options={shapChartOptions} />
          </div>
        </section>
      )}

      {/* LLM Explanation */}
      {explanation?.summary && (
        <section aria-label="AI explanation">
          <h2 className="text-base font-semibold text-gray-800 mb-2">Explanation</h2>
          <div className="rounded-lg border border-gray-200 bg-white p-4 text-sm text-gray-700 leading-relaxed">
            {explanation.summary}
          </div>
        </section>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <section aria-label="Personalized recommendations">
          <h2 className="text-base font-semibold text-gray-800 mb-3">Recommendations</h2>
          <ul className="space-y-3">
            {recommendations.map((rec, i) => (
              <li
                key={i}
                className={`rounded-lg border p-4 text-sm ${
                  rec.requires_physician
                    ? "border-red-200 bg-red-50"
                    : "border-gray-200 bg-white"
                }`}
              >
                <div className="flex items-start gap-2">
                  {rec.requires_physician && (
                    <span
                      aria-label="Physician consultation required"
                      className="mt-0.5 inline-block h-2 w-2 flex-shrink-0 rounded-full bg-red-500"
                    />
                  )}
                  <div>
                    <span className="font-medium text-gray-700 capitalize">
                      {DISEASE_LABELS[rec.disease] ?? rec.disease}:{" "}
                    </span>
                    <span className="text-gray-600">{rec.text}</span>
                    {rec.requires_physician && (
                      <p className="mt-1 text-xs font-medium text-red-600">
                        Consult a physician
                      </p>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

import { useEffect, useState, useCallback } from "react";
import { useAuthContext } from "../context/AuthContext";
import {
  getHighRiskPatients,
  getPatientDetail,
  updatePatientStatus,
  bulkExport,
  generateReport,
  getReportDownloadUrl,
  type HighRiskPatient,
  type PatientDetail,
} from "../api/clinical";

// ── Helpers ───────────────────────────────────────────────────────────────────

function riskColor(score: number): string {
  if (score >= 75) return "text-red-600 font-semibold";
  if (score >= 50) return "text-yellow-600 font-semibold";
  return "text-green-600";
}

function RiskBadge({ score }: { score: number }) {
  const bg =
    score >= 75 ? "bg-red-100 text-red-700" :
    score >= 50 ? "bg-yellow-100 text-yellow-700" :
    "bg-green-100 text-green-700";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${bg}`}>
      {score.toFixed(1)}%
    </span>
  );
}

// ── Patient list ──────────────────────────────────────────────────────────────

interface PatientListProps {
  patients: HighRiskPatient[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onView: (id: string) => void;
  onDownloadPDF: (id: string) => void;
  onBulkExport: () => void;
  onBulkStatusUpdate: (status: string) => void;
  loading: boolean;
}

function PatientList({
  patients,
  selected,
  onToggle,
  onSelectAll,
  onView,
  onDownloadPDF,
  onBulkExport,
  onBulkStatusUpdate,
  loading,
}: PatientListProps) {
  const allSelected = patients.length > 0 && selected.size === patients.length;

  return (
    <div>
      {/* Bulk action bar */}
      {selected.size > 0 && (
        <div className="mb-3 flex items-center gap-3 rounded-lg bg-blue-50 border border-blue-200 px-4 py-2">
          <span className="text-sm text-blue-700 font-medium">
            {selected.size} selected
          </span>
          <button
            onClick={onBulkExport}
            className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
          >
            Export selected
          </button>
          <button
            onClick={() => onBulkStatusUpdate("reviewed")}
            className="text-sm bg-gray-600 text-white px-3 py-1 rounded hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 transition"
          >
            Mark reviewed
          </button>
          <button
            onClick={() => onBulkStatusUpdate("follow_up")}
            className="text-sm bg-yellow-600 text-white px-3 py-1 rounded hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 transition"
          >
            Mark follow-up
          </button>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={onSelectAll}
                  aria-label="Select all patients"
                  className="rounded border-gray-300"
                />
              </th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">Patient</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">Diabetes</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">CVD</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">CKD</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">Alerts</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">Status</th>
              <th className="px-4 py-3 text-left font-medium text-gray-600">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {loading ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-400">
                  Loading patients…
                </td>
              </tr>
            ) : patients.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-400">
                  No high-risk patients found.
                </td>
              </tr>
            ) : (
              patients.map((p) => (
                <tr key={p.id} className="hover:bg-gray-50 transition">
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selected.has(p.id)}
                      onChange={() => onToggle(p.id)}
                      aria-label={`Select ${p.email}`}
                      className="rounded border-gray-300"
                    />
                  </td>
                  <td className="px-4 py-3 text-gray-800">{p.email}</td>
                  <td className="px-4 py-3">
                    <RiskBadge score={p.risk_scores.diabetes} />
                  </td>
                  <td className="px-4 py-3">
                    <RiskBadge score={p.risk_scores.cvd} />
                  </td>
                  <td className="px-4 py-3">
                    <RiskBadge score={p.risk_scores.ckd} />
                  </td>
                  <td className="px-4 py-3">
                    {p.alerts.length > 0 ? (
                      <span className="inline-flex items-center gap-1 text-red-600 text-xs font-medium">
                        <span aria-hidden="true">⚠</span>
                        {p.alerts.length} alert{p.alerts.length > 1 ? "s" : ""}
                      </span>
                    ) : (
                      <span className="text-gray-400 text-xs">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className="inline-block px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600 capitalize">
                      {p.status || "active"}
                    </span>
                  </td>
                  <td className="px-4 py-3 flex gap-2">
                    <button
                      onClick={() => onView(p.id)}
                      aria-label={`View details for ${p.email}`}
                      className="text-xs text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
                    >
                      View
                    </button>
                    <button
                      onClick={() => onDownloadPDF(p.id)}
                      aria-label={`Download PDF for ${p.email}`}
                      className="text-xs text-gray-600 hover:underline focus:outline-none focus:ring-2 focus:ring-gray-400 rounded"
                    >
                      PDF
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Patient detail ────────────────────────────────────────────────────────────

interface PatientDetailViewProps {
  patientId: string;
  onBack: () => void;
  onDownloadPDF: (id: string) => void;
}

function PatientDetailView({ patientId, onBack, onDownloadPDF }: PatientDetailViewProps) {
  const [detail, setDetail] = useState<PatientDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusInput, setStatusInput] = useState("");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  useEffect(() => {
    const start = Date.now();
    setLoading(true);
    setError(null);
    getPatientDetail(patientId)
      .then((d) => {
        // Warn if load exceeded 3s (Req 10.3)
        if (Date.now() - start > 3000) {
          console.warn("Patient detail load exceeded 3 seconds");
        }
        setDetail(d);
        setStatusInput(d.status || "");
      })
      .catch(() => setError("Failed to load patient details."))
      .finally(() => setLoading(false));
  }, [patientId]);

  async function handleStatusUpdate() {
    if (!statusInput.trim()) return;
    try {
      const res = await updatePatientStatus(patientId, statusInput.trim());
      setStatusMsg(res.message || "Status updated.");
    } catch {
      setStatusMsg("Failed to update status.");
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-gray-400">
        Loading patient details…
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700">
        {error ?? "Patient not found."}
        <button onClick={onBack} className="ml-4 text-sm underline text-red-600">
          Back
        </button>
      </div>
    );
  }

  const latest = detail.prediction_history[0] ?? null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={onBack}
            className="text-sm text-blue-600 hover:underline mb-1 flex items-center gap-1"
            aria-label="Back to patient list"
          >
            ← Back
          </button>
          <h2 className="text-xl font-bold text-gray-900">{detail.email}</h2>
          <span className="text-sm text-gray-500 capitalize">Status: {detail.status || "active"}</span>
        </div>
        <button
          onClick={() => onDownloadPDF(patientId)}
          className="bg-blue-600 text-white text-sm px-4 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
        >
          Download PDF
        </button>
      </div>

      {/* Alerts */}
      {detail.alerts.length > 0 && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4">
          <p className="text-sm font-semibold text-red-700 mb-2">
            ⚠ High-risk alerts
          </p>
          <ul className="space-y-1">
            {detail.alerts.map((a, i) => (
              <li key={i} className="text-sm text-red-600">
                {a.disease}: <span className={riskColor(a.risk_score)}>{a.risk_score.toFixed(1)}%</span>
                <span className="text-gray-400 ml-2 text-xs">
                  {new Date(a.created_at).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Risk scores */}
      {latest && (
        <section aria-labelledby="risk-scores-heading">
          <h3 id="risk-scores-heading" className="text-base font-semibold text-gray-800 mb-3">
            Latest Risk Scores
            <span className="ml-2 text-xs font-normal text-gray-400">
              ({latest.model_used} model — {new Date(latest.timestamp).toLocaleString()})
            </span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {(["diabetes", "cvd", "ckd"] as const).map((disease) => {
              const score = latest.risk_scores[disease];
              return (
                <div
                  key={disease}
                  className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm"
                >
                  <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                    {disease === "cvd" ? "Cardiovascular" : disease.charAt(0).toUpperCase() + disease.slice(1)}
                  </p>
                  <p className={`text-2xl ${riskColor(score)}`}>{score.toFixed(1)}%</p>
                  {latest.quantum_scores && latest.classical_scores && (
                    <p className="text-xs text-gray-400 mt-1">
                      Q: {latest.quantum_scores[disease].toFixed(1)}% / C: {latest.classical_scores[disease].toFixed(1)}%
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Lab values */}
      {detail.lab_values.length > 0 && (
        <section aria-labelledby="lab-values-heading">
          <h3 id="lab-values-heading" className="text-base font-semibold text-gray-800 mb-3">
            Lab Values
          </h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Parameter</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Value</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Unit</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Reference</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Flag</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {detail.lab_values.map((lv, i) => (
                  <tr key={i} className={lv.is_abnormal ? "bg-red-50" : ""}>
                    <td className="px-4 py-2 text-gray-800 capitalize">{lv.name}</td>
                    <td className="px-4 py-2 text-gray-800">{lv.value}</td>
                    <td className="px-4 py-2 text-gray-500">{lv.unit}</td>
                    <td className="px-4 py-2 text-gray-500">
                      {lv.reference_range[0]}–{lv.reference_range[1]}
                    </td>
                    <td className="px-4 py-2">
                      {lv.is_abnormal ? (
                        <span className="text-red-600 text-xs font-medium">Abnormal</span>
                      ) : (
                        <span className="text-green-600 text-xs">Normal</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* SHAP explanations */}
      {Object.keys(detail.shap_explanations).length > 0 && (
        <section aria-labelledby="shap-heading">
          <h3 id="shap-heading" className="text-base font-semibold text-gray-800 mb-3">
            SHAP Explanations
          </h3>
          <div className="space-y-4">
            {Object.entries(detail.shap_explanations).map(([predId, shap]) => (
              <div key={predId} className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                <p className="text-xs text-gray-400 mb-2">Prediction: {predId}</p>
                <p className="text-sm text-gray-700 mb-3">{shap.summary}</p>
                <div className="space-y-1">
                  {shap.waterfall_data.labels.map((label, idx) => {
                    const val = shap.waterfall_data.values[idx];
                    const width = Math.min(Math.abs(val) * 2, 100);
                    return (
                      <div key={idx} className="flex items-center gap-2 text-xs">
                        <span className="w-28 text-gray-600 truncate">{label}</span>
                        <div className="flex-1 bg-gray-100 rounded h-3 overflow-hidden">
                          <div
                            className={`h-3 rounded ${val >= 0 ? "bg-red-400" : "bg-blue-400"}`}
                            style={{ width: `${width}%` }}
                            aria-label={`${label}: ${val.toFixed(3)}`}
                          />
                        </div>
                        <span className={`w-12 text-right ${val >= 0 ? "text-red-600" : "text-blue-600"}`}>
                          {val > 0 ? "+" : ""}{val.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Prediction history */}
      {detail.prediction_history.length > 1 && (
        <section aria-labelledby="history-heading">
          <h3 id="history-heading" className="text-base font-semibold text-gray-800 mb-3">
            Prediction History
          </h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Date</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Model</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Diabetes</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">CVD</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">CKD</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {detail.prediction_history.map((p) => (
                  <tr key={p.id}>
                    <td className="px-4 py-2 text-gray-500">
                      {new Date(p.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-2 capitalize text-gray-700">{p.model_used}</td>
                    <td className="px-4 py-2"><RiskBadge score={p.risk_scores.diabetes} /></td>
                    <td className="px-4 py-2"><RiskBadge score={p.risk_scores.cvd} /></td>
                    <td className="px-4 py-2"><RiskBadge score={p.risk_scores.ckd} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Status update */}
      <section aria-labelledby="status-update-heading">
        <h3 id="status-update-heading" className="text-base font-semibold text-gray-800 mb-2">
          Update Status
        </h3>
        <div className="flex gap-2 items-center">
          <select
            value={statusInput}
            onChange={(e) => setStatusInput(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Patient status"
          >
            <option value="active">Active</option>
            <option value="reviewed">Reviewed</option>
            <option value="follow_up">Follow-up</option>
            <option value="discharged">Discharged</option>
          </select>
          <button
            onClick={handleStatusUpdate}
            className="bg-gray-700 text-white text-sm px-4 py-2 rounded-lg hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition"
          >
            Save
          </button>
          {statusMsg && (
            <span className="text-sm text-green-600">{statusMsg}</span>
          )}
        </div>
      </section>
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────

export default function DoctorDashboard() {
  const { user, logout } = useAuthContext();
  const [patients, setPatients] = useState<HighRiskPatient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  const loadPatients = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHighRiskPatients();
      setPatients(data);
    } catch {
      setError("Failed to load high-risk patients.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPatients();
  }, [loadPatients]);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function selectAll() {
    if (selected.size === patients.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(patients.map((p) => p.id)));
    }
  }

  async function handleDownloadPDF(patientId: string) {
    try {
      const job = await generateReport(patientId);
      // Poll briefly then open download
      const url = getReportDownloadUrl(job.report_id);
      const link = document.createElement("a");
      link.href = url;
      link.download = `report-${patientId}.pdf`;
      link.click();
      showToast("PDF download started.");
    } catch {
      showToast("Failed to generate PDF report.");
    }
  }

  async function handleBulkExport() {
    if (selected.size === 0) return;
    try {
      await bulkExport(Array.from(selected));
      showToast(`Bulk export started for ${selected.size} patients.`);
      setSelected(new Set());
    } catch {
      showToast("Bulk export failed.");
    }
  }

  async function handleBulkStatusUpdate(status: string) {
    if (selected.size === 0) return;
    try {
      await Promise.all(
        Array.from(selected).map((id) => updatePatientStatus(id, status))
      );
      showToast(`Updated ${selected.size} patients to "${status}".`);
      setSelected(new Set());
      loadPatients();
    } catch {
      showToast("Bulk status update failed.");
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Toast */}
      {toast && (
        <div
          role="status"
          aria-live="polite"
          className="fixed top-4 right-4 z-50 bg-gray-900 text-white text-sm px-4 py-2 rounded-lg shadow-lg"
        >
          {toast}
        </div>
      )}

      {/* Site header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-blue-700">QuantumHealthAI</span>
            <span className="hidden sm:inline text-xs text-gray-400">Clinical Dashboard</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-sm text-gray-600">{user?.email}</span>
            <button
              onClick={logout}
              aria-label="Sign out"
              className="rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page header */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Clinical Dashboard</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              High-risk patients (risk score &gt; 75%) — logged in as{" "}
              <span className="font-medium">{user?.email}</span>
            </p>
          </div>
          <button
            onClick={loadPatients}
            className="text-sm text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
            aria-label="Refresh patient list"
          >
            Refresh
          </button>
        </div>

        {error && (
          <div
            role="alert"
            className="mb-4 rounded-lg bg-red-50 border border-red-200 p-4 text-red-700 text-sm"
          >
            {error}
          </div>
        )}

        {selectedPatientId ? (
          <PatientDetailView
            patientId={selectedPatientId}
            onBack={() => setSelectedPatientId(null)}
            onDownloadPDF={handleDownloadPDF}
          />
        ) : (
          <PatientList
            patients={patients}
            selected={selected}
            onToggle={toggleSelect}
            onSelectAll={selectAll}
            onView={setSelectedPatientId}
            onDownloadPDF={handleDownloadPDF}
            onBulkExport={handleBulkExport}
            onBulkStatusUpdate={handleBulkStatusUpdate}
            loading={loading}
          />
        )}
      </div>
      </main>
    </div>
  );
}

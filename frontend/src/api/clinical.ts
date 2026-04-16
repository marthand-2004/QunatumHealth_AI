import api from "./auth";
import type { LabParameter, PredictionResult, SHAPExplanation } from "./patient";

// ── High-risk patient list ────────────────────────────────────────────────────

export interface PatientAlert {
  disease: string;
  risk_score: number;
  created_at: string;
}

export interface HighRiskPatient {
  id: string;
  email: string;
  risk_scores: { diabetes: number; cvd: number; ckd: number };
  alerts: PatientAlert[];
  status: string;
}

export async function getHighRiskPatients(): Promise<HighRiskPatient[]> {
  const { data } = await api.get<HighRiskPatient[]>("/clinical/high-risk");
  return data;
}

// ── Patient detail ────────────────────────────────────────────────────────────

export interface PatientDetail {
  id: string;
  email: string;
  status: string;
  prediction_history: PredictionResult[];
  lab_values: LabParameter[];
  shap_explanations: Record<string, SHAPExplanation>;
  alerts: PatientAlert[];
}

export async function getPatientDetail(patientId: string): Promise<PatientDetail> {
  const { data } = await api.get<PatientDetail>(`/clinical/patient/${patientId}`);
  return data;
}

// ── Status update ─────────────────────────────────────────────────────────────

export async function updatePatientStatus(
  patientId: string,
  status: string
): Promise<{ message: string }> {
  const { data } = await api.put<{ message: string }>(
    `/clinical/patient/${patientId}/status`,
    { status }
  );
  return data;
}

// ── Bulk export ───────────────────────────────────────────────────────────────

export interface BulkExportJob {
  job_id: string;
  status: string;
}

export async function bulkExport(patientIds: string[]): Promise<BulkExportJob> {
  const { data } = await api.post<BulkExportJob>("/clinical/export/bulk", {
    patient_ids: patientIds,
  });
  return data;
}

// ── PDF report ────────────────────────────────────────────────────────────────

export interface ReportJob {
  report_id: string;
  status: string;
}

export async function generateReport(patientId: string): Promise<ReportJob> {
  const { data } = await api.post<ReportJob>(`/reports/generate/${patientId}`);
  return data;
}

export function getReportDownloadUrl(reportId: string): string {
  return `/api/reports/${reportId}/download`;
}

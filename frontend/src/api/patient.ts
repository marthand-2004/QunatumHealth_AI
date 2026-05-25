import api from "./auth";

// ── Onboarding ────────────────────────────────────────────────────────────────

export interface LifestyleProfile {
  bmi: number;
  age: number;
  systolic_bp: number;
  diastolic_bp: number;
  family_history: { diabetes: boolean; cvd: boolean; ckd: boolean };
  smoking_status: "never" | "former" | "current";
  alcohol_frequency: "never" | "occasional" | "regular";
  exercise_frequency: number; // days/week 0-7
  diet_type: string;
  sleep_hours: number;
  stress_level: number; // 1-10
  medications: string[];
}

export async function submitOnboardingProfile(
  profile: LifestyleProfile
): Promise<{ message: string }> {
  const { data } = await api.post<{ message: string }>(
    "/onboarding/profile",
    profile
  );
  return data;
}

export async function updateOnboardingProfile(
  profile: LifestyleProfile
): Promise<{ message: string }> {
  const { data } = await api.put<{ message: string }>(
    "/onboarding/profile",
    profile
  );
  return data;
}

// ── OCR / Upload ──────────────────────────────────────────────────────────────

export interface JobStatus {
  job_id: string;
  ocr_status: "pending" | "processing" | "complete" | "failed";
  status?: "pending" | "processing" | "complete" | "failed"; // legacy alias
  message?: string;
}

export interface LabParameter {
  name: string;
  value: number;
  unit: string;
  reference_range: [number, number];
  is_abnormal: boolean;
  raw_text: string;
}

export interface OCRResult {
  job_id: string;
  document_id?: string; // same as job_id
  lab_parameters: LabParameter[];
  extracted_text?: string;
  zero_text_notification?: boolean;
}

export async function uploadDocument(file: File): Promise<JobStatus> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<JobStatus>("/ocr/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getOCRStatus(jobId: string): Promise<JobStatus> {
  const { data } = await api.get<JobStatus>(`/ocr/status/${jobId}`);
  return data;
}

export async function getOCRResult(jobId: string): Promise<OCRResult> {
  const { data } = await api.get<OCRResult>(`/ocr/result/${jobId}`);
  return data;
}

// Unit conversion multipliers for display units → SI units
const UNIT_MULTIPLIERS: Record<string, number> = {
  "lakhs/cumm": 100000,
  "lakh/cumm": 100000,
  "lakhs/µl": 100000,
  "10^3/µl": 1000,
  "10^3/ul": 1000,
  "thousand/µl": 1000,
};

// Name normalization: display name → canonical snake_case
function normalizeParamName(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
}

// Normalize value based on unit
function normalizeValue(value: number, unit: string): number {
  const multiplier = UNIT_MULTIPLIERS[unit.toLowerCase().trim()];
  return multiplier ? value * multiplier : value;
}

// Normalize unit string
function normalizeUnit(unit: string): string {
  const u = unit.toLowerCase().trim();
  if (u === "lakhs/cumm" || u === "lakh/cumm" || u === "lakhs/µl") return "/µL";
  if (u === "10^3/µl" || u === "10^3/ul" || u === "thousand/µl") return "/µL";
  return unit;
}

export async function verifyDocument(
  documentId: string,
  labParameters: LabParameter[]
): Promise<{ message: string }> {
  // Normalize parameters before sending
  const normalized = labParameters.map((p) => {
    const normValue = normalizeValue(p.value, p.unit);
    const normUnit = normalizeUnit(p.unit);
    const ref = p.reference_range;
    // Clamp Infinity to large safe numbers
    const refLow = isFinite(ref[0]) ? ref[0] : 0;
    const refHigh = isFinite(ref[1]) ? ref[1] : 999999;
    return {
      name: normalizeParamName(p.name),
      value: parseFloat(normValue.toFixed(6)),
      unit: normUnit,
      reference_range: [refLow, refHigh] as [number, number],
      is_abnormal: p.is_abnormal,
      raw_text: p.raw_text || "",
    };
  });

  const { data } = await api.post<{ message: string }>("/documents/verify", {
    doc_id: documentId,
    lab_parameters: normalized,
  });
  return data;
}

// ── Predictions ───────────────────────────────────────────────────────────────

export interface PredictionResult {
  id: string;
  model_used: "quantum" | "classical";
  risk_scores: { diabetes: number; cvd: number; ckd: number };
  quantum_scores: { diabetes: number; cvd: number; ckd: number } | null;
  classical_scores: { diabetes: number; cvd: number; ckd: number } | null;
  timestamp: string;
}

export interface DiseaseExplanation {
  disease: string;
  shap_values: number[];
  base_value: number;
  waterfall_chart: {
    data: {
      labels: string[];
      datasets: Array<{ data: number[]; backgroundColor: string[] }>;
    };
    meta?: { base_value: number; prediction: number };
  };
}

export interface SHAPExplanation {
  prediction_id: string;
  model_used: string;
  risk_scores: Record<string, number>;
  feature_names: string[];
  explanations: DiseaseExplanation[];
  llm_summary: string;
  // Computed helpers for the chart (derived from explanations)
  waterfall_data?: {
    labels: string[];
    values: number[];
    base_value: number;
  };
  summary?: string;
}

export interface Recommendation {
  disease: string;
  text: string;
  priority: number;
  source: "rule" | "llm";
  requires_physician: boolean;
}

export async function getLatestPrediction(): Promise<PredictionResult | null> {
  try {
    const { data } = await api.get<PredictionResult>("/predict/latest");
    return data;
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status;
    if (status === 404) return null; // no predictions yet — not an error
    return null;
  }
}

export async function getExplanation(
  predictionId: string
): Promise<SHAPExplanation> {
  const { data } = await api.post<SHAPExplanation>(
    `/explain/${predictionId}`
  );

  // Transform backend response to add waterfall_data and summary helpers
  if (data.explanations && data.explanations.length > 0) {
    // Use the highest-risk disease explanation for the chart
    const primary = data.explanations[0];
    const labels = data.feature_names ?? [];
    const values = primary.shap_values ?? [];

    data.waterfall_data = {
      labels,
      values,
      base_value: primary.base_value ?? 50,
    };
    data.summary = data.llm_summary;
  }

  return data;
}

export async function getRecommendations(
  predictionId: string
): Promise<Recommendation[]> {
  const { data } = await api.get<Recommendation[]>(
    `/recommendations/${predictionId}`
  );
  return data;
}

// ── V2 Prediction Types ───────────────────────────────────────────────────────

export interface SHAPFeatureV2 {
  feature_name: string;
  shap_value: number;
  direction: "increases risk" | "decreases risk";
}

export interface LatencyBreakdown {
  ocr_ms?: number | null;
  feature_extraction_ms: number;
  scaler_normalization_ms: number;
  classical_ml_ms: number;
  vqc_ms: number;
  hybrid_fusion_ms: number;
  clinical_rules_ms: number;
  xai_layer_ms: number;
  total_ms: number;
}

export interface PredictionResponseV2 {
  disease: "diabetes" | "cvd" | "ckd";
  risk_score: number;
  confidence: number;
  risk_level: "Low" | "Moderate" | "High" | "Critical";
  explanation: string;
  shap_features: SHAPFeatureV2[];
  triggered_rules: string[];
  threshold_used: number;
  missing_flags: Record<string, boolean>;
  robustness_score: number;
  latency: LatencyBreakdown;
  disclaimer: string;
  limitations: string[];
}

export async function getLatestPredictionV2(documentId?: string): Promise<PredictionResponseV2[] | null> {
  try {
    const url = documentId
      ? `/predict/v2/latest?document_id=${encodeURIComponent(documentId)}`
      : "/predict/v2/latest";
    const { data } = await api.get<PredictionResponseV2[]>(url);
    return data;
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status;
    if (status === 404) return null;
    return null;
  }
}

// ── Health Assistant ──────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  disclaimer?: string;
}

export async function sendChatMessage(
  content: string,
  sessionId: string | null
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/assistant/message", {
    message: content,
    session_id: sessionId,
  });
  return data;
}

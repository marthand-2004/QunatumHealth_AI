import api from "./auth";

// ── Onboarding ────────────────────────────────────────────────────────────────

export interface LifestyleProfile {
  bmi: number;
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

export async function verifyDocument(
  documentId: string,
  labParameters: LabParameter[]
): Promise<{ message: string }> {
  const { data } = await api.post<{ message: string }>("/documents/verify", {
    document_id: documentId,
    lab_parameters: labParameters,
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

export interface SHAPExplanation {
  shap_values: Record<string, number[]>;
  waterfall_data: {
    labels: string[];
    values: number[];
    base_value: number;
  };
  summary: string;
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
  } catch {
    return null;
  }
}

export async function getExplanation(
  predictionId: string
): Promise<SHAPExplanation> {
  const { data } = await api.post<SHAPExplanation>(
    `/explain/${predictionId}`
  );
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

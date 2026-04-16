# Implementation Plan: QuantumHealthAI

## Overview

Incremental build of the QuantumHealthAI platform: FastAPI backend + React 18 frontend. Each task wires into the previous, ending with a fully integrated system. The existing `app.py` (Flask RAG prototype) is replaced; `data/` directory structure is preserved.

## Tasks

- [x] 1. Project scaffolding and core infrastructure
  - Replace `app.py` with a FastAPI entry point; configure CORS, lifespan events, and MongoDB connection via Motor
  - Create directory structure: `backend/` (routers, services, models, core), `frontend/` (React 18 + Vite + Tailwind)
  - Define all Pydantic data models: `User`, `LifestyleProfile`, `LabParameter`, `Document`, `FeatureVector`, `PredictionResult`, `Recommendation`, `Report`, `ChatSession`, `ChatMessage`, `AuditLog`
  - Set up pytest with `pytest-asyncio` and `hypothesis`; configure Vitest + React Testing Library for frontend
  - _Requirements: 1.1, 3.7, 5.1, 15.3_

- [x] 2. Authentication service
  - [x] 2.1 Implement registration, login, JWT issuance, and refresh endpoints
    - `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/refresh`, `GET /api/auth/me`
    - bcrypt password hashing; JWT signed with HS256, 15-min access token + refresh token
    - Return HTTP 409 on duplicate email; HTTP 422 on missing fields
    - _Requirements: 1.1, 1.2, 1.4, 1.7_
  - [x] 2.2 Implement JWT middleware and RBAC dependency
    - `get_current_user` dependency; `require_role(roles)` dependency injected into protected routes
    - Return HTTP 401 on expired/invalid JWT; HTTP 403 on insufficient role
    - _Requirements: 1.3, 1.5, 1.6_
  - [ ]* 2.3 Write property test for JWT expiry enforcement (Property 10)
    - **Property 10: JWT Expiry Enforcement**
    - **Validates: Requirements 1.2, 1.3**
  - [ ]* 2.4 Write property test for RBAC patient data isolation (Property 11)
    - **Property 11: RBAC Patient Data Isolation**
    - **Validates: Requirements 1.5, 13.4**
  - [ ]* 2.5 Write unit tests for auth service
    - Test password hashing, JWT encode/decode, duplicate email rejection, role enforcement logic
    - _Requirements: 1.1–1.7_

- [x] 3. Health and lifestyle onboarding
  - [x] 3.1 Implement lifestyle profile endpoints
    - `POST /api/onboarding/profile` and `PUT /api/onboarding/profile` — persist to `lifestyle_profiles` collection
    - Validate all required fields (BMI, family_history, smoking_status, alcohol_frequency, exercise_frequency, diet_type, sleep_hours, stress_level, medications); return HTTP 422 with field-level errors on missing fields
    - Record `updated_at` timestamp on update
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 3.2 Write unit tests for onboarding validation
    - Test field-level validation errors, successful persistence, update timestamp recording
    - _Requirements: 2.1–2.4_

- [x] 4. OCR pipeline and file upload
  - [x] 4.1 Implement file upload endpoint with validation
    - `POST /api/ocr/upload` — accept PDF, JPG, PNG; reject files > 20 MB (HTTP 413) and unsupported formats (HTTP 415)
    - Store encrypted file in `data/uploads/` (AES-256); create `Document` record with `ocr_status="pending"`; return job ID within 2 seconds
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.7_
  - [x] 4.2 Implement background OCR processing
    - Run PaddleOCR first (table detection), fall back to Tesseract for plain-text regions; merge and deduplicate results
    - Complete within 30 seconds; update `ocr_status` to `"complete"` or `"failed"`; notify patient on zero-text extraction
    - `GET /api/ocr/status/{job_id}` and `GET /api/ocr/result/{job_id}`
    - _Requirements: 3.2, 3.3, 3.6_
  - [ ]* 4.3 Write unit tests for upload validation
    - Test file size rejection, unsupported format rejection, valid upload acceptance
    - _Requirements: 3.1, 3.4, 3.5_

- [x] 5. Document intelligence and lab parameter extraction
  - [x] 5.1 Implement lab parameter parser and unit normalizer
    - Parse OCR text into structured `LabParameter` objects; detect tabular structures and map column headers to values
    - Normalize units to SI using conversion lookup table (mg/dL ↔ mmol/L etc.); flag values outside clinical reference ranges as `is_abnormal=True`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 5.2 Write property test for lab parameter serialization round-trip (Property 1)
    - **Property 1: Lab Parameter Serialization Round-Trip**
    - **Validates: Requirements 4.7, 4.8**
  - [x] 5.3 Implement verification endpoint
    - `POST /api/documents/verify` — accept patient-corrected values, persist verified lab parameters, set `verified=True` and `verified_at` timestamp
    - `GET /api/documents/{doc_id}` — return document detail with all lab parameters and abnormal flags
    - _Requirements: 4.5, 4.6, 4.7_
  - [ ]* 5.4 Write unit tests for document intelligence
    - Test unit normalization functions, abnormal flag logic, table parser, JSON serialization
    - _Requirements: 4.1–4.7_

- [x] 6. Feature vector builder
  - [x] 6.1 Implement feature vector construction service
    - Build 14-dimensional `FeatureVector` from verified lab values + lifestyle profile: [glucose, HbA1c, creatinine, cholesterol, triglycerides, hemoglobin, BMI, age, systolic_bp, diastolic_bp, smoking_encoded, exercise_frequency, sleep_hours, stress_level]
    - Apply missing value imputation for any absent features; persist `FeatureVector` to MongoDB
    - Make lifestyle profile available to this service (Requirement 2.5)
    - _Requirements: 5.1, 6.3, 2.5_
  - [ ]* 6.2 Write property test for feature vector dimensionality invariant (Property 2)
    - **Property 2: Feature Vector Dimensionality Invariant**
    - **Validates: Requirements 5.1, 6.3**
  - [ ]* 6.3 Write unit tests for feature vector builder
    - Test dimension validation, missing value imputation, correct feature ordering
    - _Requirements: 5.1, 6.3_

- [x] 7. Quantum prediction engine
  - [x] 7.1 Implement VQC circuit and prediction endpoint
    - Build 14-qubit PennyLane circuit: `AngleEmbedding` → 3× `StronglyEntanglingLayers` → expectation value measurement → map to [0, 100]
    - `POST /api/predict/quantum` — accept `FeatureVector`, return `PredictionResult` with `risk_scores` for Diabetes, CVD, CKD within 15 seconds
    - Use `default.qubit` device (PennyLane 0.37); persist result to `predictions` collection
    - _Requirements: 5.2, 5.3, 5.4, 5.6, 5.7_
  - [ ]* 7.2 Write property test for risk score range invariant — quantum (Property 3)
    - **Property 3: Risk Score Range Invariant (Quantum)**
    - **Validates: Requirements 5.3**
  - [ ]* 7.3 Write unit tests for quantum engine
    - Test circuit construction, output range, result persistence
    - _Requirements: 5.2–5.7_

- [x] 8. Classical ML prediction engine
  - [x] 8.1 Implement RF + XGBoost ensemble and prediction endpoint
    - Load pre-trained models from `data/models/` at startup; `POST /api/predict/classical` — return `PredictionResult` within 2 seconds
    - Accept same 14-dimensional `FeatureVector`; persist result to `predictions` collection
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ]* 8.2 Write property test for risk score range invariant — classical (Property 3)
    - **Property 3: Risk Score Range Invariant (Classical)**
    - **Validates: Requirements 6.1**
  - [ ]* 8.3 Write unit tests for classical ML engine
    - Test ensemble prediction, output range, 2-second response time
    - _Requirements: 6.1–6.3_

- [x] 9. Quantum fallback and dual prediction wiring
  - [x] 9.1 Implement prediction router with quantum fallback
    - Orchestrate quantum + classical predictions in parallel; on quantum exception or timeout (>15s), set `model_used="classical"` and return classical result
    - Display both scores side by side when both succeed; persist `quantum_scores` and `classical_scores` on `PredictionResult`
    - _Requirements: 5.5, 6.4, 6.5, 14.2_
  - [ ]* 9.2 Write property test for fallback activation on quantum failure (Property 5)
    - **Property 5: Fallback Activation on Quantum Failure**
    - Mock PennyLane device to raise exception; assert `model_used="classical"` in result
    - **Validates: Requirements 5.5, 14.2**
  - [ ]* 9.3 Write integration test for full prediction pipeline
    - Upload PDF → OCR → verify → build feature vector → predict (quantum + classical) → assert both scores present
    - _Requirements: 5.1–5.7, 6.1–6.5_

- [x] 10. Checkpoint — Ensure all backend prediction tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. XAI layer — SHAP explainability
  - [x] 11.1 Implement SHAP computation and explanation endpoint
    - Use `shap.TreeExplainer` for classical model outputs; kernel-based approximation for quantum outputs
    - `POST /api/explain/{prediction_id}` — return SHAP values per feature, waterfall chart JSON (Chart.js format), and LLM-generated natural language summary
    - Complete within 10 seconds of prediction; attribute explanations to classical model when fallback is active
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 11.2 Write property test for SHAP values sum to prediction (Property 6)
    - **Property 6: SHAP Values Sum to Prediction**
    - Assert sum of SHAP values ≈ predicted score − base score (tolerance 0.01)
    - **Validates: Requirements 7.1**
  - [ ]* 11.3 Write unit tests for XAI layer
    - Test SHAP computation, waterfall chart data structure, LLM summary generation
    - _Requirements: 7.1–7.5_

- [x] 12. Recommendation engine
  - [x] 12.1 Implement rule-based and LLM recommendation generation
    - Apply clinical threshold rules (e.g., HbA1c > 6.5% → diabetes management); generate ≥ 3 recommendations per disease with risk > 30
    - Include `requires_physician=True` recommendation when any risk score > 75
    - Prioritize recommendations by SHAP magnitude; enrich with LLM using patient lifestyle context
    - `GET /api/recommendations/{prediction_id}`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [ ]* 12.2 Write property test for recommendation count (Property 7)
    - **Property 7: Recommendation Count for High-Risk Predictions**
    - **Validates: Requirements 9.1**
  - [ ]* 12.3 Write property test for physician referral on critical risk (Property 8)
    - **Property 8: Physician Referral for Critical Risk**
    - **Validates: Requirements 9.5**
  - [ ]* 12.4 Write unit tests for recommendation engine
    - Test rule-based threshold logic, SHAP-based prioritization, physician referral trigger
    - _Requirements: 9.1–9.5_

- [x] 13. Health assistant chatbot
  - [x] 13.1 Implement chat endpoints and session management
    - `POST /api/assistant/message` and `WS /api/assistant/ws/{session_id}` — streaming LLM responses
    - Inject patient lab values and risk scores into system prompt; maintain session context in `chat_sessions` / `chat_messages` collections
    - Respond within 5 seconds; include medical disclaimer when diagnosis is requested
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  - [ ]* 13.2 Write unit tests for health assistant
    - Test disclaimer injection on diagnosis request, session context retrieval, response within 5 seconds
    - _Requirements: 8.2, 8.3, 8.6_

- [x] 14. Clinical dashboard API
  - [x] 14.1 Implement doctor-facing endpoints
    - `GET /api/clinical/high-risk` — patients with any risk score > 75; `GET /api/clinical/patient/{id}` — full detail with prediction history, lab values, SHAP; load within 3 seconds
    - `POST /api/clinical/export/bulk` and `PUT /api/clinical/patient/{id}/status` — bulk export and status update
    - Restrict all endpoints to Doctor/Admin role (HTTP 403 otherwise)
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.7_
  - [x] 14.2 Implement high-risk alert generation
    - When a prediction with risk > 75 is persisted, create an alert record visible in the clinical dashboard response
    - _Requirements: 10.6_
  - [x] 14.3 Implement doctor access audit logging
    - On every `GET /api/clinical/patient/{id}` call, write an `AuditLog` entry with doctor user ID, patient user ID, and timestamp
    - _Requirements: 13.5_
  - [ ]* 14.4 Write property test for doctor access audit log (Property 12)
    - **Property 12: Doctor Access Audit Log**
    - **Validates: Requirements 13.5**
  - [ ]* 14.5 Write unit tests for clinical dashboard
    - Test high-risk filter, role enforcement, audit log creation, alert visibility
    - _Requirements: 10.1–10.7, 13.5_

- [x] 15. Report generator
  - [x] 15.1 Implement PDF generation and download endpoints
    - `POST /api/reports/generate/{patient_id}` — background task producing PDF with lab values, risk scores, SHAP waterfall chart, recommendations, model used, and timestamp using ReportLab + WeasyPrint; complete within 10 seconds
    - `GET /api/reports/{report_id}/download` — stream PDF; `GET /api/reports/shared/{token}` — return PDF or HTTP 410 if expired
    - _Requirements: 11.1, 11.2, 11.4_
  - [x] 15.2 Implement shareable link generation and expiry
    - `POST /api/reports/{report_id}/share` — generate time-limited URL with 72-hour TTL; store `share_token` and `share_expires_at` on `Report`
    - Return HTTP 410 on expired token access
    - _Requirements: 11.3, 11.5_
  - [ ]* 15.3 Write property test for shareable link expiry enforcement (Property 9)
    - **Property 9: Shareable Link Expiry Enforcement**
    - **Validates: Requirements 11.3, 11.5**
  - [ ]* 15.4 Write unit tests for report generator
    - Test share token generation, expiry calculation, PDF metadata persistence
    - _Requirements: 11.1–11.5_

- [x] 16. Checkpoint — Ensure all backend service tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. React frontend — Auth and routing
  - [x] 17.1 Scaffold React 18 + Vite + Tailwind CSS app
    - Set up React Router with protected routes; implement login, registration, and role-based redirect (Patient → dashboard, Doctor → clinical dashboard, Admin → admin panel)
    - Implement JWT storage, refresh token logic, and axios interceptors for 401 handling
    - _Requirements: 1.1–1.6, 15.3_
  - [ ]* 17.2 Write frontend unit tests for auth forms
    - Test validation, error display, successful login redirect
    - _Requirements: 1.1–1.3_

- [ ] 18. React frontend — Patient dashboard
  - [x] 18.1 Implement onboarding form
    - Lifestyle profile form with all required fields; field-level validation error display; submit to `POST /api/onboarding/profile`
    - _Requirements: 2.1–2.3, 15.1, 15.2_
  - [x] 18.2 Implement file upload and OCR verification interface
    - Drag-and-drop upload component; poll `GET /api/ocr/status/{job_id}`; display extracted lab values with abnormal flags; allow patient to edit values before confirming
    - _Requirements: 3.1–3.6, 4.5, 4.6, 15.1_
  - [x] 18.3 Implement prediction results view
    - Display quantum and classical risk scores side by side; render SHAP waterfall chart using Chart.js; show LLM explanation and recommendations
    - Show "classical prediction used" notice when fallback is active
    - _Requirements: 5.3, 5.5, 6.5, 7.2, 9.1–9.5, 15.1_
  - [x] 18.4 Implement health assistant chat interface
    - Chat UI with message history; WebSocket or polling; display disclaimer on diagnosis request
    - _Requirements: 8.1–8.6, 15.1_
  - [ ] 18.5 Write frontend unit tests for patient dashboard
    - Test OCR verification editing, SHAP chart rendering, health assistant disclaimer
    - _Requirements: 4.5, 7.2, 8.6_

- [x] 19. React frontend — Doctor clinical dashboard
  - [x] 19.1 Implement high-risk patient list and patient detail view
    - Fetch `GET /api/clinical/high-risk`; display patient list with risk scores and alerts; click-through to patient detail with prediction history, lab values, and SHAP explanations (load within 3 seconds)
    - _Requirements: 10.1, 10.2, 10.3, 10.6, 15.1_
  - [x] 19.2 Implement PDF download and bulk operations
    - "Download PDF" button per patient; bulk select + export; bulk status update
    - _Requirements: 10.4, 10.5_
  - [ ]* 19.3 Write frontend unit tests for clinical dashboard
    - Test high-risk list rendering, patient detail load, bulk selection
    - _Requirements: 10.1–10.5_

- [x] 20. Responsive layout and accessibility
  - Apply Tailwind responsive classes to all pages; verify correct rendering at 375px viewport
  - Ensure semantic HTML, ARIA labels, keyboard navigation, and sufficient color contrast on all patient-facing and doctor-facing interfaces
  - _Requirements: 15.1, 15.2_

- [x] 21. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use the `hypothesis` library (min 100 examples per property)
- Checkpoints ensure incremental validation before moving to the next phase
- The existing `data/` directory structure is preserved throughout

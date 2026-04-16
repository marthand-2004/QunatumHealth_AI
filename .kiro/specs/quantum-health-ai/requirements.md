# Requirements Document

## Introduction

QuantumHealthAI is a production-ready web platform that combines OCR-based medical report extraction, user lifestyle profiling, and quantum machine learning for disease risk prediction. The platform targets three diseases — Diabetes, Cardiovascular Disease, and Chronic Kidney Disease — and serves three user roles: Patient, Doctor, and Admin. It integrates a Variational Quantum Classifier (VQC) prediction engine alongside classical ML models, SHAP-based explainability, an AI health assistant, and a clinical dashboard for doctor review of high-risk cases.

## Glossary

- **System**: The QuantumHealthAI web platform as a whole
- **Auth_Service**: The component responsible for user registration, login, token issuance, and role enforcement
- **OCR_Pipeline**: The component that processes uploaded medical documents using Tesseract and PaddleOCR to extract text and structured lab values
- **Document_Intelligence**: The component that parses OCR output, detects tables, normalizes units, flags abnormal values, and presents a verification interface to the user
- **Quantum_Engine**: The VQC-based prediction component built with PennyLane that produces disease risk scores
- **Classical_ML**: The Random Forest + XGBoost ensemble model used for disease risk prediction and as a fallback
- **XAI_Layer**: The explainability component that generates SHAP feature attributions, waterfall charts, and LLM-based natural language explanations
- **Health_Assistant**: The context-aware AI chatbot that answers lab-related questions and provides lifestyle suggestions
- **Recommendation_Engine**: The component that generates personalized health recommendations using rule-based logic and LLM hybrid
- **Clinical_Dashboard**: The doctor-facing interface for reviewing high-risk patient cases, managing patient lists, and generating PDF reports
- **Report_Generator**: The component that produces PDF health reports, prediction history exports, and shareable links
- **Patient**: A registered user who uploads medical reports and receives risk predictions
- **Doctor**: A registered user with elevated privileges who reviews high-risk cases via the Clinical Dashboard
- **Admin**: A registered user with full system management privileges
- **Feature_Vector**: The 14-dimensional numerical input derived from lab values and lifestyle factors used by the prediction models
- **Risk_Score**: A numerical value between 0 and 100 representing predicted disease probability as a percentage
- **SHAP**: SHapley Additive exPlanations — a method for computing feature importance values
- **VQC**: Variational Quantum Classifier — a parameterized quantum circuit used for classification
- **JWT**: JSON Web Token used for stateless authentication

---

## Requirements

### Requirement 1: User Authentication and Registration

**User Story:** As a user, I want to register and log in securely with a defined role, so that I can access features appropriate to my role.

#### Acceptance Criteria

1. WHEN a new user submits a registration request with a valid email, password, and role (Patient, Doctor, or Admin), THE Auth_Service SHALL create a new account and return a success response within 3 seconds.
2. WHEN a user submits valid login credentials, THE Auth_Service SHALL issue a signed JWT with a 15-minute expiry and a refresh token.
3. WHEN a request arrives with an expired or invalid JWT, THE Auth_Service SHALL reject the request with an HTTP 401 response.
4. THE Auth_Service SHALL hash all passwords using bcrypt before storing them in the database.
5. WHEN a user attempts to access a resource outside their role's permissions, THE Auth_Service SHALL return an HTTP 403 response.
6. THE Auth_Service SHALL enforce Role-Based Access Control (RBAC) with three roles: Patient, Doctor, and Admin.
7. IF a registration request contains a duplicate email, THEN THE Auth_Service SHALL return a descriptive error message without creating a duplicate account.

---

### Requirement 2: Health and Lifestyle Onboarding

**User Story:** As a Patient, I want to enter my lifestyle and health background, so that the platform can incorporate this context into my risk predictions.

#### Acceptance Criteria

1. THE System SHALL collect the following lifestyle fields during onboarding: BMI, family history of target diseases, smoking status, alcohol consumption frequency, exercise frequency, diet type, average sleep duration in hours, stress level on a 1–10 scale, and current medications.
2. WHEN a Patient submits the onboarding form with all required fields, THE System SHALL persist the profile to the database and confirm success within 2 seconds.
3. WHEN a Patient submits the onboarding form with one or more missing required fields, THE System SHALL return a field-level validation error identifying each missing field.
4. WHEN a Patient updates their lifestyle profile, THE System SHALL overwrite the previous values and record the update timestamp.
5. THE System SHALL make the lifestyle profile available to the Quantum_Engine and Classical_ML components when constructing the Feature_Vector.

---

### Requirement 3: Medical Report Upload and Processing

**User Story:** As a Patient, I want to upload my medical reports in common formats, so that the platform can extract my lab values automatically.

#### Acceptance Criteria

1. THE System SHALL accept medical report uploads in PDF, JPG, and PNG formats.
2. WHEN a Patient uploads a supported file, THE OCR_Pipeline SHALL begin processing and return a job status response within 2 seconds of upload.
3. THE OCR_Pipeline SHALL complete text extraction from an uploaded document within 30 seconds of upload.
4. WHEN an uploaded file exceeds 20 MB, THE System SHALL reject the upload and return a descriptive error message before processing begins.
5. WHEN an uploaded file is in an unsupported format, THE System SHALL reject the upload and return a descriptive error message.
6. IF the OCR_Pipeline fails to extract any text from a document, THEN THE System SHALL notify the Patient and prompt manual data entry.
7. THE System SHALL store uploaded files in encrypted form at rest.

---

### Requirement 4: Document Intelligence and Data Extraction

**User Story:** As a Patient, I want the platform to intelligently parse my lab report and let me verify the extracted values, so that the prediction is based on accurate data.

#### Acceptance Criteria

1. WHEN OCR text extraction is complete, THE Document_Intelligence SHALL identify and extract structured lab parameters including glucose, HbA1c, creatinine, cholesterol, triglycerides, hemoglobin, and other standard blood panel values present in the document.
2. THE Document_Intelligence SHALL detect tabular structures in OCR output and map column headers to corresponding lab values.
3. THE Document_Intelligence SHALL normalize extracted lab values to standard SI units before storing them.
4. WHEN an extracted lab value falls outside the clinically normal reference range for that parameter, THE Document_Intelligence SHALL flag the value as abnormal.
5. WHEN extraction is complete, THE System SHALL present the Patient with a verification interface displaying all extracted values and their abnormal flags, allowing the Patient to correct any value before prediction.
6. WHEN a Patient confirms or corrects extracted values, THE System SHALL persist the verified values and use them for prediction.
7. THE Document_Intelligence SHALL parse extracted text into structured lab parameter objects and serialize them to JSON for downstream consumption.
8. FOR ALL valid lab parameter objects, parsing then serializing then parsing SHALL produce an equivalent object (round-trip property).

---

### Requirement 5: Quantum Disease Risk Prediction Engine

**User Story:** As a Patient, I want a quantum ML-based risk score for my target diseases, so that I receive a cutting-edge assessment of my health risk.

#### Acceptance Criteria

1. WHEN a Patient requests a prediction, THE Quantum_Engine SHALL construct a 14-dimensional Feature_Vector from the verified lab values and lifestyle profile.
2. THE Quantum_Engine SHALL encode the Feature_Vector using AngleEmbedding into a parameterized quantum circuit with StronglyEntanglingLayers.
3. WHEN the Feature_Vector is ready, THE Quantum_Engine SHALL compute a Risk_Score between 0 and 100 for each of the three target diseases: Diabetes, Cardiovascular Disease, and Chronic Kidney Disease.
4. THE Quantum_Engine SHALL complete prediction and return results within 15 seconds of receiving the Feature_Vector.
5. WHEN the Quantum_Engine is unavailable or returns an error, THE System SHALL automatically fall back to the Classical_ML component and notify the Patient that classical prediction was used.
6. THE Quantum_Engine SHALL use PennyLane 0.37 with the default.qubit device.
7. WHEN a prediction is completed, THE System SHALL persist the Risk_Score, the Feature_Vector, the model used (quantum or classical), and the timestamp to the database.

---

### Requirement 6: Classical ML Comparison Model

**User Story:** As a Patient, I want a classical ML prediction alongside the quantum result, so that I can see how the two approaches compare.

#### Acceptance Criteria

1. THE Classical_ML SHALL use a Random Forest and XGBoost ensemble to produce a Risk_Score for each target disease.
2. WHEN a prediction request is received, THE Classical_ML SHALL return a Risk_Score within 2 seconds.
3. THE Classical_ML SHALL accept the same 14-dimensional Feature_Vector as the Quantum_Engine.
4. WHEN the Quantum_Engine is unavailable, THE Classical_ML SHALL serve as the primary prediction source.
5. THE System SHALL display both the quantum Risk_Score and the classical Risk_Score side by side when both are available.

---

### Requirement 7: Explainable AI Layer

**User Story:** As a Patient, I want to understand which factors most influenced my risk score, so that I can take informed action.

#### Acceptance Criteria

1. WHEN a prediction is complete, THE XAI_Layer SHALL compute SHAP feature importance values for each feature in the Feature_Vector.
2. THE XAI_Layer SHALL render a SHAP waterfall chart showing the contribution of each feature to the final Risk_Score.
3. WHEN SHAP values are computed, THE XAI_Layer SHALL generate a natural language explanation of the top contributing factors using an LLM.
4. THE XAI_Layer SHALL complete SHAP computation and explanation generation within 10 seconds of prediction completion.
5. THE XAI_Layer SHALL attribute explanations to the Classical_ML model when the quantum fallback is active.

---

### Requirement 8: AI Health Assistant

**User Story:** As a Patient, I want to ask questions about my lab results and get lifestyle advice, so that I can better understand my health.

#### Acceptance Criteria

1. THE Health_Assistant SHALL provide a conversational chat interface accessible from the Patient dashboard.
2. WHEN a Patient sends a message, THE Health_Assistant SHALL respond within 5 seconds.
3. THE Health_Assistant SHALL maintain context across messages within a single session, referencing the Patient's lab values and Risk_Scores when relevant.
4. THE Health_Assistant SHALL provide explanations of lab parameters and their clinical significance when asked.
5. THE Health_Assistant SHALL suggest lifestyle modifications based on the Patient's profile and Risk_Scores.
6. IF a Patient's message contains a request for a definitive medical diagnosis, THEN THE Health_Assistant SHALL clarify that it provides informational guidance only and recommend consulting a licensed physician.

---

### Requirement 9: Personalized Recommendations

**User Story:** As a Patient, I want personalized health recommendations based on my risk profile, so that I know what steps to take.

#### Acceptance Criteria

1. WHEN a prediction is complete, THE Recommendation_Engine SHALL generate at least three personalized recommendations for each disease with a Risk_Score above 30.
2. THE Recommendation_Engine SHALL apply rule-based logic for recommendations derived from clinically established thresholds (e.g., HbA1c > 6.5% triggers diabetes management recommendations).
3. THE Recommendation_Engine SHALL use an LLM to generate contextual recommendations that incorporate the Patient's lifestyle profile.
4. THE Recommendation_Engine SHALL prioritize recommendations by the magnitude of the corresponding SHAP feature importance values.
5. WHEN a Patient's Risk_Score for any disease exceeds 75, THE Recommendation_Engine SHALL include a recommendation to consult a physician.

---

### Requirement 10: Clinical Dashboard

**User Story:** As a Doctor, I want a dashboard showing high-risk patients, so that I can prioritize clinical follow-up.

#### Acceptance Criteria

1. WHEN a Doctor logs in, THE Clinical_Dashboard SHALL display a list of all Patients whose Risk_Score for any disease exceeds 75.
2. THE Clinical_Dashboard SHALL allow a Doctor to view the full prediction history, lab values, and SHAP explanations for any Patient.
3. WHEN a Doctor selects a Patient, THE Clinical_Dashboard SHALL load the Patient's detail view within 3 seconds.
4. THE Clinical_Dashboard SHALL allow a Doctor to generate and download a PDF health report for any Patient.
5. THE Clinical_Dashboard SHALL support bulk management operations including bulk export and bulk status update for multiple Patients simultaneously.
6. WHEN a new Patient's Risk_Score exceeds 75, THE System SHALL generate an alert visible to Doctors on the Clinical_Dashboard.
7. THE Clinical_Dashboard SHALL be accessible only to users with the Doctor or Admin role.

---

### Requirement 11: Report Generation and Export

**User Story:** As a Patient or Doctor, I want to export health reports as PDFs and share them, so that I can use them outside the platform.

#### Acceptance Criteria

1. WHEN a Patient or Doctor requests a PDF report, THE Report_Generator SHALL produce a formatted PDF containing the Patient's lab values, Risk_Scores, SHAP waterfall chart, and recommendations within 10 seconds.
2. THE Report_Generator SHALL include the prediction model used (quantum or classical) and the prediction timestamp in every generated report.
3. WHEN a shareable link is requested, THE Report_Generator SHALL generate a time-limited URL that expires after 72 hours.
4. THE System SHALL maintain a prediction history log per Patient, accessible to both the Patient and their assigned Doctor.
5. IF a shareable link has expired, THEN THE System SHALL return an HTTP 410 response when the link is accessed.

---

### Requirement 12: Non-Functional — Performance

**User Story:** As a user, I want the platform to respond quickly under normal load, so that my experience is not degraded by latency.

#### Acceptance Criteria

1. THE System SHALL serve all page loads within 3 seconds under a load of 100 concurrent users.
2. THE OCR_Pipeline SHALL complete document processing within 30 seconds for documents up to 20 MB.
3. THE Quantum_Engine SHALL return prediction results within 15 seconds.
4. THE Classical_ML SHALL return prediction results within 2 seconds.
5. THE System SHALL support a minimum of 100 concurrent users without degradation of the above response time thresholds.

---

### Requirement 13: Non-Functional — Security and Privacy

**User Story:** As a user, I want my health data to be protected, so that my privacy is maintained.

#### Acceptance Criteria

1. THE System SHALL transmit all data over HTTPS.
2. THE Auth_Service SHALL issue JWTs with a maximum expiry of 15 minutes for access tokens.
3. THE System SHALL encrypt all health data at rest using AES-256 or equivalent.
4. THE System SHALL comply with RBAC such that Patients cannot access other Patients' data.
5. WHEN a Doctor accesses a Patient's record, THE System SHALL log the access event with the Doctor's user ID and timestamp.

---

### Requirement 14: Non-Functional — Availability and Resilience

**User Story:** As a user, I want the platform to be reliably available, so that I can access my health data when needed.

#### Acceptance Criteria

1. THE System SHALL maintain 99.5% uptime measured on a rolling 30-day window.
2. WHEN the Quantum_Engine is unavailable, THE System SHALL fall back to Classical_ML within 5 seconds without user intervention.
3. THE System SHALL display a user-facing error message within 5 seconds when any backend service is unavailable.

---

### Requirement 15: Non-Functional — Accessibility and Responsiveness

**User Story:** As a user on any device, I want the platform to be usable and accessible, so that I can access it from mobile or desktop.

#### Acceptance Criteria

1. THE System SHALL render correctly on viewports as narrow as 375px.
2. THE System SHALL meet WCAG 2.1 Level AA accessibility guidelines for all patient-facing and doctor-facing interfaces.
3. THE System SHALL use React 18 with Tailwind CSS for all frontend components.

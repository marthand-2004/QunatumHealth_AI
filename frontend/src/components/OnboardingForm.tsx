import { useState } from "react";
import { submitOnboardingProfile, updateOnboardingProfile, type LifestyleProfile } from "../api/patient";

interface Props {
  onComplete?: () => void;
}

const DIET_OPTIONS = ["omnivore", "vegetarian", "vegan", "pescatarian", "keto", "other"];

type FormState = {
  bmi: string;
  age: string;
  systolic_bp: string;
  diastolic_bp: string;
  family_history_diabetes: boolean;
  family_history_cvd: boolean;
  family_history_ckd: boolean;
  smoking_status: LifestyleProfile["smoking_status"];
  alcohol_frequency: LifestyleProfile["alcohol_frequency"];
  exercise_frequency: string;
  diet_type: string;
  sleep_hours: string;
  stress_level: string;
  medications: string;
};

const INITIAL: FormState = {
  bmi: "",
  age: "",
  systolic_bp: "",
  diastolic_bp: "",
  family_history_diabetes: false,
  family_history_cvd: false,
  family_history_ckd: false,
  smoking_status: "never",
  alcohol_frequency: "never",
  exercise_frequency: "",
  diet_type: "omnivore",
  sleep_hours: "",
  stress_level: "",
  medications: "",
};

function Field({
  id,
  label,
  hint,
  required,
  error,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-gray-700">
        {label}
        {required && <span aria-hidden="true" className="ml-0.5 text-red-500">*</span>}
        {hint && <span className="ml-1 text-xs font-normal text-gray-400">{hint}</span>}
      </label>
      <div className="mt-1">{children}</div>
      {error && (
        <p id={`${id}-error`} role="alert" className="mt-1 text-xs text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}

export default function OnboardingForm({ onComplete }: Props) {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);

  function set(key: keyof FormState, value: string | boolean) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function validate(): Record<string, string> {
    const e: Record<string, string> = {};

    const bmi = parseFloat(form.bmi);
    if (!form.bmi || isNaN(bmi) || bmi < 10 || bmi > 80)
      e.bmi = "BMI must be between 10 and 80.";

    const age = parseFloat(form.age);
    if (!form.age || isNaN(age) || age < 1 || age > 120)
      e.age = "Age must be between 1 and 120.";

    const sbp = parseFloat(form.systolic_bp);
    if (!form.systolic_bp || isNaN(sbp) || sbp < 60 || sbp > 300)
      e.systolic_bp = "Systolic BP must be between 60 and 300 mmHg.";

    const dbp = parseFloat(form.diastolic_bp);
    if (!form.diastolic_bp || isNaN(dbp) || dbp < 40 || dbp > 200)
      e.diastolic_bp = "Diastolic BP must be between 40 and 200 mmHg.";

    if (!isNaN(sbp) && !isNaN(dbp) && dbp >= sbp)
      e.diastolic_bp = "Diastolic BP must be lower than systolic BP.";

    const ex = parseInt(form.exercise_frequency);
    if (!form.exercise_frequency || isNaN(ex) || ex < 0 || ex > 7)
      e.exercise_frequency = "Exercise frequency must be 0–7 days/week.";

    const sleep = parseFloat(form.sleep_hours);
    if (!form.sleep_hours || isNaN(sleep) || sleep < 0 || sleep > 24)
      e.sleep_hours = "Sleep hours must be between 0 and 24.";

    const stress = parseInt(form.stress_level);
    if (!form.stress_level || isNaN(stress) || stress < 1 || stress > 10)
      e.stress_level = "Stress level must be between 1 and 10.";

    return e;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setServerError(null);
    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setSubmitting(true);

    const profile: LifestyleProfile = {
      bmi: parseFloat(form.bmi),
      age: parseFloat(form.age),
      systolic_bp: parseFloat(form.systolic_bp),
      diastolic_bp: parseFloat(form.diastolic_bp),
      family_history: {
        diabetes: form.family_history_diabetes,
        cvd: form.family_history_cvd,
        ckd: form.family_history_ckd,
      },
      smoking_status: form.smoking_status,
      alcohol_frequency: form.alcohol_frequency,
      exercise_frequency: parseInt(form.exercise_frequency),
      diet_type: form.diet_type,
      sleep_hours: parseFloat(form.sleep_hours),
      stress_level: parseInt(form.stress_level),
      medications: form.medications.split(",").map((m) => m.trim()).filter(Boolean),
    };

    try {
      // Try create first; if 409 (already exists) fall back to update
      try {
        await submitOnboardingProfile(profile);
      } catch (err: unknown) {
        const status = (err as { response?: { status?: number } })?.response?.status;
        if (status === 409) {
          await updateOnboardingProfile(profile);
        } else {
          throw err;
        }
      }
      setSuccess(true);
      onComplete?.();
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { message?: string; detail?: unknown } } };
      const detail = axiosErr.response?.data?.detail;
      if (Array.isArray(detail)) {
        const fieldErrors: Record<string, string> = {};
        for (const d of detail) {
          const field = (d.loc as string[]).slice(-1)[0];
          fieldErrors[field] = d.msg;
        }
        setErrors(fieldErrors);
      } else {
        setServerError(
          axiosErr.response?.data?.message ?? "Submission failed. Please try again."
        );
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (success) {
    return (
      <div role="status" className="rounded-lg bg-green-50 border border-green-200 p-6 text-center">
        <svg className="mx-auto mb-2 h-8 w-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        <p className="text-green-700 font-medium">Health profile saved.</p>
        <p className="mt-1 text-sm text-gray-500">You can now upload a medical report for your risk assessment.</p>
      </div>
    );
  }

  const inputClass = (field: string) =>
    `block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
      errors[field] ? "border-red-400 bg-red-50" : "border-gray-300"
    }`;

  return (
    <form onSubmit={handleSubmit} noValidate aria-label="Health profile form" className="space-y-6">
      {serverError && (
        <div role="alert" className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {serverError}
        </div>
      )}

      {/* ── Section: Body Measurements ── */}
      <section>
        <h3 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b border-gray-200">
          Body Measurements
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field id="bmi" label="BMI" hint="kg/m²" required error={errors.bmi}>
            <input
              id="bmi" type="number" step="0.1" min="10" max="80"
              value={form.bmi} onChange={(e) => set("bmi", e.target.value)}
              aria-describedby={errors.bmi ? "bmi-error" : undefined}
              aria-invalid={!!errors.bmi}
              placeholder="e.g. 24.5"
              className={inputClass("bmi")}
            />
          </Field>

          <Field id="age" label="Age" hint="years" required error={errors.age}>
            <input
              id="age" type="number" min="1" max="120"
              value={form.age} onChange={(e) => set("age", e.target.value)}
              aria-describedby={errors.age ? "age-error" : undefined}
              aria-invalid={!!errors.age}
              placeholder="e.g. 45"
              className={inputClass("age")}
            />
          </Field>

          <Field id="systolic_bp" label="Systolic Blood Pressure" hint="mmHg" required error={errors.systolic_bp}>
            <input
              id="systolic_bp" type="number" min="60" max="300"
              value={form.systolic_bp} onChange={(e) => set("systolic_bp", e.target.value)}
              aria-describedby={errors.systolic_bp ? "systolic_bp-error" : undefined}
              aria-invalid={!!errors.systolic_bp}
              placeholder="e.g. 120"
              className={inputClass("systolic_bp")}
            />
          </Field>

          <Field id="diastolic_bp" label="Diastolic Blood Pressure" hint="mmHg" required error={errors.diastolic_bp}>
            <input
              id="diastolic_bp" type="number" min="40" max="200"
              value={form.diastolic_bp} onChange={(e) => set("diastolic_bp", e.target.value)}
              aria-describedby={errors.diastolic_bp ? "diastolic_bp-error" : undefined}
              aria-invalid={!!errors.diastolic_bp}
              placeholder="e.g. 80"
              className={inputClass("diastolic_bp")}
            />
          </Field>
        </div>
      </section>

      {/* ── Section: Lifestyle ── */}
      <section>
        <h3 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b border-gray-200">
          Lifestyle
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field id="smoking_status" label="Smoking Status" required>
            <select
              id="smoking_status" value={form.smoking_status}
              onChange={(e) => set("smoking_status", e.target.value as LifestyleProfile["smoking_status"])}
              className={inputClass("smoking_status")}
            >
              <option value="never">Never</option>
              <option value="former">Former smoker</option>
              <option value="current">Current smoker</option>
            </select>
          </Field>

          <Field id="alcohol_frequency" label="Alcohol Frequency" required>
            <select
              id="alcohol_frequency" value={form.alcohol_frequency}
              onChange={(e) => set("alcohol_frequency", e.target.value as LifestyleProfile["alcohol_frequency"])}
              className={inputClass("alcohol_frequency")}
            >
              <option value="never">Never</option>
              <option value="occasional">Occasional</option>
              <option value="regular">Regular</option>
            </select>
          </Field>

          <Field id="exercise_frequency" label="Exercise" hint="days/week" required error={errors.exercise_frequency}>
            <input
              id="exercise_frequency" type="number" min="0" max="7"
              value={form.exercise_frequency} onChange={(e) => set("exercise_frequency", e.target.value)}
              aria-describedby={errors.exercise_frequency ? "exercise_frequency-error" : undefined}
              aria-invalid={!!errors.exercise_frequency}
              placeholder="0–7"
              className={inputClass("exercise_frequency")}
            />
          </Field>

          <Field id="diet_type" label="Diet Type" required error={errors.diet_type}>
            <select
              id="diet_type" value={form.diet_type}
              onChange={(e) => set("diet_type", e.target.value)}
              className={inputClass("diet_type")}
            >
              {DIET_OPTIONS.map((d) => (
                <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>
              ))}
            </select>
          </Field>

          <Field id="sleep_hours" label="Average Sleep" hint="hours/night" required error={errors.sleep_hours}>
            <input
              id="sleep_hours" type="number" step="0.5" min="0" max="24"
              value={form.sleep_hours} onChange={(e) => set("sleep_hours", e.target.value)}
              aria-describedby={errors.sleep_hours ? "sleep_hours-error" : undefined}
              aria-invalid={!!errors.sleep_hours}
              placeholder="e.g. 7.5"
              className={inputClass("sleep_hours")}
            />
          </Field>

          <Field id="stress_level" label="Stress Level" hint="1 = low, 10 = high" required error={errors.stress_level}>
            <input
              id="stress_level" type="number" min="1" max="10"
              value={form.stress_level} onChange={(e) => set("stress_level", e.target.value)}
              aria-describedby={errors.stress_level ? "stress_level-error" : undefined}
              aria-invalid={!!errors.stress_level}
              placeholder="1–10"
              className={inputClass("stress_level")}
            />
          </Field>
        </div>
      </section>

      {/* ── Section: Medical History ── */}
      <section>
        <h3 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b border-gray-200">
          Medical History
        </h3>

        <fieldset className="mb-4">
          <legend className="text-sm font-medium text-gray-700 mb-2">Family History of Disease</legend>
          <div className="flex flex-wrap gap-4">
            {(["diabetes", "cvd", "ckd"] as const).map((disease) => (
              <label key={disease} className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form[`family_history_${disease}` as keyof FormState] as boolean}
                  onChange={(e) => set(`family_history_${disease}` as keyof FormState, e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="font-medium">{disease === "cvd" ? "CVD" : disease.charAt(0).toUpperCase() + disease.slice(1)}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <Field id="medications" label="Current Medications" hint="comma-separated, leave blank if none">
          <input
            id="medications" type="text"
            placeholder="e.g. Metformin, Lisinopril"
            value={form.medications} onChange={(e) => set("medications", e.target.value)}
            className={inputClass("medications")}
          />
        </Field>
      </section>

      <button
        type="submit"
        disabled={submitting}
        className="w-full rounded-md bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
      >
        {submitting ? "Saving…" : "Save Health Profile"}
      </button>
    </form>
  );
}

import { useState } from "react";
import { submitOnboardingProfile, type LifestyleProfile } from "../api/patient";

interface Props {
  onComplete?: () => void;
}

const DIET_OPTIONS = ["omnivore", "vegetarian", "vegan", "pescatarian", "keto", "other"];

export default function OnboardingForm({ onComplete }: Props) {
  const [form, setForm] = useState<{
    bmi: string;
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
  }>({
    bmi: "",
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
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);

  function validate(): Record<string, string> {
    const e: Record<string, string> = {};
    const bmi = parseFloat(form.bmi);
    if (!form.bmi || isNaN(bmi) || bmi < 10 || bmi > 80)
      e.bmi = "BMI must be a number between 10 and 80.";
    const ex = parseInt(form.exercise_frequency);
    if (!form.exercise_frequency || isNaN(ex) || ex < 0 || ex > 7)
      e.exercise_frequency = "Exercise frequency must be 0–7 days/week.";
    const sleep = parseFloat(form.sleep_hours);
    if (!form.sleep_hours || isNaN(sleep) || sleep < 0 || sleep > 24)
      e.sleep_hours = "Sleep hours must be between 0 and 24.";
    const stress = parseInt(form.stress_level);
    if (!form.stress_level || isNaN(stress) || stress < 1 || stress > 10)
      e.stress_level = "Stress level must be between 1 and 10.";
    if (!form.diet_type) e.diet_type = "Diet type is required.";
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
    try {
      const profile: LifestyleProfile = {
        bmi: parseFloat(form.bmi),
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
        medications: form.medications
          .split(",")
          .map((m) => m.trim())
          .filter(Boolean),
      };
      await submitOnboardingProfile(profile);
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
      <div
        role="status"
        className="rounded-lg bg-green-50 border border-green-200 p-6 text-center"
      >
        <p className="text-green-700 font-medium">Profile saved successfully.</p>
      </div>
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      noValidate
      aria-label="Lifestyle onboarding form"
      className="space-y-6"
    >
      {serverError && (
        <div role="alert" className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {serverError}
        </div>
      )}

      {/* BMI */}
      <div>
        <label htmlFor="bmi" className="block text-sm font-medium text-gray-700">
          BMI <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <input
          id="bmi"
          type="number"
          step="0.1"
          min="10"
          max="80"
          value={form.bmi}
          onChange={(e) => setForm({ ...form, bmi: e.target.value })}
          aria-describedby={errors.bmi ? "bmi-error" : undefined}
          aria-invalid={!!errors.bmi}
          className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.bmi ? "border-red-400" : "border-gray-300"
          }`}
        />
        {errors.bmi && (
          <p id="bmi-error" role="alert" className="mt-1 text-xs text-red-600">
            {errors.bmi}
          </p>
        )}
      </div>

      {/* Family History */}
      <fieldset>
        <legend className="text-sm font-medium text-gray-700">Family History</legend>
        <div className="mt-2 flex flex-wrap gap-4">
          {(["diabetes", "cvd", "ckd"] as const).map((disease) => (
            <label key={disease} className="flex items-center gap-2 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={form[`family_history_${disease}` as keyof typeof form] as boolean}
                onChange={(e) =>
                  setForm({ ...form, [`family_history_${disease}`]: e.target.checked })
                }
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              {disease.toUpperCase()}
            </label>
          ))}
        </div>
      </fieldset>

      {/* Smoking Status */}
      <div>
        <label htmlFor="smoking_status" className="block text-sm font-medium text-gray-700">
          Smoking Status <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <select
          id="smoking_status"
          value={form.smoking_status}
          onChange={(e) =>
            setForm({ ...form, smoking_status: e.target.value as LifestyleProfile["smoking_status"] })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="never">Never</option>
          <option value="former">Former</option>
          <option value="current">Current</option>
        </select>
      </div>

      {/* Alcohol Frequency */}
      <div>
        <label htmlFor="alcohol_frequency" className="block text-sm font-medium text-gray-700">
          Alcohol Frequency <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <select
          id="alcohol_frequency"
          value={form.alcohol_frequency}
          onChange={(e) =>
            setForm({
              ...form,
              alcohol_frequency: e.target.value as LifestyleProfile["alcohol_frequency"],
            })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="never">Never</option>
          <option value="occasional">Occasional</option>
          <option value="regular">Regular</option>
        </select>
      </div>

      {/* Exercise Frequency */}
      <div>
        <label htmlFor="exercise_frequency" className="block text-sm font-medium text-gray-700">
          Exercise Frequency (days/week) <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <input
          id="exercise_frequency"
          type="number"
          min="0"
          max="7"
          value={form.exercise_frequency}
          onChange={(e) => setForm({ ...form, exercise_frequency: e.target.value })}
          aria-describedby={errors.exercise_frequency ? "exercise-error" : undefined}
          aria-invalid={!!errors.exercise_frequency}
          className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.exercise_frequency ? "border-red-400" : "border-gray-300"
          }`}
        />
        {errors.exercise_frequency && (
          <p id="exercise-error" role="alert" className="mt-1 text-xs text-red-600">
            {errors.exercise_frequency}
          </p>
        )}
      </div>

      {/* Diet Type */}
      <div>
        <label htmlFor="diet_type" className="block text-sm font-medium text-gray-700">
          Diet Type <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <select
          id="diet_type"
          value={form.diet_type}
          onChange={(e) => setForm({ ...form, diet_type: e.target.value })}
          aria-invalid={!!errors.diet_type}
          className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.diet_type ? "border-red-400" : "border-gray-300"
          }`}
        >
          {DIET_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {d.charAt(0).toUpperCase() + d.slice(1)}
            </option>
          ))}
        </select>
        {errors.diet_type && (
          <p role="alert" className="mt-1 text-xs text-red-600">
            {errors.diet_type}
          </p>
        )}
      </div>

      {/* Sleep Hours */}
      <div>
        <label htmlFor="sleep_hours" className="block text-sm font-medium text-gray-700">
          Average Sleep (hours/night) <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <input
          id="sleep_hours"
          type="number"
          step="0.5"
          min="0"
          max="24"
          value={form.sleep_hours}
          onChange={(e) => setForm({ ...form, sleep_hours: e.target.value })}
          aria-describedby={errors.sleep_hours ? "sleep-error" : undefined}
          aria-invalid={!!errors.sleep_hours}
          className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.sleep_hours ? "border-red-400" : "border-gray-300"
          }`}
        />
        {errors.sleep_hours && (
          <p id="sleep-error" role="alert" className="mt-1 text-xs text-red-600">
            {errors.sleep_hours}
          </p>
        )}
      </div>

      {/* Stress Level */}
      <div>
        <label htmlFor="stress_level" className="block text-sm font-medium text-gray-700">
          Stress Level (1–10) <span aria-hidden="true" className="text-red-500">*</span>
        </label>
        <input
          id="stress_level"
          type="number"
          min="1"
          max="10"
          value={form.stress_level}
          onChange={(e) => setForm({ ...form, stress_level: e.target.value })}
          aria-describedby={errors.stress_level ? "stress-error" : undefined}
          aria-invalid={!!errors.stress_level}
          className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            errors.stress_level ? "border-red-400" : "border-gray-300"
          }`}
        />
        {errors.stress_level && (
          <p id="stress-error" role="alert" className="mt-1 text-xs text-red-600">
            {errors.stress_level}
          </p>
        )}
      </div>

      {/* Medications */}
      <div>
        <label htmlFor="medications" className="block text-sm font-medium text-gray-700">
          Current Medications
          <span className="ml-1 text-xs text-gray-400">(comma-separated, leave blank if none)</span>
        </label>
        <input
          id="medications"
          type="text"
          placeholder="e.g. Metformin, Lisinopril"
          value={form.medications}
          onChange={(e) => setForm({ ...form, medications: e.target.value })}
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <button
        type="submit"
        disabled={submitting}
        className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
      >
        {submitting ? "Saving…" : "Save Profile"}
      </button>
    </form>
  );
}

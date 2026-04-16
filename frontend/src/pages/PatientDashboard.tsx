import { useState } from "react";
import { useAuthContext } from "../context/AuthContext";
import OnboardingForm from "../components/OnboardingForm";
import UploadOCR from "../components/UploadOCR";
import PredictionResults from "../components/PredictionResults";
import HealthAssistant from "../components/HealthAssistant";

type Tab = "onboarding" | "upload" | "results" | "assistant";

const TABS: { id: Tab; label: string }[] = [
  { id: "onboarding", label: "Health Profile" },
  { id: "upload", label: "Upload Report" },
  { id: "results", label: "Prediction Results" },
  { id: "assistant", label: "Health Assistant" },
];

export default function PatientDashboard() {
  const { user, logout } = useAuthContext();
  const [activeTab, setActiveTab] = useState<Tab>("onboarding");
  const [verifiedDocId, setVerifiedDocId] = useState<string | null>(null);

  function handleDocumentVerified(docId: string) {
    setVerifiedDocId(docId);
    setActiveTab("results");
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-blue-700">QuantumHealthAI</span>
            <span className="hidden sm:inline text-xs text-gray-400">Patient Portal</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-600 hidden sm:inline">{user?.email}</span>
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

      <main className="mx-auto max-w-5xl px-4 py-6 sm:px-6">
        {/* Welcome */}
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-gray-900">
            Welcome back{user?.email ? `, ${user.email.split("@")[0]}` : ""}
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your health profile, upload reports, and view your risk predictions.
          </p>
        </div>

        {/* Tab bar */}
        <nav
          role="tablist"
          aria-label="Patient dashboard sections"
          className="mb-6 flex gap-1 overflow-x-auto rounded-lg bg-gray-100 p-1"
        >
          {TABS.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              id={`tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                activeTab === tab.id
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Tab panels */}
        <div
          id="panel-onboarding"
          role="tabpanel"
          aria-labelledby="tab-onboarding"
          hidden={activeTab !== "onboarding"}
          className="rounded-lg bg-white border border-gray-200 p-6 shadow-sm"
        >
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Lifestyle &amp; Health Profile
          </h2>
          <p className="mb-5 text-sm text-gray-500">
            Complete your profile so we can personalise your risk predictions.
          </p>
          <OnboardingForm onComplete={() => setActiveTab("upload")} />
        </div>

        <div
          id="panel-upload"
          role="tabpanel"
          aria-labelledby="tab-upload"
          hidden={activeTab !== "upload"}
          className="rounded-lg bg-white border border-gray-200 p-6 shadow-sm"
        >
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Upload Medical Report
          </h2>
          <p className="mb-5 text-sm text-gray-500">
            Upload a lab report (PDF, JPG, or PNG). We'll extract your values automatically.
          </p>
          <UploadOCR onVerified={handleDocumentVerified} />
          {verifiedDocId && (
            <p className="mt-4 text-xs text-gray-400">
              Last verified document: <code>{verifiedDocId}</code>
            </p>
          )}
        </div>

        <div
          id="panel-results"
          role="tabpanel"
          aria-labelledby="tab-results"
          hidden={activeTab !== "results"}
          className="rounded-lg bg-white border border-gray-200 p-6 shadow-sm"
        >
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Prediction Results
          </h2>
          <PredictionResults />
        </div>

        <div
          id="panel-assistant"
          role="tabpanel"
          aria-labelledby="tab-assistant"
          hidden={activeTab !== "assistant"}
          className="rounded-lg bg-white border border-gray-200 p-6 shadow-sm"
        >
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Health Assistant
          </h2>
          <HealthAssistant />
        </div>
      </main>
    </div>
  );
}

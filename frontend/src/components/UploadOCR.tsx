import { useState, useRef, useCallback } from "react";
import {
  uploadDocument,
  getOCRStatus,
  getOCRResult,
  verifyDocument,
  type LabParameter,
  type OCRResult,
} from "../api/patient";

interface Props {
  onVerified?: (documentId: string) => void;
}

type Stage = "idle" | "uploading" | "polling" | "review" | "confirmed" | "error";

const ACCEPTED = ["application/pdf", "image/jpeg", "image/png"];
const MAX_SIZE_MB = 20;

export default function UploadOCR({ onVerified }: Props) {
  const [stage, setStage] = useState<Stage>("idle");
  const [dragOver, setDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [editedParams, setEditedParams] = useState<LabParameter[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  const processFile = useCallback(async (file: File) => {
    setErrorMsg(null);

    if (!ACCEPTED.includes(file.type)) {
      setErrorMsg("Unsupported file type. Please upload a PDF, JPG, or PNG.");
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File exceeds ${MAX_SIZE_MB} MB limit.`);
      return;
    }

    setStage("uploading");
    try {
      const job = await uploadDocument(file);
      setStage("polling");

      const pollStart = Date.now();
      const MAX_POLL_MS = 60_000; // 60 second timeout

      pollRef.current = setInterval(async () => {
        // Timeout guard
        if (Date.now() - pollStart > MAX_POLL_MS) {
          stopPolling();
          setErrorMsg("OCR processing timed out. Please try again.");
          setStage("error");
          return;
        }

        try {
          const statusResp = await getOCRStatus(job.job_id);
          const ocr_status = statusResp.ocr_status ?? (statusResp as any).status;

          if (ocr_status === "complete") {
            stopPolling();
            const result = await getOCRResult(job.job_id);
            if (result.lab_parameters.length === 0) {
              setErrorMsg(
                "No lab values could be extracted. Please enter your values manually."
              );
              setStage("error");
              return;
            }
            setOcrResult(result);
            setEditedParams(result.lab_parameters.map((p) => ({ ...p })));
            setStage("review");
          } else if (ocr_status === "failed") {
            stopPolling();
            setErrorMsg(
              statusResp.message ?? "OCR processing failed. Please try again."
            );
            setStage("error");
          }
        } catch {
          stopPolling();
          setErrorMsg("Lost connection while checking OCR status.");
          setStage("error");
        }
      }, 3000);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { status?: number; data?: { message?: string } } };
      const status = axiosErr.response?.status;
      if (status === 413) setErrorMsg("File exceeds the 20 MB size limit.");
      else if (status === 415) setErrorMsg("Unsupported file format.");
      else setErrorMsg(axiosErr.response?.data?.message ?? "Upload failed.");
      setStage("error");
    }
  }, []);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }

  function updateParam(index: number, value: string) {
    setEditedParams((prev) => {
      const next = [...prev];
      const parsed = parseFloat(value);
      next[index] = {
        ...next[index],
        value: isNaN(parsed) ? next[index].value : parsed,
        is_abnormal:
          !isNaN(parsed)
            ? parsed < next[index].reference_range[0] ||
              parsed > next[index].reference_range[1]
            : next[index].is_abnormal,
      };
      return next;
    });
  }

  async function handleConfirm() {
    if (!ocrResult) return;
    setStage("uploading"); // reuse spinner
    try {
      await verifyDocument(ocrResult.document_id ?? ocrResult.job_id, editedParams);
      setStage("confirmed");
      onVerified?.(ocrResult.document_id);
    } catch {
      setErrorMsg("Failed to save verified values. Please try again.");
      setStage("review");
    }
  }

  function reset() {
    stopPolling();
    setStage("idle");
    setErrorMsg(null);
    setOcrResult(null);
    setEditedParams([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // ── Confirmed ────────────────────────────────────────────────────────────────
  if (stage === "confirmed") {
    return (
      <div role="status" className="rounded-lg bg-green-50 border border-green-200 p-6 text-center">
        <p className="text-green-700 font-medium">Lab values confirmed and saved.</p>
        <button
          onClick={reset}
          className="mt-3 text-sm text-blue-600 underline hover:text-blue-800"
        >
          Upload another document
        </button>
      </div>
    );
  }

  // ── Review ───────────────────────────────────────────────────────────────────
  if (stage === "review" && ocrResult) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          Review the extracted values below. Edit any incorrect entries before confirming.
        </p>
        <div className="overflow-x-auto rounded-lg border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Parameter</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Value</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Unit</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Reference</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 bg-white">
              {editedParams.map((param, i) => (
                <tr
                  key={param.name}
                  className={param.is_abnormal ? "bg-red-50" : ""}
                >
                  <td className="px-4 py-2 font-medium text-gray-800 capitalize">
                    {param.name.replace(/_/g, " ")}
                  </td>
                  <td className="px-4 py-2">
                    <input
                      type="number"
                      step="any"
                      value={param.value}
                      onChange={(e) => updateParam(i, e.target.value)}
                      aria-label={`Edit value for ${param.name}`}
                      className={`w-24 rounded border px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                        param.is_abnormal
                          ? "border-red-400 bg-red-50 text-red-700"
                          : "border-gray-300"
                      }`}
                    />
                  </td>
                  <td className="px-4 py-2 text-gray-600">{param.unit}</td>
                  <td className="px-4 py-2 text-gray-500">
                    {param.reference_range[0]}–{param.reference_range[1]}
                  </td>
                  <td className="px-4 py-2">
                    {param.is_abnormal ? (
                      <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">
                        Abnormal
                      </span>
                    ) : (
                      <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                        Normal
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {errorMsg && (
          <p role="alert" className="text-sm text-red-600">
            {errorMsg}
          </p>
        )}
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={handleConfirm}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Confirm Values
          </button>
          <button
            onClick={reset}
            className="rounded-md border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  // ── Uploading / Polling ───────────────────────────────────────────────────────
  if (stage === "uploading" || stage === "polling") {
    return (
      <div role="status" aria-live="polite" className="flex flex-col items-center gap-3 py-10">
        <svg
          className="h-8 w-8 animate-spin text-blue-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v8H4z"
          />
        </svg>
        <p className="text-sm text-gray-600">
          {stage === "uploading" ? "Uploading document…" : "Extracting lab values…"}
        </p>
      </div>
    );
  }

  // ── Idle / Error ──────────────────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop zone for medical report upload"
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-12 transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50"
        }`}
      >
        <svg
          className="mb-3 h-10 w-10 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
        <p className="text-sm font-medium text-gray-700">
          Drag &amp; drop your report here, or click to browse
        </p>
        <p className="mt-1 text-xs text-gray-400">PDF, JPG, PNG — max 20 MB</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.jpg,.jpeg,.png"
          onChange={handleFileChange}
          className="sr-only"
          aria-hidden="true"
        />
      </div>
      {errorMsg && (
        <p role="alert" className="text-sm text-red-600">
          {errorMsg}
        </p>
      )}
    </div>
  );
}

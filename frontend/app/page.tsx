"use client";

import { useEffect, useState } from "react";

// where the backend api is running. change this when deploying.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const NUMERIC_FIELDS = [
  "months_as_customer",
  "age",
  "policy_annual_premium",
  "policy_deductable",
  "umbrella_limit",
  "number_of_vehicles_involved",
  "bodily_injuries",
  "witnesses",
  "total_claim_amount",
  "incident_hour_of_the_day",
];

// some default values so you can just click predict and see it work
const defaults: Record<string, string | number> = {
  months_as_customer: 200,
  age: 38,
  policy_annual_premium: 1250,
  policy_deductable: 1000,
  umbrella_limit: 0,
  number_of_vehicles_involved: 1,
  bodily_injuries: 0,
  witnesses: 1,
  total_claim_amount: 50000,
  incident_hour_of_the_day: 12,
};

// make field names look nicer for the labels
function pretty(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

type Result = {
  fraud_probability: number;
  prediction: string;
  risk_level: string;
};

export default function Home() {
  const [options, setOptions] = useState<Record<string, string[]>>({});
  const [form, setForm] = useState<Record<string, string | number>>(defaults);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // get the dropdown options from the backend when the page loads
  useEffect(() => {
    fetch(`${API_URL}/features`)
      .then((res) => res.json())
      .then((data) => {
        setOptions(data.categorical);
        // set the first option as the default for each dropdown
        const catDefaults: Record<string, string> = {};
        for (const key in data.categorical) {
          catDefaults[key] = data.categorical[key][0];
        }
        setForm((f) => ({ ...f, ...catDefaults }));
      })
      .catch(() => setError("Could not reach the API. Is the backend running?"));
  }, []);

  function handleChange(name: string, value: string) {
    setForm({ ...form, [name]: value });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    // numbers come from the inputs as strings, convert them back
    const payload: Record<string, string | number> = { ...form };
    for (const field of NUMERIC_FIELDS) {
      payload[field] = Number(form[field]);
    }

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Something went wrong calling the API.");
    }
    setLoading(false);
  }

  // pick a colour for the result based on the risk level
  function riskColor(level: string) {
    if (level === "High") return "bg-red-100 text-red-700 border-red-300";
    if (level === "Medium") return "bg-amber-100 text-amber-700 border-amber-300";
    return "bg-green-100 text-green-700 border-green-300";
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 py-10 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold">ClaimGuard</h1>
        <p className="text-slate-600 mt-1">
          Insurance claim fraud detection. Fill in the claim details and get a fraud risk score.
        </p>

        <form onSubmit={handleSubmit} className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="font-semibold mb-3">Claim details</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {NUMERIC_FIELDS.map((field) => (
              <div key={field}>
                <label className="block text-sm text-slate-600 mb-1">{pretty(field)}</label>
                <input
                  type="number"
                  value={form[field]}
                  onChange={(e) => handleChange(field, e.target.value)}
                  className="w-full border border-slate-300 rounded px-3 py-2"
                />
              </div>
            ))}

            {Object.keys(options).map((field) => (
              <div key={field}>
                <label className="block text-sm text-slate-600 mb-1">{pretty(field)}</label>
                <select
                  value={form[field]}
                  onChange={(e) => handleChange(field, e.target.value)}
                  className="w-full border border-slate-300 rounded px-3 py-2 bg-white"
                >
                  {options[field].map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-6 bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded disabled:opacity-50"
          >
            {loading ? "Checking..." : "Check for fraud"}
          </button>
        </form>

        {error && <p className="mt-4 text-red-600">{error}</p>}

        {result && (
          <div className={`mt-6 border rounded-lg p-6 ${riskColor(result.risk_level)}`}>
            <p className="text-sm uppercase tracking-wide">Result</p>
            <p className="text-2xl font-bold mt-1">
              {(result.fraud_probability * 100).toFixed(1)}% fraud probability
            </p>
            <p className="mt-1">
              Prediction: <span className="font-semibold">{result.prediction}</span> &middot; Risk:{" "}
              <span className="font-semibold">{result.risk_level}</span>
            </p>
          </div>
        )}

        <p className="text-xs text-slate-400 mt-10">
          Built as a learning project. Model is trained on a small public dataset so predictions are
          not perfect.
        </p>
      </div>
    </main>
  );
}

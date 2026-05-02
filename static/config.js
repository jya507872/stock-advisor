// Stock Advisor — API configuration
// ─────────────────────────────────────────────────────────────────────────
// LOCAL:   Leave window.API_BASE empty — the FastAPI server serves both
//          the frontend AND the API on http://localhost:8000.
//
// NETLIFY: After deploying the backend on Render, paste its URL here:
//
//   window.API_BASE = "https://stock-advisor-api.onrender.com";
//
// Then redeploy the Netlify site (or use a Netlify env variable instead).
// ─────────────────────────────────────────────────────────────────────────

window.API_BASE = "";   // ← replace with your Render URL when deploying to Netlify

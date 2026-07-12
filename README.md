# TipFork 🍴

**AMD Hackathon — Track 3 (Unicorn Pre-Screening) submission**

TipFork is a mobile-first **menu agent**: photograph a restaurant menu and it extracts the dishes and prices, translates dish names into your language, generates AI visuals of each dish, and estimates tax and tip with a per-person split — reconciled against the real receipt at the end of the meal.

All AI inference (vision, text, and image generation) is served by **Fireworks AI** (approved compute for this hackathon).

## What it does

1. **Scan the menu** — camera/gallery photo, OCR'd locally with Tesseract.js.
2. **Extract dishes + prices** — a Fireworks vision-language model reads the menu photo directly (OCR text is fallback context only).
3. **Translate dishes** — a Fireworks text model translates dish names concisely for a mobile list, keeping meanings specific.
4. **Generate dish visuals** — Fireworks FLUX.1 [schnell] renders a recognizable image of each dish so you know what you're ordering.
5. **Estimate tax & tip** — free built-in state/region tax rate table plus tip presets; per-person split (even or by dish).
6. **Receipt reconciliation** — photograph the receipt and TipFork adjusts each person's share to match what was actually charged.

There are **no payment features** — TipFork estimates and splits; it never moves money.

## AMD / Fireworks resource usage

| Feature | Model (Fireworks serverless) | Endpoint |
|---|---|---|
| Menu extraction (image → dishes/prices) | `accounts/fireworks/models/kimi-k2p6` (multimodal) | `POST /v1/chat/completions` |
| Dish translation | `accounts/fireworks/models/gpt-oss-120b` | `POST /v1/chat/completions` |
| Dish visuals | `accounts/fireworks/models/flux-1-schnell-fp8` | `POST /v1/workflows/.../text_to_image` |

All models are configurable via `.env` (see `.env.example`). Provider calls live in one place: `backend/server.js` → `callFireworksChat()` and `callFireworksImageGeneration()`.

## Where the implementation lives

```text
tipfork/
├─ backend/server.js          # ★ Main code path: Express API + all Fireworks AI calls
│    ├─ POST /api/agent/menu/extract     (vision model + local OCR merge)
│    ├─ POST /api/agent/menu/translate   (text model, JSON-constrained)
│    ├─ POST /api/agent/menu/visuals     (FLUX image gen: cache, concurrency, timeouts)
│    └─ GET  /api/health                 (config sanity check)
├─ www/index.html             # ★ Full SPA frontend (OCR, UI, tax/tip math, receipt scan)
├─ server.js                  # compatibility entrypoint -> backend/server.js
├─ scripts/check-fireworks.js # verifies your key + all three models locally
├─ samples/menus/             # sample menu photos for testing
├─ .env.example               # required configuration, documented
└─ capacitor.config.json      # optional mobile packaging (Capacitor)
```

## Setup (complete)

Requires Node.js 18+ (uses built-in `fetch`).

```bash
npm install
cp .env.example .env          # then set FIREWORKS_API_KEY
npm run check:fireworks       # verifies key + all three models
node server.js                # backend on http://127.0.0.1:3000
```

Then open `www/index.html` in a browser (or serve the `www/` folder). `CONFIG.BACKEND_URL` in `www/index.html` defaults to `http://127.0.0.1:3000`.

Try it with a sample: upload any image from `samples/menus/` at Step 1.

## External services (documented)

- **Fireworks AI** — required. Extraction, translation, and (when enabled) FLUX image generation. Key in `.env` (`FIREWORKS_API_KEY`), never in client code.
- **Qwen / DashScope** — optional fallback for dish visuals (`VISUAL_PROVIDER=qwen` or `auto`). Key in `.env` (`QWEN_API_KEY` / `DASHSCOPE_API_KEY`).
- **Google Maps Geocoding** — optional. Only used to auto-detect your state for the tax-rate lookup; without a key the app falls back to a manual region dropdown. (`CONFIG.GOOGLE_MAPS_KEY` in `www/index.html`.)
- **Tesseract.js / jsDelivr CDN** — client-side OCR, no key.

No other services. Tax rates come from a bundled table; tips are computed locally.

## Reliability behavior

Every AI route degrades gracefully rather than failing the flow: extraction falls back to local OCR parsing, translation falls back to a labeled local passthrough, and visuals fall back to styled placeholders. Generated images are cached in-process (LRU) and produced with bounded concurrency, per-item timeouts, and a route-level time budget.

## Mobile packaging (optional, Capacitor)

```bash
npm run add:ios && npm run open:ios       # or add:android / open:android
```

## Originality

Built by Sophia Liu for this hackathon. The AI orchestration, extraction/translation/visual pipeline, and tax/tip/receipt logic are original work; open-source libraries used are Express, cors, Tesseract.js, and Capacitor (all listed in `package.json`).


/* =====================================================================
   TipFork backend (Node + Express)
   ---------------------------------------------------------------------
   AI provider: Fireworks AI (https://fireworks.ai)
   - Menu extraction: Fireworks vision-language model (Qwen2.5-VL)
   - Dish translation: Fireworks text model (Llama 3.3 70B)
   - Dish visuals:     Fireworks FLUX.1 [schnell] image generation

   Tax & tip are computed entirely client-side (free built-in rate table
   + receipt reconciliation) — no tax endpoint, no payment endpoints.

   Setup:
     npm install
     cp .env.example .env   # add FIREWORKS_API_KEY
     node server.js
   ===================================================================== */
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

function loadLocalEnv(){
  // Look for .env next to the project root (parent of backend/) first,
  // then the current working directory — so the server works no matter
  // which directory it is launched from.
  const candidates = [
    path.join(__dirname, '..', '.env'),
    path.join(process.cwd(), '.env')
  ];
  const envPath = candidates.find(p => fs.existsSync(p));
  if(!envPath) return;
  const raw = fs.readFileSync(envPath, 'utf8');
  raw.split(/\r?\n/).forEach(line => {
    const trimmed = line.trim();
    if(!trimmed || trimmed.startsWith('#')) return;
    const m = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if(!m) return;
    const key = m[1];
    if(process.env[key] != null) return;
    let val = m[2].trim();
    if((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))){
      val = val.slice(1, -1);
    }
    process.env[key] = val;
  });
}

loadLocalEnv();

const app = express();
app.use(cors());               // lock this down to your app's origin in prod
app.use(express.json({ limit: '25mb' }));

/* ---- Fireworks AI configuration ---- */
const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY || '';
const FIREWORKS_BASE_URL = (process.env.FIREWORKS_BASE_URL || 'https://api.fireworks.ai/inference/v1').replace(/\/+$/, '');
// Vision-language model used for menu extraction (reads the menu photo).
const FIREWORKS_VISION_MODEL = process.env.FIREWORKS_VISION_MODEL || 'accounts/fireworks/models/qwen2p5-vl-32b-instruct';
// Text model used for dish translation.
const FIREWORKS_TEXT_MODEL = process.env.FIREWORKS_TEXT_MODEL || 'accounts/fireworks/models/llama-v3p3-70b-instruct';
// Image model used for dish visuals (FLUX.1 [schnell] FP8 workflow).
const FIREWORKS_IMAGE_MODEL = process.env.FIREWORKS_IMAGE_MODEL || 'accounts/fireworks/models/flux-1-schnell-fp8';

/* ---- Visual provider selection ----
   fireworks | qwen | auto (auto = try Fireworks first, fall back to Qwen) */
const VISUAL_PROVIDER = String(process.env.VISUAL_PROVIDER || 'auto').trim().toLowerCase();

/* ---- Qwen / DashScope image generation (fallback provider) ---- */
const QWEN_API_KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY || '';
const QWEN_IMAGE_MODEL = process.env.QWEN_IMAGE_MODEL || 'qwen-image-2.0';
const QWEN_IMAGE_ENDPOINT = (process.env.QWEN_IMAGE_ENDPOINT ||
  'https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation').replace(/\/+$/, '');
const VISUAL_IMAGE_SIZE = process.env.VISUAL_IMAGE_SIZE || '1024*1024';

/* ---- Visual generation tuning ---- */
const VISUALS_CONCURRENCY = Math.max(1, Math.min(4, Number.parseInt(process.env.VISUALS_CONCURRENCY || '2', 10) || 2));
const VISUALS_ITEM_TIMEOUT_MS = Math.max(6000, Number.parseInt(process.env.VISUALS_ITEM_TIMEOUT_MS || '28000', 10) || 28000);
const VISUALS_ROUTE_BUDGET_MS = Math.max(12000, Number.parseInt(process.env.VISUALS_ROUTE_BUDGET_MS || '90000', 10) || 90000);
const VISUAL_CACHE_MAX = Math.max(50, Number.parseInt(process.env.VISUAL_CACHE_MAX || '400', 10) || 400);
const visualImageCache = new Map();

function fireworksConfigured(){
  return !!FIREWORKS_API_KEY;
}

function menuAgentDemoMode(){
  return !fireworksConfigured();
}

/* =====================================================================
   Fireworks API calls
   ===================================================================== */

// OpenAI-compatible chat completions (text and vision models).
async function callFireworksChat({ model, messages, maxTokens = 1000, temperature = 0.2 }){
  if(!fireworksConfigured()){
    throw new Error('Fireworks API key missing. Set FIREWORKS_API_KEY in .env.');
  }
  const response = await fetch(`${FIREWORKS_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${FIREWORKS_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: String(model || FIREWORKS_TEXT_MODEL),
      messages: Array.isArray(messages) ? messages : [],
      temperature,
      max_tokens: maxTokens
    })
  });

  const json = await response.json().catch(() => ({}));
  if(!response.ok){
    const message =
      (json && (json.message || (json.error && (json.error.message || json.error)))) ||
      `Fireworks chat completion failed (${response.status}).`;
    console.error(`[fireworks] chat ${model} -> HTTP ${response.status}:`, JSON.stringify(json).slice(0, 300));
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }

  const content =
    json && Array.isArray(json.choices) &&
    json.choices[0] && json.choices[0].message &&
    typeof json.choices[0].message.content === 'string'
      ? json.choices[0].message.content.trim()
      : '';
  if(!content) throw new Error('Fireworks chat completion returned empty content.');
  return content;
}

// FLUX.1 [schnell] text-to-image workflow. Returns a data URL.
async function callFireworksImageGeneration(prompt){
  if(!fireworksConfigured()){
    throw new Error('Fireworks API key missing. Set FIREWORKS_API_KEY in .env.');
  }
  const endpoint = `${FIREWORKS_BASE_URL}/workflows/${FIREWORKS_IMAGE_MODEL}/text_to_image`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${FIREWORKS_API_KEY}`,
      'Content-Type': 'application/json',
      'Accept': 'image/jpeg'
    },
    body: JSON.stringify({ prompt })
  });

  if(!response.ok){
    let message = `Fireworks image generation failed (${response.status}).`;
    try{
      const errText = await response.text();
      console.error(`[fireworks] image ${FIREWORKS_IMAGE_MODEL} -> HTTP ${response.status}:`, errText.slice(0, 300));
      const errJson = JSON.parse(errText);
      message = (errJson && (errJson.message || (errJson.error && errJson.error.message))) || message;
    }catch(_){}
    throw new Error(message);
  }

  const contentType = (response.headers.get('content-type') || '').split(';')[0].trim();
  if(/^image\//i.test(contentType)){
    const data = Buffer.from(await response.arrayBuffer()).toString('base64');
    return `data:${contentType};base64,${data}`;
  }
  // Some deployments return JSON with base64 payload instead of raw bytes.
  const json = await response.json().catch(() => ({}));
  const b64 = typeof json.base64 === 'string' ? json.base64
    : (Array.isArray(json.base64) && typeof json.base64[0] === 'string' ? json.base64[0] : '');
  if(b64) return `data:image/png;base64,${b64.replace(/^data:image\/\w+;base64,/, '')}`;
  throw new Error('Fireworks image generation returned no image data.');
}

/* =====================================================================
   Qwen / DashScope image generation (fallback visual provider)
   ===================================================================== */

function qwenImageConfigured(){
  return !!(QWEN_API_KEY && QWEN_IMAGE_ENDPOINT);
}

async function fetchImageToDataUrl(imageUrl){
  const url = String(imageUrl || '').trim();
  if(!url) throw new Error('Image URL is empty.');
  if(url.startsWith('data:image/')) return url;
  if(!/^https?:\/\//i.test(url)) throw new Error('Unsupported image URL format.');

  const response = await fetch(url);
  if(!response.ok){
    throw new Error(`Could not download generated image (${response.status}).`);
  }
  const contentType = (response.headers.get('content-type') || 'image/png').split(';')[0].trim();
  const mimeType = /^image\//i.test(contentType) ? contentType : 'image/png';
  const data = Buffer.from(await response.arrayBuffer()).toString('base64');
  return `data:${mimeType};base64,${data}`;
}

async function callQwenImageGeneration(prompt){
  if(!qwenImageConfigured()){
    throw new Error('Qwen API key missing. Set QWEN_API_KEY or DASHSCOPE_API_KEY.');
  }

  const response = await fetch(QWEN_IMAGE_ENDPOINT, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${QWEN_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: QWEN_IMAGE_MODEL,
      input: {
        messages: [{
          role: 'user',
          content: [{ text: prompt }]
        }]
      },
      parameters: {
        size: VISUAL_IMAGE_SIZE,
        watermark: false,
        prompt_extend: true
      }
    })
  });

  const json = await response.json().catch(() => ({}));
  if(!response.ok){
    const message =
      (json && (json.message || json.error_message || (json.error && json.error.message))) ||
      `Qwen image generation failed (${response.status}).`;
    console.error(`[qwen] image ${QWEN_IMAGE_MODEL} -> HTTP ${response.status}:`, JSON.stringify(json).slice(0, 300));
    throw new Error(message);
  }

  const choices = json && json.output && Array.isArray(json.output.choices) ? json.output.choices : [];
  const firstMessage = choices[0] && choices[0].message ? choices[0].message : {};
  const content = Array.isArray(firstMessage.content) ? firstMessage.content : [];
  let imageRef = '';
  for(const part of content){
    if(part && typeof part.image === 'string' && part.image.trim()){
      imageRef = part.image.trim();
      break;
    }
    if(part && typeof part.image_url === 'string' && part.image_url.trim()){
      imageRef = part.image_url.trim();
      break;
    }
    if(part && part.image_url && typeof part.image_url.url === 'string' && part.image_url.url.trim()){
      imageRef = part.image_url.url.trim();
      break;
    }
  }
  if(!imageRef && json && json.output && typeof json.output.image_url === 'string'){
    imageRef = json.output.image_url.trim();
  }
  if(!imageRef){
    throw new Error('Qwen image generation returned no image URL.');
  }

  return fetchImageToDataUrl(imageRef);
}

/* ---- Provider-aware visual generation ---- */
function visualsConfigured(){
  if(VISUAL_PROVIDER === 'fireworks') return fireworksConfigured();
  if(VISUAL_PROVIDER === 'qwen') return qwenImageConfigured();
  return fireworksConfigured() || qwenImageConfigured();  // auto
}

async function callVisualImageGeneration(prompt){
  if(VISUAL_PROVIDER === 'fireworks') return callFireworksImageGeneration(prompt);
  if(VISUAL_PROVIDER === 'qwen') return callQwenImageGeneration(prompt);
  // auto: try Fireworks FLUX first, fall back to Qwen
  if(fireworksConfigured()){
    try{
      return await callFireworksImageGeneration(prompt);
    }catch(err){
      if(!qwenImageConfigured()) throw err;
    }
  }
  return callQwenImageGeneration(prompt);
}

/* =====================================================================
   Shared helpers
   ===================================================================== */

function normalizeMenuItems(items){
  return Array.isArray(items)
    ? items
        .filter(item => item && typeof item.name === 'string' && item.name.trim())
        .map(item => ({
          id: String(item.id),
          name: item.name.trim(),
          sourceName: typeof item.sourceName === 'string' ? item.sourceName.trim() : '',
          visualPrompt: typeof item.visualPrompt === 'string' ? item.visualPrompt.trim() : ''
        }))
    : [];
}

function normalizeMenuImageDataUrl(value){
  if(typeof value !== 'string') return null;
  const imageRef = value.trim();
  if(!imageRef) return null;
  if(imageRef.length > 8_000_000) return null;
  if(imageRef.startsWith('data:image/')) return imageRef;
  if(/^https?:\/\//i.test(imageRef)) return imageRef;
  return null;
}

function cleanModelJson(text){
  return (text || '')
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
}

function parseModelJsonSafely(text){
  const cleaned = cleanModelJson(text);
  try{
    return cleaned ? JSON.parse(cleaned) : null;
  }catch(_){
    const obj = cleaned.match(/\{[\s\S]*\}/);
    if(obj){
      try{ return JSON.parse(obj[0]); }catch(__){}
    }
    const arr = cleaned.match(/\[[\s\S]*\]/);
    if(arr){
      try{ return JSON.parse(arr[0]); }catch(__){}
    }
    return null;
  }
}

function firstNonEmptyString(values){
  for(const v of values){
    if(typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
}

function buildTranslatedItemsFromModel(parsed, items){
  const byId = new Map(items.map(item => [String(item.id), item]));
  const candidates = [];

  if(Array.isArray(parsed)) candidates.push(...parsed);
  if(parsed && typeof parsed === 'object'){
    if(Array.isArray(parsed.items)) candidates.push(...parsed.items);
    if(Array.isArray(parsed.translations)) candidates.push(...parsed.translations);
    if(Array.isArray(parsed.dishes)) candidates.push(...parsed.dishes);
    if(Array.isArray(parsed.results)) candidates.push(...parsed.results);
  }

  const matched = [];
  for(const c of candidates){
    if(!c || typeof c !== 'object') continue;
    const rawId = firstNonEmptyString([c.id, c.itemId, c.item_id, c.sourceId, c.source_id]);
    if(!rawId || !byId.has(String(rawId))) continue;
    const source = byId.get(String(rawId));
    matched.push({
      id: String(source.id),
      translatedName: firstNonEmptyString([c.translatedName, c.translated_name, c.translation, c.name]) || source.name,
      visualPrompt: firstNonEmptyString([c.visualPrompt, c.visual_prompt, c.prompt]) || dishVisualPrompt(source)
    });
  }

  // If the model omitted IDs but returned one entry per item, align by position.
  if(!matched.length && candidates.length){
    const aligned = [];
    for(let i=0; i<Math.min(items.length, candidates.length); i++){
      const c = candidates[i];
      const source = items[i];
      if(!c || typeof c !== 'object') continue;
      aligned.push({
        id: String(source.id),
        translatedName: firstNonEmptyString([c.translatedName, c.translated_name, c.translation, c.name]) || source.name,
        visualPrompt: firstNonEmptyString([c.visualPrompt, c.visual_prompt, c.prompt]) || dishVisualPrompt(source)
      });
    }
    return aligned;
  }

  // De-duplicate by id, keep first result.
  const seen = new Set();
  return matched.filter(item => {
    if(seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

function dishVisualPrompt(item){
  return item.visualPrompt || `Restaurant menu photography of ${item.sourceName || item.name}, plated beautifully, overhead or 3/4 angle, appetizing, recognizable, no text, neutral background.`;
}

function withTimeout(promise, ms, label){
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(label || 'Request timed out.')), ms);
    Promise.resolve(promise).then(
      val => {
        clearTimeout(timer);
        resolve(val);
      },
      err => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

function normalizeVisualPromptKey(prompt){
  return String(prompt || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function getVisualFromCache(prompt){
  const key = normalizeVisualPromptKey(prompt);
  if(!key) return null;
  const hit = visualImageCache.get(key);
  if(!hit) return null;
  visualImageCache.delete(key);
  visualImageCache.set(key, hit);
  return hit;
}

function setVisualInCache(prompt, imageDataUrl){
  const key = normalizeVisualPromptKey(prompt);
  const image = typeof imageDataUrl === 'string' ? imageDataUrl.trim() : '';
  if(!key || !image.startsWith('data:image/')) return;
  if(visualImageCache.has(key)) visualImageCache.delete(key);
  visualImageCache.set(key, image);
  while(visualImageCache.size > VISUAL_CACHE_MAX){
    const oldest = visualImageCache.keys().next().value;
    if(!oldest) break;
    visualImageCache.delete(oldest);
  }
}

function demoTranslation(items, targetLanguage){
  return {
    items: items.map(item => ({
      id: item.id,
      translatedName: targetLanguage === 'English'
        ? item.name
        : `${item.name} (${targetLanguage})`,
      visualPrompt: dishVisualPrompt(item)
    })),
    summary: `Demo mode translated ${items.length} dish${items.length === 1 ? '' : 'es'} into ${targetLanguage}.`
  };
}

function demoVisuals(items){
  return {
    items: items.map(item => ({
      id: item.id,
      visualPrompt: dishVisualPrompt(item),
      imageDataUrl: null
    })),
    summary: `Demo mode returned ${items.length} placeholder-ready visual prompt${items.length === 1 ? '' : 's'}.`
  };
}

/* =====================================================================
   OCR text parsing helpers (local fallback for extraction)
   ===================================================================== */

function parseDetectedPrice(value){
  if(typeof value === 'number'){
    if(!isFinite(value) || value <= 0 || value > 9999) return 0;
    return Math.round(value * 100) / 100;
  }
  if(typeof value !== 'string') return 0;
  const cleaned = value.trim().replace(/[£$€¥₹\s]/g, '');
  if(!cleaned) return 0;
  let normalized = cleaned;
  if(normalized.includes(',') && normalized.includes('.')){
    if(normalized.lastIndexOf(',') > normalized.lastIndexOf('.')){
      normalized = normalized.replace(/\./g, '').replace(',', '.');
    } else {
      normalized = normalized.replace(/,/g, '');
    }
  } else if(normalized.includes(',') && !normalized.includes('.')){
    const parts = normalized.split(',');
    if(parts.length === 2 && parts[1].length <= 2){
      normalized = parts[0] + '.' + parts[1];
    } else {
      normalized = normalized.replace(/,/g, '');
    }
  }
  const n = parseFloat(normalized);
  if(!isFinite(n) || n <= 0 || n > 9999) return 0;
  return Math.round(n * 100) / 100;
}

function normalizeMenuPrice(value){
  let price = parseDetectedPrice(value);
  if(price <= 0) return 0;
  const rawStr = typeof value === 'string' ? value.trim() : '';
  const hasCurrency = /[£$€¥₹]/.test(rawStr);
  const hasSep = /[.,]/.test(rawStr);
  if(!hasCurrency && !hasSep && /^\d{3,}$/.test(rawStr)) return 0;
  if(hasCurrency && !hasSep && /^\D*\d{3,4}$/.test(rawStr) && price >= 100 && price <= 5000){
    price = Math.round((price / 100) * 100) / 100;
  }
  if(hasCurrency && price > 300 && Number.isInteger(price) && price <= 5000){
    const scaled = Math.round((price / 100) * 100) / 100;
    if(scaled >= 0.5 && scaled <= 120) price = scaled;
  }
  if(price < 0.5 || price > 300) return 0;
  return Math.round(price * 100) / 100;
}

function dishNameKey(name){
  return String(name || '')
    .toLowerCase()
    .replace(/\([^)]*\)/g, ' ')
    .replace(/\b(x|qty)\s*\d+\b/g, ' ')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function isLikelyNonDishLine(name){
  const low = String(name || '').toLowerCase();
  if(!low) return true;
  if(/\btable\s*[#:.-]?\s*\d+\b/.test(low)) return true;
  if(/^\s*add\s+/.test(low)) return true;
  if(/^\s*(choice of|make it|served with)\b/.test(low)) return true;
  const commaCount = (low.match(/,/g) || []).length;
  const wordCount = low.split(/\s+/).filter(Boolean).length;
  if((commaCount >= 2 && wordCount >= 4) || (commaCount >= 1 && wordCount >= 6)) return true;
  const banned = [
    'subtotal','total','grand total','tax','tip','gratuity','service charge',
    'thank you','server','cash','change','balance','visa','mastercard',
    'receipt','order #','phone','tel','www','http','address',
    'food menu','main course','appetizers','drinks','free delivery',
    'packaging fee','gratuity included','take-out fee','take out fee','dine-in only','dine in only','parties of',
    'includes house chips','consumer advisory','please advise your server'
  ];
  return banned.some(token => low.includes(token));
}

function looksLikeDishName(name){
  const trimmed = String(name || '').trim();
  if(trimmed.length < 2) return false;
  if(isLikelyNonDishLine(trimmed)) return false;
  const alphaCount = (trimmed.match(/\p{L}/gu) || []).length;
  const nonSpace = trimmed.replace(/\s+/g, '').length || 1;
  if(alphaCount < 2) return false;
  if((alphaCount / nonSpace) < 0.55) return false;
  if(/[<>]/.test(trimmed)) return false;
  const words = trimmed.split(/\s+/).filter(Boolean);
  if(words.length >= 3){
    const oneCharWords = words.filter(w => w.length === 1).length;
    if(oneCharWords >= 2) return false;
  }
  return true;
}

function normalizeDishCandidate(name){
  return String(name || '')
    .replace(/^\d{1,3}[.)-]?\s+/, '')
    .replace(/^[^\p{L}\p{N}]+/gu, '')
    .replace(/[|•]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function looksLikeDishHeaderLine(line){
  const name = normalizeDishCandidate(line);
  if(!looksLikeDishName(name)) return false;
  const words = name.split(/\s+/).filter(Boolean);
  if(words.length > 9) return false;
  if(/[0-9]{4,}/.test(name)) return false;
  return true;
}

function sanitizeNameSegment(segment){
  return normalizeDishCandidate(
    String(segment || '')
      .replace(/\b\d{1,4}(?:[.,]\d{1,2})?\s*(?:ml|cl|oz|kg|g|gr|gm)\b/gi, ' ')
      .replace(/([£$€¥₹]?\s?\d{1,4}(?:[.,]\d{1,2})?)/g, ' ')
      .replace(/\b(?:ml|cl|oz|kg|g|gr|gm)\b/gi, ' ')
      .replace(/^[.·:;,\-–—\s]+/, '')
      .replace(/[.·:;,\-–—\s]+$/, '')
      // OCR sometimes prefixes uppercase dish names with tiny garbage fragments like "yy i".
      .replace(/^(?:[a-z]{1,2}\s+){1,3}(?=[A-Z][A-Z\s&()'-]{4,})/, '')
  );
}

function isMeasurementLikeToken(line, match){
  const index = match && Number.isInteger(match.index) ? match.index : -1;
  if(index < 0) return false;
  const raw = String(match[0] || '');
  const after = line.slice(index + raw.length, index + raw.length + 6).toLowerCase();
  const before = line.slice(Math.max(0, index - 2), index).toLowerCase();
  if(/^\s*(ml|cl|oz|kg|g|gr|gm)\b/.test(after)) return true;
  if(/[a-z]$/i.test(before) && /^(ml|cl|oz|kg|g|gr|gm)\b/.test(after.trim())) return true;
  return false;
}

function extractPairsFromLine(line){
  const matches = [...line.matchAll(/([£$€¥₹]?\s?\d{1,4}(?:[.,]\d{1,2})?)/g)];
  if(!matches.length) return [];
  const pairs = [];
  for(let i = 0; i < matches.length; i++){
    if(isMeasurementLikeToken(line, matches[i])) continue;
    const token = matches[i][1];
    const price = normalizeMenuPrice(token);
    if(price <= 0) continue;
    const hasCurrency = /[£$€¥₹]/.test(token);
    const hasDecimals = /[.,]\d{1,2}$/.test(token);
    if(!hasCurrency && !hasDecimals && (price < 2 || price > 500)) continue;
    let prevEnd = 0;
    for(let j = i - 1; j >= 0; j--){
      if(isMeasurementLikeToken(line, matches[j])) continue;
      prevEnd = (matches[j].index || 0) + matches[j][0].length;
      break;
    }
    const leftPart = line.slice(prevEnd, matches[i].index || 0);
    let name = sanitizeNameSegment(leftPart);
    if(!looksLikeDishName(name)){
      const nextStart = (matches[i].index || 0) + token.length;
      const nextEnd = i + 1 < matches.length ? (matches[i + 1].index || line.length) : line.length;
      name = sanitizeNameSegment(line.slice(nextStart, nextEnd));
    }
    if(!looksLikeDishName(name)) continue;
    pairs.push({ name, price });
  }
  return pairs;
}

function extractPriceFromPriceOnlyLine(line){
  const raw = String(line || '').trim();
  if(!raw) return 0;
  const residue = raw
    .replace(/([£$€¥₹]?\s?\d{1,4}(?:[.,]\d{1,2})?)/g, '')
    .replace(/[.\-–—·,\/|:()\s]/g, '');
  if(residue) return 0;
  const matches = [...raw.matchAll(/([£$€¥₹]?\s?\d{1,4}(?:[.,]\d{1,2})?)/g)];
  if(!matches.length) return 0;
  let best = 0;
  for(const m of matches){
    const price = normalizeMenuPrice(m[1]);
    if(price <= 0) continue;
    const hasCurrency = /[£$€¥₹]/.test(m[1]);
    const hasDecimals = /[.,]\d{1,2}$/.test(m[1]);
    if(!hasCurrency && !hasDecimals && (price < 2 || price > 500)) continue;
    if(price > best) best = price;
  }
  return best;
}

function parseMenuItemsFromText(text){
  const out = [];
  const seen = new Set();
  let pendingName = '';
  (text || '').split(/\r?\n/).forEach(raw => {
    const line = raw.replace(/\s+/g, ' ').trim();
    if(!line) return;
    const pairs = extractPairsFromLine(line);
    if(pairs.length){
      pairs.forEach(parsed => {
        const key = dishNameKey(parsed.name);
        if(key && !seen.has(key)){
          seen.add(key);
          out.push(parsed);
        }
      });
      pendingName = '';
      return;
    }
    const priceOnly = extractPriceFromPriceOnlyLine(line);
    if(priceOnly > 0 && pendingName){
      const key = dishNameKey(pendingName);
      if(key && !seen.has(key)){
        seen.add(key);
        out.push({ name: pendingName, price: priceOnly });
      }
      pendingName = '';
      return;
    }
    if(looksLikeDishHeaderLine(line)){
      pendingName = normalizeDishCandidate(line);
      return;
    }
    if(isLikelyNonDishLine(line)) pendingName = '';
  });
  return out;
}

function normalizeExtractedMenuItems(parsed){
  const candidates = [];
  if(Array.isArray(parsed)) candidates.push(...parsed);
  if(parsed && typeof parsed === 'object'){
    if(Array.isArray(parsed.items)) candidates.push(...parsed.items);
    if(Array.isArray(parsed.dishes)) candidates.push(...parsed.dishes);
    if(Array.isArray(parsed.results)) candidates.push(...parsed.results);
    if(Array.isArray(parsed.menu)) candidates.push(...parsed.menu);
  }

  const out = [];
  const seen = new Set();
  for(const c of candidates){
    if(!c || typeof c !== 'object') continue;
    const name = sanitizeNameSegment(firstNonEmptyString([
      c.name, c.dish, c.item, c.title, c.label, c.originalName, c.sourceName
    ]).replace(/\s+/g, ' '));
    const price = normalizeMenuPrice(c.price ?? c.amount ?? c.cost ?? c.value ?? c.menuPrice ?? c.menu_price);
    if(!looksLikeDishName(name) || price <= 0) continue;
    const key = dishNameKey(name);
    if(seen.has(key)) continue;
    seen.add(key);
    out.push({ name, price });
  }
  return out;
}

function mergeDetectedItems(primary, fallback){
  const merged = [];
  const byKey = new Map();
  [...(primary || []), ...(fallback || [])].forEach(item => {
    if(!item || typeof item !== 'object') return;
    const name = String(item.name || '').replace(/\s+/g, ' ').trim();
    const price = normalizeMenuPrice(item.price);
    if(!looksLikeDishName(name) || price <= 0) return;
    const key = dishNameKey(name);
    if(!key) return;
    const existing = byKey.get(key);
    if(!existing){
      const next = { name, price };
      byKey.set(key, next);
      merged.push(next);
      return;
    }
    if(name.length > existing.name.length + 2) existing.name = name;
    if(!existing.price || existing.price <= 0) existing.price = price;
  });
  return merged;
}

/* =====================================================================
   Routes
   ===================================================================== */

app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    provider: 'fireworks',
    configured: fireworksConfigured(),
    models: {
      vision: FIREWORKS_VISION_MODEL,
      text: FIREWORKS_TEXT_MODEL,
      image: FIREWORKS_IMAGE_MODEL
    },
    visuals: {
      provider: VISUAL_PROVIDER,
      fireworksConfigured: fireworksConfigured(),
      qwenConfigured: qwenImageConfigured()
    }
  });
});

/* ---- Menu extraction: menu photo (+ OCR text) -> dishes with prices ---- */
app.post('/api/agent/menu/extract', async (req, res) => {
  const ocrText = typeof (req.body && req.body.ocrText) === 'string' ? req.body.ocrText.trim() : '';
  const menuImageDataUrl = normalizeMenuImageDataUrl(req.body && req.body.menuImageDataUrl);
  const ocrItems = parseMenuItemsFromText(ocrText);

  if(!ocrText && !menuImageDataUrl){
    return res.status(400).json({ error: 'Provide OCR text or a menu image for extraction.' });
  }

  if(menuAgentDemoMode()){
    return res.json({
      items: ocrItems,
      summary: ocrItems.length
        ? `Demo mode detected ${ocrItems.length} dish${ocrItems.length === 1 ? '' : 'es'} with prices (OCR only, no Fireworks key set).`
        : 'Demo mode could not detect priced dishes from OCR text.'
    });
  }

  try{
    const userContent = [
      {
        type: 'text',
        text: [
          'Extract restaurant dish names and their prices from this menu.',
          'Return strict JSON only with shape: {"items":[{"name":"...","price":12.34}],"summary":"..."}',
          'Use the image as the primary source and OCR text as fallback context.',
          '',
          'OCR text:',
          ocrText || '(none)'
        ].join('\n')
      }
    ];
    if(menuImageDataUrl){
      userContent.push({
        type: 'image_url',
        image_url: { url: menuImageDataUrl }
      });
    }

    const rawText = await callFireworksChat({
      model: menuImageDataUrl ? FIREWORKS_VISION_MODEL : FIREWORKS_TEXT_MODEL,
      // Generous cap: reasoning models consume output tokens while thinking.
      maxTokens: 8000,
      temperature: 0.1,
      messages: [
        {
          role: 'system',
          content: [
            'You are TipFork\'s menu extraction agent.',
            'Extract menu dish names and numeric prices from a restaurant menu photo.',
            'Use the image as the source of truth and OCR text only as fallback context.',
            'Many menus are two-column layouts; a single OCR line can contain two dish-price pairs. Extract both pairs when present.',
            'Do not invent dishes or prices that are not visible.',
            'Ignore decorative poster text like "Restaurant menu" and section headers without prices.',
            'Skip lines that are not dishes (tax, subtotal, total, tip, address, phone, thank-you, table info).',
            'Keep dish names concise and readable; keep qualifiers when they matter.',
            'Return strict JSON only. Do not include commentary outside JSON.',
            'Use decimal numbers for price with no currency symbols.',
            'Output shape: {"items":[{"name":"...","price":12.34}],"summary":"..."}'
          ].join(' ')
        },
        { role: 'user', content: userContent }
      ]
    });

    const parsed = parseModelJsonSafely(rawText);
    const modelItems = normalizeExtractedMenuItems(parsed);
    const items = modelItems.length >= 10 ? modelItems : mergeDetectedItems(modelItems, ocrItems);

    return res.json({
      items,
      summary: parsed && typeof parsed.summary === 'string' && parsed.summary.trim()
        ? parsed.summary.trim()
        : (items.length
          ? `Detected ${items.length} dish${items.length === 1 ? '' : 'es'} with prices from the menu image.`
          : 'Could not confidently detect priced dishes from the menu image.')
    });
  }catch(error){
    // Graceful fallback: return OCR-derived items rather than failing the flow.
    if(ocrItems.length){
      return res.json({
        items: ocrItems,
        summary: `Used local OCR fallback (${ocrItems.length} dishes) because Fireworks extraction failed: ${String(error.message || '').slice(0, 140)}`
      });
    }
    return res.status(500).json({ error: error.message || 'Menu extraction failed.' });
  }
});

/* ---- Dish translation ---- */
app.post('/api/agent/menu/translate', async (req, res) => {
  const targetLanguage = (req.body && req.body.targetLanguage) || 'English';
  const items = normalizeMenuItems(req.body && req.body.items);
  const ocrTextRaw = typeof (req.body && req.body.ocrText) === 'string' ? req.body.ocrText.trim() : '';
  const ocrText = ocrTextRaw.slice(0, 1200);
  const useImageContext = !!(req.body && req.body.useImageContext);
  const menuImageDataUrl = normalizeMenuImageDataUrl(req.body && req.body.menuImageDataUrl);

  if(!items.length){
    return res.status(400).json({ error: 'No dishes were provided for translation.' });
  }

  if(/^english$/i.test(String(targetLanguage).trim())){
    const passthrough = items.map(item => ({
      id: item.id,
      translatedName: item.name,
      visualPrompt: dishVisualPrompt(item)
    }));
    return res.json({
      items: passthrough,
      summary: `Dishes are already in English.`
    });
  }

  if(menuAgentDemoMode()){
    return res.json(demoTranslation(items, targetLanguage));
  }

  try{
    const withImage = !!(useImageContext && menuImageDataUrl);
    const promptText = [
      `Target language: ${targetLanguage}`,
      '',
      'Dish items (JSON):',
      JSON.stringify(items),
      '',
      'OCR text from the same menu image:',
      ocrText || '(none)'
    ].join('\n');

    const userContent = withImage
      ? [
          { type: 'text', text: promptText },
          { type: 'image_url', image_url: { url: menuImageDataUrl } }
        ]
      : promptText;

    const rawText = await callFireworksChat({
      model: withImage ? FIREWORKS_VISION_MODEL : FIREWORKS_TEXT_MODEL,
      // Generous cap: reasoning models (e.g. gpt-oss) consume output tokens
      // while thinking before emitting the final JSON.
      maxTokens: 6000,
      temperature: 0.2,
      messages: [
        {
          role: 'system',
          content: [
            'You are TipFork\'s menu agent.',
            'Translate restaurant dish names for diners while keeping the meaning specific and helpful.',
            'Prioritize speed while keeping meanings accurate.',
            'When an image is provided, use it as primary context for dish interpretation and use OCR text only as a supporting signal.',
            'Return strict JSON only: {"items":[{"id":"...","translatedName":"...","visualPrompt":"..."}],"summary":"..."}',
            'Each item must keep its original id.',
            'Each translatedName should be concise enough to fit in a mobile list row.',
            'Each visualPrompt should describe a plated, recognizable restaurant dish with no text in the image.'
          ].join(' ')
        },
        { role: 'user', content: userContent }
      ]
    });

    const parsed = parseModelJsonSafely(rawText);
    let safeItems = buildTranslatedItemsFromModel(parsed, items);
    if(!safeItems.length){
      safeItems = items.map(item => ({
        id: String(item.id),
        translatedName: item.name,
        visualPrompt: dishVisualPrompt(item)
      }));
    }

    return res.json({
      items: safeItems,
      summary: parsed && typeof parsed.summary === 'string' && parsed.summary.trim()
        ? parsed.summary.trim()
        : `Translated ${safeItems.length} dish${safeItems.length === 1 ? '' : 'es'} into ${targetLanguage}${withImage ? ' using menu photo context' : ''}.`
    });
  }catch(error){
    console.error('[translate] Fireworks call failed:', error && error.message);
    const fallback = demoTranslation(items, targetLanguage);
    return res.json({
      ...fallback,
      summary: `Used local fallback translation because Fireworks translation failed (${String(error.message || '').slice(0, 140)}).`
    });
  }
});

/* ---- Dish visuals (FLUX.1 [schnell] via Fireworks) ---- */
app.post('/api/agent/menu/visuals', async (req, res) => {
  const items = normalizeMenuItems(req.body && req.body.items);
  const requestedMaxAi = Number.parseInt(req.body && req.body.maxAiImages, 10);
  const maxAiImages = Number.isFinite(requestedMaxAi)
    ? Math.max(0, requestedMaxAi)
    : Math.max(0, Number.parseInt(process.env.VISUALS_MAX_AI_IMAGES || '8', 10) || 8);

  if(!items.length){
    return res.status(400).json({ error: 'No dishes were provided for visual generation.' });
  }

  if(!visualsConfigured()){
    const demo = demoVisuals(items);
    demo.summary = 'No visual provider is configured (set FIREWORKS_API_KEY or QWEN_API_KEY). Returning local placeholders.';
    return res.json(demo);
  }

  try{
    const visualItems = items.map(item => ({
      id: item.id,
      visualPrompt: dishVisualPrompt(item),
      imageDataUrl: null
    }));

    let cachedCount = 0;
    const uncachedIndexes = [];
    visualItems.forEach((entry, idx) => {
      const cached = getVisualFromCache(entry.visualPrompt);
      if(cached){
        entry.imageDataUrl = cached;
        cachedCount += 1;
      } else {
        uncachedIndexes.push(idx);
      }
    });

    const targetIndexes = uncachedIndexes.slice(0, Math.min(maxAiImages, uncachedIndexes.length));
    const deadline = Date.now() + VISUALS_ROUTE_BUDGET_MS;
    let queueCursor = 0;
    let generatedCount = 0;
    let timeoutCount = 0;
    let errorCount = 0;
    let firstError = '';

    async function worker(){
      while(queueCursor < targetIndexes.length){
        if(Date.now() >= deadline) break;
        const cursor = queueCursor++;
        const currentIndex = targetIndexes[cursor];
        const item = items[currentIndex];
        const visualPrompt = visualItems[currentIndex].visualPrompt;
        try{
          const imageDataUrl = await withTimeout(
            callVisualImageGeneration(visualPrompt),
            Math.min(VISUALS_ITEM_TIMEOUT_MS, Math.max(1000, deadline - Date.now())),
            'Visual generation timed out.'
          );
          visualItems[currentIndex] = {
            id: item.id,
            visualPrompt,
            imageDataUrl
          };
          if(imageDataUrl){
            setVisualInCache(visualPrompt, imageDataUrl);
            generatedCount += 1;
          }
        }catch(err){
          const msg = String(err && err.message || '');
          if(/timed out/i.test(msg)){
            timeoutCount += 1;
          } else {
            errorCount += 1;
            if(!firstError) firstError = msg;
          }
        }
      }
    }

    if(targetIndexes.length > 0){
      const workers = Array.from({ length: Math.min(VISUALS_CONCURRENCY, targetIndexes.length) }, () => worker());
      await Promise.all(workers);
    }

    if(generatedCount === 0 && targetIndexes.length > 0 && Date.now() < deadline){
      const rescueIndexes = targetIndexes.slice(0, Math.min(2, targetIndexes.length));
      for(const currentIndex of rescueIndexes){
        if(Date.now() >= deadline) break;
        if(visualItems[currentIndex] && visualItems[currentIndex].imageDataUrl) continue;
        const item = items[currentIndex];
        const visualPrompt = visualItems[currentIndex].visualPrompt;
        try{
          const remaining = Math.max(1000, deadline - Date.now());
          const rescueTimeout = Math.min(Math.max(VISUALS_ITEM_TIMEOUT_MS, 20000), remaining);
          const imageDataUrl = await withTimeout(
            callVisualImageGeneration(visualPrompt),
            rescueTimeout,
            'Visual generation timed out.'
          );
          visualItems[currentIndex] = {
            id: item.id,
            visualPrompt,
            imageDataUrl
          };
          if(imageDataUrl){
            setVisualInCache(visualPrompt, imageDataUrl);
            generatedCount += 1;
          }
        }catch(err){
          const msg = String(err && err.message || '');
          if(/timed out/i.test(msg)){
            timeoutCount += 1;
          } else {
            errorCount += 1;
            if(!firstError) firstError = msg;
          }
        }
      }
    }

    const placeholderCount = visualItems.filter(item => !item.imageDataUrl).length;
    const notAuthorized = /unauthorized|forbidden/i.test(firstError || '');
    return res.json({
      items: visualItems,
      summary: notAuthorized
        ? 'Image generation is not authorized on the configured account (Unauthorized). Using styled placeholders — check provider credits/billing to unlock AI visuals.'
        : [
            `Generated ${generatedCount} new AI visual${generatedCount === 1 ? '' : 's'}.`,
            cachedCount ? `Reused ${cachedCount} cached visual${cachedCount === 1 ? '' : 's'}.` : '',
            placeholderCount ? `Using placeholders for ${placeholderCount} dish${placeholderCount === 1 ? '' : 'es'} for now.` : '',
            timeoutCount ? `(${timeoutCount} timed out and were skipped this round.)` : '',
            errorCount ? `(${errorCount} failed: ${(firstError || 'provider error').slice(0, 160)}.)` : ''
          ].filter(Boolean).join(' ')
    });
  }catch(error){
    return res.status(500).json({ error: error.message || 'Visual generation failed.' });
  }
});

/* Tax & tip: no endpoint needed. The app uses a free built-in rate table
   and confirms the exact tax from the receipt photo (Step 5). */

const nodeMajor = Number.parseInt(process.versions.node.split('.')[0], 10);
if(nodeMajor < 18){
  console.error(`⚠️  Node ${process.versions.node} detected — TipFork needs Node 18+ (built-in fetch). All Fireworks calls will fail on this version.`);
}
if(!fireworksConfigured()){
  console.error('⚠️  FIREWORKS_API_KEY is not set (looked for .env in ' + process.cwd() + '). Running in demo/fallback mode.');
}

app.listen(process.env.PORT || 3000, () => console.log('TipFork backend running (Fireworks AI)'));

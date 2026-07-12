/* Checks your Fireworks AI key, discovers which models your account can use,
   tests them, and prints ready-to-paste .env lines.
   Usage: node scripts/check-fireworks.js */
const fs = require('fs');
const path = require('path');

// Minimal .env loader (same behavior as backend/server.js)
const envPath = path.join(process.cwd(), '.env');
if(fs.existsSync(envPath)){
  fs.readFileSync(envPath, 'utf8').split(/\r?\n/).forEach(line => {
    const m = line.trim().match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if(m && process.env[m[1]] == null) process.env[m[1]] = m[2].trim();
  });
}

const KEY = process.env.FIREWORKS_API_KEY;
const BASE = (process.env.FIREWORKS_BASE_URL || 'https://api.fireworks.ai/inference/v1').replace(/\/+$/, '');

if(!KEY){
  console.error('❌ FIREWORKS_API_KEY is not set. Copy .env.example to .env and fill it in.');
  process.exit(1);
}

// 1x1 transparent PNG — used to verify a model truly accepts image input.
const TINY_IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

async function listModels(){
  const res = await fetch(`${BASE}/models`, {
    headers: { 'Authorization': `Bearer ${KEY}` }
  });
  if(!res.ok){
    const text = await res.text().catch(() => '');
    throw new Error(`GET /models failed: ${res.status} ${text.slice(0, 200)}`);
  }
  const json = await res.json();
  const models = (json.data || json.models || []).map(m => m.id || m.name).filter(Boolean);
  return models;
}

async function testChat(model, withImage){
  const content = withImage
    ? [{ type: 'text', text: 'Reply with the single word: ok' }, { type: 'image_url', image_url: { url: TINY_IMAGE } }]
    : 'Reply with the single word: ok';
  const res = await fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, max_tokens: 20, messages: [{ role: 'user', content }] })
  });
  if(res.ok) return { ok: true };
  const json = await res.json().catch(() => ({}));
  return { ok: false, status: res.status, msg: JSON.stringify(json).slice(0, 160) };
}

async function testImage(model){
  const res = await fetch(`${BASE}/workflows/${model}/text_to_image`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${KEY}`, 'Content-Type': 'application/json', 'Accept': 'image/jpeg' },
    body: JSON.stringify({ prompt: 'A plated margherita pizza, overhead shot, no text' })
  });
  if(res.ok){
    const bytes = (await res.arrayBuffer()).byteLength;
    return { ok: true, kb: Math.round(bytes / 1024) };
  }
  const text = await res.text().catch(() => '');
  return { ok: false, status: res.status, msg: text.slice(0, 160) };
}

function looksVision(id){
  return /(-vl|vl-|vision|llama4|kontext-dev-vlm)/i.test(id);
}
function looksImageGen(id){
  return /(flux|stable-diffusion|sdxl|-image|playground-v|dall)/i.test(id);
}
function looksEmbeddingOrAudio(id){
  return /(embed|whisper|audio|asr|tts|rerank|guard|moderat)/i.test(id);
}

(async () => {
  console.log('Checking Fireworks AI setup…\n');

  let models;
  try{
    models = await listModels();
  }catch(err){
    console.error(`❌ Could not list models: ${err.message}`);
    console.error('   If this is a 401, the API key is invalid — generate a new one at https://fireworks.ai');
    process.exit(1);
  }
  console.log(`✅ Key is valid. Your account can see ${models.length} models:`);
  models.forEach(m => console.log(`   - ${m}`));
  console.log('');

  const imageGen = models.filter(looksImageGen);
  const chatModels = models.filter(m => !looksImageGen(m) && !looksEmbeddingOrAudio(m));
  // Probe every chat model for image support — multimodal models don't always have "vl" in the name.
  const vision = chatModels;
  const text = chatModels;

  // Prefer instruct-style, reasonably sized models first for text.
  const rank = arr => [...arr].sort((a, b) => {
    const score = id => (/instruct|chat/i.test(id) ? -2 : 0) + (/\b(7b|8b|17b|30b|32b|70b|a3b|a22b)\b/i.test(id) ? -1 : 0);
    return score(a) - score(b);
  });

  const envSuggestions = {};

  // --- Vision (menu extraction): probe every chat model with a real image ---
  console.log('— Vision model (menu extraction) —');
  for(const m of rank(vision).slice(0, 10)){
    const r = await testChat(m, true);
    if(r.ok){ console.log(`  ✅ ${m} accepts image input`); envSuggestions.FIREWORKS_VISION_MODEL = m; break; }
    console.log(`  ❌ ${m} (${r.status}) ${r.msg || ''}`);
  }
  if(!envSuggestions.FIREWORKS_VISION_MODEL) console.log('  ⚠️  No working vision model found. Extraction will use OCR fallback.');

  // --- Text (translation) ---
  console.log('\n— Text model (translation) —');
  for(const m of rank(text).slice(0, 6)){
    const r = await testChat(m, false);
    if(r.ok){ console.log(`  ✅ ${m}`); envSuggestions.FIREWORKS_TEXT_MODEL = m; break; }
    console.log(`  ❌ ${m} (${r.status})`);
  }
  if(!envSuggestions.FIREWORKS_TEXT_MODEL) console.log('  ⚠️  No working text model found.');

  // --- Image generation (dish visuals) ---
  console.log('\n— Image model (dish visuals) —');
  if(!imageGen.length) console.log('  ⚠️  No image-generation models visible to this account.');
  let sawImage401 = false;
  for(const m of imageGen.slice(0, 5)){
    const r = await testImage(m);
    if(r.ok){ console.log(`  ✅ ${m} (received ${r.kb} KB)`); envSuggestions.FIREWORKS_IMAGE_MODEL = m; break; }
    if(r.status === 401 || r.status === 403) sawImage401 = true;
    console.log(`  ❌ ${m} (${r.status}) ${r.msg}`);
  }
  if(!envSuggestions.FIREWORKS_IMAGE_MODEL){
    console.log('  ⚠️  No working image model. Visuals will use styled placeholders.');
    if(sawImage401){
      console.log('     A 401/403 on image models usually means your account needs credits or');
      console.log('     billing enabled for image generation: https://app.fireworks.ai/billing');
    }
  }

  // --- Result ---
  const keys = Object.keys(envSuggestions);
  if(keys.length){
    console.log('\nAdd these lines to your .env (replacing any existing ones):\n');
    keys.forEach(k => console.log(`${k}=${envSuggestions[k]}`));
    console.log('\nThen restart the backend: node server.js');
  } else {
    console.log('\nNo working models found — check your account plan at https://fireworks.ai');
  }
})();

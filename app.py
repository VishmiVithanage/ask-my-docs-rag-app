# app.py - minimal web UI for rag.py.   Run:  python app.py docs/   then open http://127.0.0.1:5000

import sys

from flask import Flask, jsonify, request

import rag as rag

app = Flask(__name__)
INDEX: rag.Index | None = None

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ask your documents</title>
<style>
 body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1b1b1b}
 form{display:flex;gap:.5rem} input{flex:1;padding:.6rem;font:inherit;border:1px solid #999;border-radius:6px}
 button{padding:.6rem 1rem;font:inherit;border:0;border-radius:6px;background:#1b4fd8;color:#fff;cursor:pointer}
 #ans{white-space:pre-wrap;margin:1.2rem 0} .warn{color:#8a4b00;font-size:.9rem}
 .src{font-size:.9rem;color:#444;margin:.3rem 0} details{margin:.4rem 0}
</style>
<h1>Ask your documents</h1>
<form id="f"><input id="q" placeholder="e.g. How many PTO days can I carry over?" autofocus><button>Ask</button></form>
<div id="ans"></div><div id="warn" class="warn"></div><div id="src"></div>
<script>
const $=id=>document.getElementById(id);
$('f').onsubmit=async e=>{
  e.preventDefault(); $('ans').textContent='Thinking…'; $('warn').textContent=''; $('src').replaceChildren();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:$('q').value})});
    const d=await r.json(); $('ans').textContent=d.answer||d.error;
    $('warn').textContent=(d.warnings||[]).join(' · ');
    (d.sources||[]).forEach((s,i)=>{
      const det=document.createElement('details'), sm=document.createElement('summary'), p=document.createElement('div');
      sm.textContent='['+(i+1)+'] '+s.label; p.className='src'; p.textContent=s.text; det.append(sm,p); $('src').append(det);
    });
  }catch(err){$('ans').textContent='Request failed: '+err}
};
</script>"""


@app.get("/")
def home():
    return PAGE


@app.post("/ask")
def ask():
    q = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not q:
        return jsonify(error="Please enter a question."), 400
    if len(q) > 1000:
        return jsonify(error="Question too long (max 1000 characters)."), 400
    a = rag.answer(INDEX, q)
    return jsonify(
        answer=a.text, mode=a.mode, warnings=a.warnings,
        sources=[{"label": h.chunk.label(), "text": h.chunk.text} for h in a.sources])


if __name__ == "__main__":
    import app as _self  # keep a single module instance (see rag.py note on pickling)
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    _self.INDEX, rebuilt = rag.build_index(folder)
    print(f"{'Indexed' if rebuilt else 'Loaded index for'} {len(_self.INDEX.chunks)} chunks")
    _self.app.run(host="127.0.0.1", port=5000)
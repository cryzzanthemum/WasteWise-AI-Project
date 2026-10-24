# ♻️ WasteWise AI — From Waste to Resource

**SOCIOTHON 2026 · Waste-management track**

> Most apps identify waste. **WasteWise AI** identifies it, tells you the correct disposal stream, books a pickup, and tracks whether the material was **actually recovered**, backed by evidence.

---

## 1. Problem statement
Campuses generate large volumes of mixed waste. Segregation at source is poor because people don't know which of India's four streams an item belongs to (wet, dry, sanitary or special-care). Even when waste is segregated, nobody tracks whether it was collected, or whether it reached a recycler, composter or authorised handler. "Recycling" numbers are usually just **reported** numbers.

## 2. Solution & differentiator
WasteWise AI closes the loop in one app:

**Scan → Segregate → Request pickup → Collect & weigh → Confirm recovery with evidence → Measure impact**

| Typical waste apps | WasteWise AI |
|---|---|
| Identify the object | Identify the material **and** run context checks (food-soiled? sanitary use? battery? chemicals? broken?) to decide the stream |
| Stop at "put it in the blue bin" | Creates a pickup request stored in a database |
| No accountability | Collector workflow: Reported → Accepted → Collected → Recovery Confirmed |
| Count reported waste as "recycled" | Recovery counts **only** with evidence (file or receipt number). Simulated demo confirmations are reported separately |

## 3. Architecture
```
┌─────────────────────────── Streamlit (main.py) ───────────────────────────┐
│  AI Scanner ─► Guidance engine ─► Pickup form ─► Collector dashboard ─►   │
│      │          (material + context      │          (status workflow,      │
│      │           → 4-stream rules)        │           weights, evidence)    │
│      ▼                                    ▼                  │             │
│  ONNX Runtime                         SQLite (wastewise.db)◄─┘             │
│  EfficientNet-Lite0 (ImageNet)        pickups · status_log · scans         │
│  → 1000-class probs → material map                  │                      │
│                                                     ▼                      │
│                                     Impact analytics (Pandas + Plotly)     │
└────────────────────────────────────────────────────────────────────────────┘
```
* **One Python file**, no separate backend, no cloud, no API keys, no login.
* The model runs **fully offline** on the laptop CPU (under 50 ms per image).
* Every status change goes to an **audit log** (`status_log` table).

### How the AI works (honestly)
1. A pretrained **EfficientNet-Lite0** (ImageNet, 1,000 classes; Google MediaPipe model, Apache-2.0), converted to ONNX, classifies the photo.
2. About 150 relevant ImageNet classes (e.g. *water bottle, beer bottle, carton, envelope, banana, diaper, cellular telephone, syringe*) are mapped to materials.
3. **Material score** = the sum of the probabilities of all classes mapped to that material. The app shows the raw top-5 classes as evidence.
4. Score ≥ 45% gives an *AI suggestion*. 20–45% gives *low confidence, please confirm*. Below 20% gives *not recognised*, and the user selects manually. **No prediction is ever made up.** If the model can't load, the app says so and switches to manual mode.
5. The user always confirms or corrects the material. Each scan records whether it was *AI (confirmed)*, *AI (corrected)* or *Manual*.

## 4. Technologies
Python · Streamlit · SQLite · ONNX Runtime · EfficientNet-Lite0 (pretrained) · NumPy · Pandas · Plotly · Pillow

## 5. Setup (Windows)
**Prerequisite:** Python **3.11 or 3.12** from python.org, with "Add python.exe to PATH" ticked during install.

**Option A, one click:** unzip the folder and double-click **`run_windows.bat`**. The first run installs dependencies (about 1–2 minutes). After that the browser opens at http://localhost:8501.

**Option B, manual (PowerShell or Command Prompt):**
```bat
cd path\to\wastewise
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```
If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt instead.

The database `wastewise.db` is created automatically on first launch, and it keeps your data across restarts. Delete the file to start fresh.

## 6. Features
* **AI Waste Scanner**: upload a photo or use the webcam, get a material suggestion with a transparent score, raw model evidence and a manual fallback.
* **Smart disposal guidance**: wet, dry, sanitary or special-care stream, bin colour, handling steps, recovery route, whether a pickup is appropriate, plus **hazard and uncertainty flags for human review**.
  * Context rules, for example: food-soiled paper becomes **wet**. Any item used for hygiene becomes **sanitary**. Anything with a battery or electronics becomes **special-care**. Anything that held chemicals or medicines, or an aerosol, becomes **special-care**. Broken glass gets **sharp-handling** steps.
* **Pickup requests**: campus location or hostel, category, weight and notes, saved in SQLite. Requests started from a scan are pre-filled.
* **Collector dashboard**: queues by status. Accept → Collect (record the measured weight) → Confirm recovery (facility, recovered weight ≤ collected weight, evidence file / receipt number / explicit **SIMULATED** checkbox). Includes a full audit trail.
* **Impact analytics**: KPI cards (reported, collected, recovery confirmed with evidence-backed and simulated shown separately, pending, recovery rate, safely disposed, flagged items, scans). Charts: waste by category, reported vs collected vs recovered per stream, daily trend, pipeline funnel, location hotspots. Also a CSV export.
* **DEMO DATA**: one sidebar click loads 28 sample requests, each marked **DEMO DATA**. Demo recoveries use simulated evidence. A toggle hides demo data from every metric, and another click clears it without touching real records.

## 7. Data integrity rules
* **Reported** weight = the requester's estimate. **Collected** weight = measured by the collector.
* **Recovered** is counted only for *Recovery Confirmed* records, and recovered weight can never exceed collected weight.
* Simulated confirmations are shown separately from evidence-backed ones.
* Sanitary waste is *safely disposed*, never counted as recovered.
* Status transitions are enforced in code. You cannot collect before accepting, or confirm before collecting.

## 8. Limitations
* ImageNet has no dedicated "crushed can" or "cardboard sheet" classes, so some recyclables (especially aluminium cans and flat cardboard) can come back as low confidence. That is why manual confirmation is built in.
* The model sees **objects, not contamination**. Context checks are answered by the user.
* There are no user accounts. The collector dashboard is open by design for the MVP.
* Evidence files are stored locally and are not verified automatically.
* Weights are entered manually (no scale integration).

## 9. Impact metrics tracked
Reported kg · Collected kg · Recovery-confirmed kg (evidence vs simulated) · Recovery rate (recovered ÷ collected) · Pending pickups · Items flagged for review · Safely-disposed sanitary kg · AI-assisted vs manual scans · Hotspot locations · Daily trend.

## 10. Affordable campus deployment
* Runs on **one existing PC or a ₹4–6k mini-PC** on the campus LAN (`streamlit run main.py --server.address 0.0.0.0`). Students open it on their phones via a QR code at each bin cluster.
* No cloud bills, no API costs, no special hardware. The model is 18 MB and runs on the CPU.
* Collectors (housekeeping staff) use the same URL on a shared phone. Recovery evidence = a photo of the scrap dealer / MRF weighment slip.

## 11. Future improvements
* Fine-tune the model on TrashNet/TACO and Indian waste photos for cans, cardboard, MLP packets and tetra-packs.
* Role-based logins for students, collectors and admin. WhatsApp/SMS notifications.
* Route optimisation for collectors. Bin-fill-level QR check-ins.
* Recycler-side portal to upload weighment slips directly, plus OCR verification of receipts.
* CO₂-avoided estimates per material. Hostel leaderboards and gamification.
* Integration with municipal (ULB) dashboards and EPR credit tracking for plastic.

## Project files
```
main.py                 the whole app
requirements.txt        Python dependencies
run_windows.bat         one-click Windows setup + launch
models/efficientnet_lite0.onnx   pretrained model (offline)
models/imagenet_labels.txt       1,000 ImageNet labels
.streamlit/config.toml  theme
PITCH.md                60-second pitch, demo script, judge Q&A
docs/INTERVIEW_PREP.md  interview guide: talking points, who says what, FAQ
```

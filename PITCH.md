# WasteWise AI: Pitch Kit

## 60-second pitch
"Every campus has colour-coded bins, and every campus still has mixed waste. People don't know whether a greasy pizza box, a used tissue or a dead power bank goes in wet, dry, sanitary or special-care. And even when they segregate correctly, nobody can tell you whether that waste was collected, let alone recycled.

**WasteWise AI closes that loop.** Snap a photo: a pretrained vision model running fully offline suggests the material, and it shows its confidence honestly. Two quick context checks then map the item to India's four-stream system, with handling steps and hazard flags. One tap books a pickup. The collector accepts it, weighs it, and confirms recovery **only with evidence**, such as a weighment slip or receipt.

So our dashboard separates what was *reported*, what was *collected*, and what was *verifiably recovered*. Most apps stop at the first number. We track all three.

It runs on any laptop: no cloud, no API keys, no new hardware. A campus can deploy it tomorrow for close to zero cost. WasteWise AI: from waste to resource, with proof."

---

## 2–3 minute demo script
1. **Sidebar → Load DEMO DATA** (before going on stage) so the analytics already have history. Point out the yellow DEMO DATA badges.
2. **AI Scanner** → upload a photo of a plastic/glass bottle. Show the *AI SUGGESTION* badge, the material score and **Model evidence** (raw top-5 classes).
3. Tick "Contaminated with food" to show the guidance change (rinse step). Or switch to Paper + food to show the stream flip to **Wet**.
4. Click **Request pickup for this item** → choose *Boys Hostel A*, weight 2 kg → Submit. Note the request number.
5. **Collector Dashboard** → *Reported* queue → **Accept**. Then *Accepted* → enter the weighed kg → **Mark collected**.
6. *Collected* queue → upload a photo of a receipt (or tick SIMULATED) → **Confirm**. Point out that recovered can't exceed collected, and that the audit trail logs every step.
7. **Impact Analytics** → show the KPIs change. Toggle **Include DEMO DATA off** to show only your live record. Evidence-backed and simulated kg are shown separately.
8. Close with the deployment story: a campus PC, a QR code at each bin cluster, housekeeping staff as collectors, a scrap dealer slip as evidence.

**Backup:** if a photo isn't recognised, that is a feature. Show the honest "Not recognised" state and the manual selection.

---

## Likely judge questions

**Q: Is the AI real, or hard-coded?**
A: Real. It's a pretrained EfficientNet-Lite0 (ImageNet) running locally via ONNX Runtime. We map about 150 ImageNet classes to materials and sum their probabilities. The raw top-5 predictions are shown on screen as evidence.

**Q: Why not train your own waste model?**
A: In 5 hours, a rushed custom model would be less reliable than a well-tested pretrained one. The architecture is model-agnostic: swapping in a model fine-tuned on TrashNet/TACO is a one-file change.

**Q: What happens when the model is wrong?**
A: Low scores are flagged "please confirm" or "not recognised". The user always confirms or corrects, and we log whether each decision was AI-confirmed, AI-corrected or manual. That log becomes training data later.

**Q: Why ask context questions? Can't the AI decide the stream?**
A: Material isn't the same as stream. A clean paper cup is dry, a food-soiled one is wet, and a tissue used for hygiene is sanitary. A camera can't see contamination or past use, so we ask, and the rules engine decides.

**Q: How do you stop fake "recycled" numbers?**
A: Status transitions are enforced (no recovery before collection). Recovered weight can't exceed collected weight. Confirmation needs an evidence file or receipt number. Simulated confirmations are counted separately, and sanitary waste is never counted as recovered.

**Q: How is the demo data separated from real data?**
A: Each record has an `is_demo` flag. It's badged in every table, there's a one-click toggle to exclude it from all metrics, and it can be cleared without touching real records.

**Q: Cost to deploy on a campus?**
A: Close to zero. It runs on an existing PC or a cheap mini-PC on the LAN. Students use their phones via a QR code. There are no cloud or API fees, and the 18 MB model runs on the CPU.

**Q: Privacy?**
A: Photos are processed in memory on the local machine and are not stored or sent anywhere. Only the evidence slips the collectors upload are saved.

**Q: How does it scale beyond one campus?**
A: Move SQLite to Postgres, add role-based logins, and give each ward or campus its own location list. The workflow and metrics map directly to ULB reporting and plastic EPR tracking.

**Q: What's the biggest limitation today?**
A: ImageNet lacks some waste-specific classes (crushed cans, flat cardboard, MLP packets), so those often need manual confirmation. Fine-tuning on Indian waste images is our first next step.

**Q: What makes this different from existing apps?**
A: Most stop at identification. We connect identification to segregation rules, collection logistics and evidence-backed recovery. That lets a campus report verified impact, not estimates.

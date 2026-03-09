# Product Requirements Document
## Torque — Robot Learning Data Infrastructure

```
import torq as tq          # pip install torq-robotics
```

| Field | Value |
|---|---|
| **Product** | Torque — Robot Learning Data Infrastructure |
| **Version** | 3.1 |
| **Date** | 2026-03-01 |
| **Classification** | Confidential |
| **Author** | Ayush |
| **Status** | Active — Validation Phase |
| **Company** | Centroid Foundry |
| **Install** | `pip install torq-robotics` |
| **Import** | `import torq as tq` |
| **Cloud Platform** | https://www.datatorq.ai |

**v3.0 Change Summary:** Added Executive Summary, Success Criteria, Product Scope, User Journeys, and Gravity Well Conversion Model sections. Package naming locked: `pip install torq-robotics` / `import torq as tq`. Cloud platform confirmed: www.datatorq.ai. Quality metric thresholds deferred to Phase 1 user validation. All `import torque` references updated to `import torq`.

**v3.1 Change Summary:** BMAD validation edits applied. Two HIGH contradictions resolved: ML-02 priority corrected P0→P1 (JAX is R2); GW-SDK-06 telemetry separated into GW-SDK-07 and deferred to R2 (network call conflict). SC-1.5 hiring gate added. SC-4.1 range replaced with specific minimum. R-03 probability updated LOW→MEDIUM. Decisions 1 and 3 closed. UJ-03 (Sarah Kim) added as R2 stub. FR traceability added to all user journeys. Neuracore intervention scenario added. Measurement methods sharpened for QM-01, QE-01, DI-03.

---

## Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-02-10 | Initial PRD — Alloy rated LOW-MED threat, feature priorities based on market white space |
| 2.0 | 2026-02-16 | COMPETITIVE REVISION: Alloy elevated to HIGH threat. Architecture redesigned for defensibility. New sections: Competitive Defense Architecture, Alloy Intervention Analysis, Accelerated Differentiation Roadmap. Risk matrix updated. |
| 3.0 | 2026-02-26 | VALIDATION REVISION: Added Executive Summary, Success Criteria, Product Scope, User Journeys (Maya + Jake), and Gravity Well Conversion Model. Package naming locked. Cloud platform: www.datatorq.ai. Quality thresholds deferred to Phase 1. |
| 3.1 | 2026-03-01 | BMAD VALIDATION EDITS: Fixed ML-02 priority P0→P1 (JAX is R2, not R1). Separated GW-SDK-06 telemetry into GW-SDK-07 deferred to R2 (network call conflict with R1 constraint). SC-4.1 range replaced with specific minimum gate. SC-1.5 hiring gate added. R-03 probability LOW→MEDIUM (aligns with Scenario D 60%). Scenario D impact updated. Neuracore intervention scenario added. UJ-03 stub added for Sarah Kim. Jake cloud streaming story flagged R2. Decisions 1 and 3 closed. FR traceability added to user journeys. NFR-P02 architecture dependency noted. QM-01, QE-01, DI-03 measurement methods sharpened. |

---

## 0. Executive Summary

**Product:** Torq — Robot Learning Data Infrastructure
**Tagline:** Turn data into motion
**Stage:** R1 Alpha — SDK complete, entering user validation
**Company:** Centroid Foundry Pty Ltd

**One-liner:**
> The pre-deployment data infrastructure for robot learning — from raw teleoperation recordings to ML-ready training DataLoaders in 5 lines of code.

**Positioning:**
> Alloy is Datadog for deployed robots. Torq is Weights & Biases for robot learning. They monitor production. We optimize training. Both are needed, neither replaces the other.

### Problem

40–60% of robotics engineering time is spent on data pipeline plumbing — format conversion, quality filtering, dataset balancing — using 3–6 incompatible tools. No single system manages the complete pre-deployment lifecycle: raw recordings → structured episodes → quality-scored datasets → ML training DataLoaders.

### Solution

Torq owns the full workflow. `tq.ingest()` handles any format. `tq.quality.score()` automatically filters bad demonstrations. `tq.compose()` builds training datasets with one API call. `tq.DataLoader()` streams directly to PyTorch or JAX. The 250-line custom data loading script every PhD student maintains becomes 5 lines of Torq.

### Target Users

| Tier | User | Context |
|---|---|---|
| Primary | Robot learning researchers | Academic labs, CoRL/RSS/ICRA |
| Secondary | ML engineers at robotics startups | Series A, manipulation/locomotion |
| Tertiary | Sim-to-real research leads | Managing synthetic + real data pipelines |

### Current Status

| Metric | Value |
|---|---|
| SDK Complete | Yes |
| SDK Tests Passing | 123 |
| SDK Lines of Code | 3,467 |
| Real-World Users | 0 |
| Revenue | $0 (pre-revenue) |
| Next Milestone | 5 active SDK users on real data — Phase 1 gate |

---

## 0.1 Success Criteria

> Success is defined by evidence of real user value, not features shipped. Each phase gate must be satisfied before proceeding to the next phase. All criteria are observable and verifiable — not self-assessed.

### Phase 1 — SDK Validation (M0–3)

**Theme:** Does the SDK work on real data and does quality scoring mean something?

| ID | Criterion | Measurement | Why |
|---|---|---|---|
| SC-1.1 | 5 or more users have run `tq.ingest()` on their own real datasets (not synthetic fixtures) without hitting an unrecoverable error. | Confirmed via pair-programming session notes or user-reported. | Validates the SDK handles real-world messy data, not just clean synthetic test fixtures. |
| SC-1.2 | At least 2 users have run `tq.quality.score()` and confirmed that the score ordering matches their own intuition about which episodes are good and which are bad. | Direct observation during pair-programming: user sees scores, ranks episodes themselves, overlap ≥ 70%. | Quality scoring that doesn't match human judgment is worse than no scoring — it actively misleads. This is the most critical validation in Phase 1. |
| SC-1.3 | At least 1 user has run the full 5-line workflow end-to-end: ingest → score → compose → DataLoader → training loop iteration. | Confirmed via session observation or user-shared code snippet. | The README promise. If no real user completes this flow, R1 is not validated. |
| SC-1.4 | Zero unrecoverable errors on MCAP or HDF5 files from real teleoperation hardware (ALOHA-2, Franka, or equivalent). | All errors during pair-programming sessions are recoverable with a clear error message. No silent data corruption. | First impression for academic users. One silent failure destroys trust permanently. |
| SC-1.5 | First engineering hire (employee or contractor) engaged by end of Month 2. | Signed employment contract or active contractor agreement. | R-07 is rated HIGH probability, CRIT impact. The Phase 1 feature scope (ingest + quality + compose + DataLoader + CLI + acceptance tests) is not achievable solo in M0–3 without this hire. |

### Phase 2 — Cloud MVP (M3–7)

**Theme:** Will anyone pay for the cloud platform?

| ID | Criterion | Measurement | Why |
|---|---|---|---|
| SC-2.1 | 3 or more paying customers at any pricing tier above free. | Stripe or equivalent payment confirmation. | Revenue is the only proof that cloud value is real. Waitlist signups do not count. |
| SC-2.2 | A new user can go from `pip install torq-robotics` to viewing a quality report on datatorq.ai in under 10 minutes. | Timed observation of 3 onboarding sessions. Median under 10 minutes. | Onboarding time is the strongest predictor of activation rate for developer tools. |
| SC-2.3 | At least 1 dataset has been shared between two users via the cloud platform and used in a downstream training run. | Confirmed via platform activity log. | Validates the core multi-player value proposition that no local SDK can replicate. |
| SC-2.4 | At least 10 unique users have clicked through a gravity well prompt to datatorq.ai. | Confirmed via datatorq.ai analytics. | Validates gravity wells are generating conversion traffic, not printing to ignored terminals. |

### Phase 3 — Commercial Traction (M7–12)

**Theme:** Can we grow beyond academics into startups?

| ID | Criterion | Measurement | Why |
|---|---|---|---|
| SC-3.1 | $10,000 MRR from paying customers. | Monthly recurring revenue from Stripe. | Minimum signal that this is a real business. |
| SC-3.2 | Net Promoter Score > 40 across active users. | In-product NPS survey, minimum 20 respondents. | NPS > 40 is the threshold for a product users actively recommend. |
| SC-3.3 | 3 or more robotics startup customers (distinct from academic users) using Torq in a production training pipeline. | Confirmed via customer interviews. | Academic adoption validates the tool. Startup adoption validates the business model. |

### Phase 4 — Scale and Defend (M12–18)

| ID | Criterion | Measurement | Why |
|---|---|---|---|
| SC-4.1 | At least $30,000 MRR from paying customers. | Monthly recurring revenue from Stripe. | Minimum gate, not a range. $80,000 is an aspirational ceiling but a range has no pass/fail threshold. |
| SC-4.2 | At least 1 enterprise customer in active contract negotiation or signed agreement. | Signed LOI or contract. | Enterprise pipeline validates the upper pricing tier and on-premise deployment path. |
| SC-4.3 | Alloy integration live: production robot data flowing from Alloy into Torq for retraining without manual export. | Integration confirmed by at least 1 mutual customer. | Transforms the competitive relationship from threat to complementary partnership. **External dependency risk:** this gate requires Alloy's cooperation. If Alloy declines, Phase 4 cannot close on SC-4.3 alone. Track as strategic dependency alongside R-08. |

---

## 0.2 Product Scope

### R1 Alpha — In Scope

- MCAP/ROS 2 ingestion with automatic topic discovery and episode segmentation
- HDF5 (Robomimic), LeRobot v3.0 (Parquet+MP4) ingestion
- Automatic format detection from file extension and magic bytes
- Multi-rate temporal alignment across mismatched sensor frequencies
- Episode quality scoring: smoothness, consistency, completeness (thresholds TBD via Phase 1)
- Composite QualityReport with per-dimension breakdown
- Dataset composition via `tq.compose()` with quality filtering and stratified sampling
- PyTorch DataLoader (`tq.DataLoader`) with multi-worker support
- Numpy fallback iterator for non-PyTorch users
- Parquet + MP4 + JSON storage with master index
- CLI commands: `tq ingest`, `tq list`, `tq info`, `tq export`
- `tq.cloud()` stub directing users to https://www.datatorq.ai
- Gravity well prompts at `tq.quality.score()`, `tq.compose()`, `tq.DataLoader()`

### R1 Alpha — Out of Scope

- Cloud platform (datatorq.ai) — local-only in R1
- Network calls of any kind — zero network dependency
- User authentication or accounts
- Team workspaces or dataset sharing
- Community quality benchmarks (requires cloud)
- Natural language search
- Sim-to-real mixing (P1 — R2)
- JAX/Flax pipeline (P0 in R2, not R1)
- W&B / MLflow integration (R2)
- Policy checkpoint registry (R2)
- Training job orchestration (R2)
- Database of any kind — JSON index only
- Windows native support (WSL2 only)

### Hardware-Agnostic Scope

Torq does not know what robot produced an episode. The Episode object is a generic container. An ALOHA-2 episode and a Franka episode have identical structure. This is a design constraint, not a limitation — it is the hardware-agnostic moat.

### Open-Core Boundary

| Classification | Rule | Example |
|---|---|---|
| **SDK — Free Forever** | Airplane Test: works fully offline, delivers complete value | `tq.quality.score()` runs offline |
| **SDK — Free Forever** | Single-Player Test: full value for one person | `tq.compose()` builds a local dataset |
| **Cloud Only** | Multi-Player Test: needs collaboration or team coordination | Sharing a dataset with a lab collaborator |
| **Cloud Only** | Impossible Locally: requires aggregate data or cloud-scale compute | Comparing your quality score against community median |

---

## 1. Strategic Context

### Alloy Threat Reassessment

> **Previous assessment:** LOW-MED threat
> **Current assessment:** HIGH threat

Alloy's February 2025 launch, AUD $4.5M pre-seed (Blackbird + Airtree), 4 Australian design partners, and explicit 'Scenarios Become Training Data' feature signal a clear intent to bridge from operational monitoring into training data infrastructure. Their MCAP/ROS ingestion pipeline, natural language search, and multi-modal data unification share the same front door as Torque. The question is no longer whether Alloy will move upstream into training — it's when.

**Critical Strategic Assumption:**
> Alloy can add crude training data export features (scenario-to-dataset, basic DataLoader export) within 6–12 months. Their architecture, built on mission/scenario ontology, cannot easily support episode quality scoring, dataset composition, sim-to-real mixing, or cross-embodiment normalization without a fundamental product rewrite. Torque's defensibility depends on going DEEPER into ML training workflows, not wider into general data management. Every feature in this PRD is evaluated against the question: 'Could Alloy replicate this by extending their existing architecture?'

### Product Vision

Torque is the data infrastructure platform for robot learning — managing the full lifecycle from episode recording through dataset curation to policy training. It captures understanding, not just trajectories, enabling robots to learn faster from fewer demonstrations.

**Architectural Mandate:** This framing is not just a talking point — it's an architectural mandate. Every technical decision in this PRD reinforces the boundary between operational monitoring and training optimization.

### Competitive Position Summary

| Dimension | Alloy | Torq |
|---|---|---|
| Ontology | Missions & Scenarios (operational) | Episodes & Demonstrations (training) |
| Primary User | Fleet ops engineer, field technician | ML researcher, robot learning engineer |
| Core Workflow | Deploy → Monitor → Debug → Report | Demonstrate → Ingest → Score → Curate → Train |
| Data Model | Time-series logs, sensor streams, video | Structured episodes with action-observation pairs, quality metadata |
| Output | Bug reports, fleet dashboards, anomaly clusters | Training datasets, PyTorch DataLoaders, trained policies |
| Overlap Zone | MCAP ingestion, NL search, scenario export | MCAP ingestion, NL search, data browsing |
| Alloy Cannot | — | Episode quality scoring, dataset composition, sim-to-real mixing, cross-embodiment normalization, training pipeline integration, policy experiment tracking |

---

## 2. Target Users & Personas

### Primary — Dr. Maya Chen, Robot Learning Researcher

**Role:** Senior Research Scientist at a university or corporate robotics lab
**Background:** PhD in robotics or ML, publishes at CoRL/RSS/ICRA

**Daily Workflow:** Collecting teleoperation demonstrations on ALOHA-2 or similar hardware, training VLA/ACT policies, iterating on dataset composition.

**Pain Points:**
- Spends 40% of time on data plumbing (format conversion, quality filtering, dataset balancing)
- Uses 3–6 incompatible tools (Foxglove for viz, custom scripts for filtering, LeRobot format for training, W&B for experiment tracking)

**Why Alloy Doesn't Serve Maya:**
Maya doesn't operate deployed robots. She doesn't have 'missions' or 'fleet anomalies.' She has teleoperation demonstrations that need quality assessment and composition into training datasets. Alloy's entire workflow is irrelevant to her daily work.

**User Stories:**
- As a researcher, I want to import episodes from MCAP, HDF5, or LeRobot format with a single CLI command so I can start curating data immediately without writing conversion scripts.
- As a researcher, I want automated quality scores on every demonstration so I can filter out bad episodes before they contaminate my training dataset.
- As a researcher, I want to compose a training dataset by specifying task distribution, quality threshold, and sim-to-real ratio in a single API call.

---

### Secondary — Jake Torres, ML Engineer at a Robotics Startup

**Role:** ML Engineer at a Series A robotics startup building manipulation or locomotion products
**Background:** MS in ML/robotics, responsible for training pipeline and model deployment

**Daily Workflow:** Managing dataset versions, running training jobs, tracking policy performance, debugging failures by reviewing training data.

**Pain Points:**
- No single system connects data collection to training output
- Dataset versioning is manual
- Can't trace a failed policy back to specific training episodes

**Alloy Overlap Risk:**
Jake's startup may already use Alloy for post-deployment monitoring. If Alloy adds training data export features, Jake might resist adopting a second platform. Torque must deliver enough training-specific value (quality scoring, composition engine, pipeline integration) that Jake sees it as essential tooling Alloy cannot replace — analogous to how teams use both Datadog AND Weights & Biases.

**User Stories:**
- As an ML engineer, I want to query all successful grasp episodes from our ALOHA-2 arm with force-torque data so I can build a high-quality training dataset in minutes instead of hours.
- As an ML engineer, I want to stream episodes directly to my PyTorch DataLoader from cloud storage so I can start training immediately without downloading TBs of data. *(R2 — requires cloud infrastructure; local DataLoader only in R1)*
- As an ML engineer, I want to trace any trained policy back to the exact dataset version and episodes that produced it for debugging and reproducibility.

---

### Tertiary — Dr. Sarah Kim, Sim-to-Real Research Lead

**Role:** Research lead managing sim-to-real transfer pipelines
**Background:** Expert in domain randomization, simulation-to-reality gap bridging

**Daily Workflow:** Generating synthetic demonstrations in Isaac Sim, mixing with real teleoperation data, evaluating transfer quality.

**Pain Points:**
- No tool manages the composition of synthetic and real data with quality-aware mixing ratios

**Alloy Gap:** Alloy has zero presence in this space. Sim-to-real composition is structurally impossible from an operational monitoring platform. This persona represents Torque's deepest competitive moat.

**User Stories:**
- As a sim-to-real lead, I want to compose training datasets with configurable sim-to-real ratios and domain gap thresholds so I can systematically optimize transfer performance.
- As a lab PI, I want automated quality reports showing data distribution, success rates, and anomalies across all our robot platforms so I can make informed decisions about what data to collect next.

---

## 2.1 User Journeys

> User journeys document the realistic step-by-step flow a persona takes through the product. Each step is a potential failure point, a design decision, and an acceptance test. Journeys reveal where onboarding breaks before users do.

### UJ-01 — Dr. Maya Chen: First Use (MCAP → Training Dataset)

**Entry Point:** Maya has 3 hours of ALOHA-2 teleoperation recordings as MCAP files. She wants to filter out bad demonstrations and build a clean dataset for ACT policy training. She has heard about Torq from a lab colleague.

**Success State:** Maya's PyTorch DataLoader is iterating over a quality-filtered dataset and her training loop is running. She has seen quality scores and trusts them.

| Step | Action | Expectation | Failure Risk | Success Signal | Design Requirement |
|---|---|---|---|---|---|
| 1 | `pip install torq-robotics` | Installs cleanly. No conflicts with ROS 2 environment. | Dependency conflict with existing numpy/opencv in conda. Most common first-failure point. | No errors. `torq` version printed. | Core deps must be compatible with ROS 2 Humble/Iron. Conflicts must surface a helpful error message naming the conflicting package. |
| 2 | `import torq as tq` | Imports in under 2 seconds. | Slow import due to eager loading of heavy dependencies. | Import completes. `tq.__version__` prints correctly. | No torch, jax, or opencv imports at module level. Lazy loading only. Must complete in under 2 seconds. |
| 3 | `episodes = tq.ingest('./recordings/', format='auto')` | Progress bar. Auto-detects MCAP. Reports episode count. Completes. | Silent failure on malformed MCAP messages. Wrong episode boundary detection. | `"Ingested 47 episodes from 3 MCAP files (00:03:12 total duration)"` | Progress bar for multi-file directories. Episode count and duration reported. Failed files log a warning and continue — never silently drop data. |
| 4 | `episodes[0]` — Maya inspects first episode | Readable summary without custom inspection code. | Episode repr shows memory address instead of useful summary. | `Episode(id='ep_001', duration=4.2s, timesteps=210, observations=['joint_pos', 'wrist_cam'], actions=['joint_vel'])` | `Episode.__repr__` must include duration, timestep count, and modality list. Readable without method calls. |
| 5 | `scored = tq.quality.score(episodes)` | Progress bar. Under 60 seconds for 47 episodes. Scores attached. | No progress feedback. Scores all identical (bug). NaN on short episodes. | `"Scoring episodes: 47/47 [00:38]"`. `scored[0].quality.overall = 0.74` | Progress bar required. Episodes under 10 timesteps return warning and score of `None`, not NaN. Scores must vary across a realistic episode set. |
| **6** | **Maya looks at score distribution** | **Score ordering matches her intuition about which episodes are good.** | **Scores feel random. Bad episodes score high. She loses trust entirely. MAKE-OR-BREAK.** | Bottom 20% episodes are her intuitively-bad ones. GW-01 fires. | SC-1.2 must be validated here. GW-01 fires after scoring completes. |
| 7 | `dataset = tq.compose(scored, task='pick', quality_min=0.75)` | Returns filtered dataset. Reports how many episodes passed threshold. | `quality_min=0.75` filters out everything. Silent empty dataset returned. | `"Composed dataset: 31 episodes (16 filtered below quality_min=0.75)"`. GW-02 fires. | Warn loudly if fewer than 5 episodes returned. Always report how many were filtered and why. |
| 8 | `loader = tq.DataLoader(dataset, batch_size=32)` + training loop | Drop-in for PyTorch DataLoader. First batch within 5 seconds. | Collation error due to variable-length episodes. Shape mismatch. | First batch loads. `obs shape: [32, 210, 14]`. Training loop runs. | Default `collate_fn` handles variable-length episodes. Error messages on shape mismatch must name the offending episode and modality. |

**Journey Insights:**
- **Highest Risk Step:** 6 — If quality scores don't match Maya's intuition, she abandons the product. Every other step is recoverable. Step 6 is not.
- **Onboarding Target:** Steps 1–8 completed in under 20 minutes with real MCAP files on hand.
- **Gravity Wells Triggered:** GW-01, GW-02
- **Related Requirements:** Step 1–2 → PD-01, NFR-P05, NFR-C01 | Step 3 → DI-01, DI-02, DI-03, NFR-P01, NFR-R04 | Step 4 → DI-04 | Step 5 → QM-01, QM-02 | Step 6 → QM-01, SC-1.2 | Step 7 → DC-01, QM-03, GW-SDK-02, GW-SDK-05 | Step 8 → ML-01, NFR-P03

---

### UJ-02 — Jake Torres: Recurring Use (Integrating into Existing ML Pipeline)

**Entry Point:** Jake's startup collects 20–30 ALOHA-2 demonstrations per day. He currently runs a 340-line custom Python script. He wants to replace it with something maintainable.

**Success State:** `tq.ingest()` and `tq.DataLoader()` are integrated into Jake's CI training pipeline. New demonstrations are automatically ingested and quality-gated each night. Jake trusts the quality gate enough to stop manually reviewing every episode.

| Step | Action | Success Signal | Failure Risk | Design Requirement |
|---|---|---|---|---|
| 1 | Jake reads README. Runs the 5-line example. | Example runs without modification on his data. | — | README example must work copy-paste with any MCAP file. No config required. |
| **2** | **Jake replaces his 340-line script with `tq.ingest()` + `tq.DataLoader()`** | **Training output is identical. Script reduced to 8 lines.** | **Subtle data differences between Torq's episode representation and Jake's custom loader cause training divergence.** | **Numerical determinism: same input files must produce identical tensors as a correctly-implemented custom loader.** |
| 3 | `tq ingest ./daily_recordings/` (nightly cron) | CLI runs headlessly. Exit code 0 on success, non-zero on error. | — | CLI must be scriptable: `--json` flag, non-zero exit on any ingestion failure. Suitable for cron and CI. |
| 4 | Jake queries across 2 weeks of episodes | `tq.query(task='pick', date_after='2026-02-01', quality_min=0.8)` returns in under 1 second for 5,000 episodes. | Query scans all parquet files sequentially. Latency unacceptable at volume. | JSON index must accelerate task, date, quality, embodiment queries. Full scan only as fallback. |
| 5 | Jake hits 50GB of local data. Storage becomes a problem. | GW-03 fires: `"Stream directly without downloading → https://www.datatorq.ai"` | — | GW-03 fires when dataset size > 50GB or DataLoader memory usage exceeds available RAM. |

**Journey Insights:**
- **Highest Risk Step:** 2 — Jake will compare training results against his existing script. Any numerical difference makes him distrust Torq and revert. Determinism is non-negotiable.
- **Gravity Wells Triggered:** GW-03, GW-05
- **Related Requirements:** Step 1 → PD-01 | Step 2 → ML-01, DI-04, NFR-P03 | Step 3 → NFR-U03 | Step 4 → QE-02, NFR-P02 | Step 5 → GW-SDK-03

---

### UJ-03 — Dr. Sarah Kim: Sim-to-Real Dataset Composition *(R2 — Stub)*

**Entry Point:** Sarah has 500 Isaac Sim episodes and 200 real ALOHA-2 episodes. She wants to compose a training dataset with a controlled sim-to-real ratio and evaluate domain gap before committing to a full training run.

**Success State:** Sarah has composed a dataset with `sim_ratio=0.3`, reviewed the domain gap report, and launched a training run. She trusts the mixing ratio is reflected accurately in the DataLoader output.

> **Note:** This journey covers DC-04 (sim-to-real mixing) which is P1 and scheduled for R2. Steps are documented here for completeness and to ensure Sarah Kim's requirements are captured in architecture. Full journey detail to be expanded in R2 planning.

| Step | Action | Success Signal | Failure Risk | Related Requirements |
|---|---|---|---|---|
| 1 | `tq.ingest('./isaac_sim/', format='auto')` + `tq.ingest('./real_demos/')` | Sim and real episodes ingested separately. Provenance tagged automatically. | Sim episodes misidentified as real — no provenance distinction. | DI-02, DI-04 |
| 2 | `tq.compose(sim_ratio=0.3, domain_gap_max=0.5)` | Dataset returned with 30% sim, 70% real. Domain gap report generated. | `sim_ratio` param silently ignored. No domain gap visibility. | DC-04, DC-05 |
| 3 | Review domain gap report | Sarah can see per-episode domain gap scores and understand which sim episodes are closest to real. | Report exists but scores feel arbitrary — Sarah cannot interpret them. | DC-05, QM-07 |
| 4 | Launch training and verify provenance | Training loop reflects exact sim/real split. Policy lineage traces back to dataset version. | Provenance lost — cannot verify actual mixing ratio used in training. | DC-03, ML-04 |

**Journey Insights:**
- **Highest Risk Step:** 3 — Domain gap scores that don't match Sarah's intuition undermine the entire sim-to-real workflow.
- **Gravity Wells Triggered:** GW-02
- **Related Requirements:** DC-04, DC-05, DI-02, DI-04, QM-07, DC-03, ML-04

---

## 3. Competitive Defense Architecture

### Tier 1 — Overlap Zone: Alloy Can Match Within 6 Months

**Strategic Principle:** Don't fight Alloy on their home turf. Accept feature parity at the ingestion layer but immediately diverge into training-specific processing. The MCAP file enters the same door but exits as a fundamentally different data product.

| Capability | Alloy State | Torq Strategy |
|---|---|---|
| MCAP/ROS Ingestion | Production-ready. Direct upload, auto-parsing of sensor topics. | Match but don't compete. Torq ingests MCAP but enriches with episode segmentation and quality metadata. Same front door, different processing. |
| NL Data Search | Core differentiator. 'Search all your robot data in plain English.' Similarity search across images and time-series. | Match with training-specific semantics. Torq search understands 'find all successful grasp demonstrations with quality > 0.8' not just 'find sensor spikes.' |
| Multi-Modal Viz | Unified image, time-series, log viewer with frame-by-frame analysis. | Don't compete on general viz. Defer to Foxglove/Rerun. Torq visualizes episode quality, action distributions, and training dataset composition. |

### Tier 2 — Differentiation Zone: Alloy Would Need 12–18 Months

| Capability | Why Alloy Can't Match | Torq Implementation |
|---|---|---|
| Episode-First Data Model | Alloy's ontology is missions/scenarios. Retrofitting episode segmentation requires rewriting their data model. | P0. Every data object in Torq is an episode. Episodes contain aligned action-observation pairs with per-timestep quality scores. |
| Quality Scoring Engine | Alloy detects anomalies (sensor spikes, GPS loss) — fundamentally different from scoring demonstration quality. They'd need new ML models and domain expertise. | P0. Automated quality scoring across dimensions: trajectory smoothness, action consistency, task completion, kinematic feasibility. |
| Format Unification Layer | Alloy handles MCAP/ROS. The robotics training ecosystem uses 6+ incompatible formats. Alloy has no incentive to support training-specific formats. | P0. Universal ingest from all 6 major formats. Canonical episode representation internally. Format-agnostic integration moat. |

### Tier 3 — Deep Moat: Alloy Cannot Reach Without Becoming Torq

| Capability | Why Structurally Unreachable | Torq Implementation |
|---|---|---|
| Dataset Composition Engine | Requires training data distribution theory: stratified sampling, quality-weighted selection, task coverage analysis. Alloy's 'export scenario' is file export, not composition. | P0. `tq.compose(tasks=['pick','place'], quality_min=0.8, balance='stratified', sim_ratio=0.3)`. Distribution heatmaps. |
| Sim-to-Real Mixing | Alloy has zero simulation integration. Adding sim data support would require an entirely new product surface. | P1. Ingest from Isaac Sim, MuJoCo, Robosuite. Sim vs. real provenance. Automated mixing ratio optimization. |
| Training Pipeline Integration | Alloy stops at 'here's your data.' They have no concept of PyTorch DataLoaders, JAX pipelines, policy checkpoints, or experiment tracking. | P0. Native `tq.DataLoader`, JAX `tf.data` pipeline, W&B integration. Policy training tracking with dataset lineage. |
| Cross-Embodiment Normalization | Alloy handles fleets of identical robots. They have no concept of cross-embodiment normalization for unified training. | P1. Task × Embodiment heatmap. Automatic action space normalization. Cross-embodiment datasets for foundation model training. |
| Policy Experiment Tracking | Would require Alloy to build an entirely new experiment tracking system — W&B territory, not operational monitoring. | P1. Full experiment lifecycle: dataset version → training config → policy checkpoint → evaluation metrics → deployment approval. |

---

## 4. Functional Requirements

> Priority reflects competitive defensibility, not just user value. All P0 requirements must ship in v0.1 (Alpha). Acceptance criteria are testable conditions.

### 4.1 Data Ingestion & Episode Extraction (Tier 1–2)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| DI-01 | P0 | Ingest MCAP/ROS2 bag files with automatic topic discovery and schema extraction | Successfully ingests MCAP files from ROS 2, extracts all topics, creates episode objects within 5s for <1GB files. Auto-discovers image, joint state, and action topics. | 1 | Yes |
| DI-02 | P0 | Ingest HDF5 (robomimic), TFRecord/RLDS, Parquet+MP4 (LeRobot), Zarr, EBML formats | Imports any LeRobot v3.0 dataset. Handles standard robomimic HDF5 up to 50GB. Imports OXE datasets without TensorFlow dependency. Handles Zarr v2 arrays. | 2 | No |
| DI-03 | P0 | Automatic episode boundary detection from continuous teleoperation streams | Detects episode boundaries with >90% accuracy measured against 10 annotated continuous-stream fixtures with known ground truth boundaries (fixtures to be created during architecture phase). Supports gripper state changes, velocity thresholds, manual markers. Segments 1-hour continuous stream into episodes in <30s. | 2 | No |
| DI-04 | P0 | Generate canonical episode representation with aligned action-observation pairs, per-timestep metadata, and provenance tracking | Episode stores all modalities (images, joint states, actions, force-torque) with nanosecond timestamps and arbitrary metadata. Schema is extensible, validated on ingest, queryable. Source file provenance tracked. | 2 | No |
| DI-05 | P1 | SDK-based streaming ingestion: `tq.record(episode)` for real-time capture during teleoperation | Subscribes to configurable ROS 2 topics, creates episodes in real-time with <100ms latency. Supports pause/resume and episode tagging. Auto-triggers quality scoring on episode completion. | 2 | No |
| DI-06 | P1 | Bulk import with progress tracking and validation reporting | `tq ingest ./data/` auto-detects format, processes in parallel across CPU cores, reports progress bar and per-file error summary. Handles 1000+ files without memory overflow. | 1 | Yes |
| DI-07 | P2 | Automatic format detection and conversion on upload | Correctly identifies file format with >95% accuracy across all 6 supported formats. Falls back to user prompt on ambiguous files. | 2 | No |
| DI-08 | P2 | Integration with teleoperation hardware APIs (ALOHA, Gello, UMI) for direct capture | Provides hardware-specific capture adapters. ALOHA integration captures bimanual joint states + wrist cameras at 50Hz. Configurable observation/action mapping per hardware. | 3 | No |

### 4.2 Episode Quality Management (Tier 2–3 — Core Moat)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| QM-01 | P0 | Automated quality scoring: trajectory smoothness (jerk analysis), action consistency (distribution analysis), task completion detection | Assigns 0.0–1.0 quality score per episode across 4+ dimensions. Scores correlate with human quality judgments (>0.7 Spearman's ρ) against a ground truth set of 50 hand-labeled episodes covering smooth, hesitant, jerky, and incomplete demonstrations — labels to be created during architecture phase (see `fixtures.quality_ground_truth` in CLAUDE.md). Processes 100 episodes in <60s. | 2 | No |
| QM-02 | P0 | Per-episode quality report with dimension breakdown and overall score (0.0–1.0) | Report includes per-dimension scores (smoothness, consistency, completion, feasibility), overall weighted score, timestep-level quality heatmap, and flagged anomaly regions. | 2 | No |
| QM-03 | P0 | Quality gates: configurable thresholds that prevent low-quality episodes from entering curated datasets | Define per-dataset quality thresholds. Episodes below threshold auto-rejected with reason codes. Gate configuration stored with dataset version metadata. Override requires explicit flag. | 3 | No |
| QM-04 | P0 | Quality distribution visualization across dataset with outlier detection | Interactive histogram/violin plots. Automatic outlier detection (>2σ or IQR-based). Click-to-inspect any episode from distribution plot. | 3 | No |
| QM-05 | P1 | Comparative quality analysis: before/after quality trends as operators improve | Time-series plots showing quality trends per operator over collection sessions. Statistical significance test for quality improvement. | 3 | No |
| QM-06 | P1 | Kinematic feasibility scoring: detect physically impossible or unsafe demonstrations | Validates joint limits, velocity limits, and collision constraints against robot URDF. Flags episodes with >5% timesteps exceeding physical limits. Supports ALOHA-2, Franka, UR5 URDFs. | 3 | No |
| QM-07 | P2 | Custom quality metric plugins: user-defined scoring functions via SDK | `tq.quality.register(name, fn)` accepts any Python callable returning 0.0–1.0. Custom metrics computed alongside built-in metrics. Plugin discovery via entry points. | 3 | No |
| QM-08 | P2 | Quality-based operator leaderboards for teleoperation team management | Ranked operator list by average quality score, episodes collected, and improvement rate. Anonymization option. | 3 | No |

### 4.3 Dataset Composition Engine (Tier 3 — Deepest Moat)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| DC-01 | P0 | SQL-like query builder for training dataset construction: filter by task, quality, embodiment, environment, provenance | `tq.compose(tasks=['pick','place'], quality_min=0.8, embodiment='aloha2')` returns in <5s for 100K episode pool. Supports AND/OR/NOT logic, range filters, regex. Query stored as reproducible recipe. | 3 | No |
| DC-02 | P0 | Stratified sampling with configurable balance: equal task distribution, quality-weighted selection, embodiment coverage | `balance='stratified'` produces equal episode counts per task (±5%). `balance='quality_weighted'` over-samples high-quality episodes. Deterministic given same seed. | 3 | No |
| DC-03 | P0 | Dataset versioning with full provenance: every dataset records exact episodes, filters, and composition rules used | Create named versions. View diffs between versions. Restore any version. Full provenance: exact query, filters, sampling config, source episodes stored immutably. | 3 | No |
| DC-04 | P1 | Sim-to-real mixing: `tq.compose(sim_ratio=0.3, domain_gap_max=0.5)` | Accepts `sim_ratio` (0.0–1.0). `domain_gap_max` filters sim episodes. Provenance tags distinguish sim vs. real. Supports Isaac Sim, MuJoCo, Robosuite. | 3 | No |
| DC-05 | P1 | Dataset composition visualization: distribution heatmaps, coverage gaps, diminishing returns curves | Task × Embodiment heatmap. Coverage gap analysis. Diminishing returns curve. Interactive, filterable. | 3 | No |
| DC-06 | P1 | Incremental dataset updates: add new episodes without full recomposition | `tq.dataset.append(new_episodes, validate=True)` adds episodes matching existing quality gates. Creates new version automatically. Validates schema compatibility. | 3 | No |
| DC-07 | P2 | Auto-composition: ML-driven suggestions for optimal dataset composition | Given target task and current policy performance, suggests episodes to add/remove. Validated: auto-composed datasets match or exceed human-curated on 3+ benchmark tasks. | 3 | No |
| DC-08 | P2 | A/B dataset comparison: side-by-side composition analysis | `tq.compare('dataset_a', 'dataset_b')` shows side-by-side distributions, quality stats, episode overlap, and unique contributions. | 3 | No |

### 4.4 ML Pipeline Integration (Tier 3 — Deepest Moat)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| ML-01 | P0 | Native PyTorch DataLoader: `tq.DataLoader(dataset='pick_v3', batch_size=32, shuffle=True)` | Drop-in replacement for standard PyTorch DataLoader. Multi-worker loading, prefetching, `pin_memory`. Compatible with DistributedDataParallel. First batch within 5s local, 30s cloud. | 3 | No |
| ML-02 | P1 | JAX/Flax `tf.data` pipeline integration | JAX-compatible iterator with automatic batching, prefetching, and device placement. Supports `jax.pmap` for multi-GPU. Zero-copy where possible via Arrow memory mapping. R2 delivery — see Release Plan. *(Corrected from P0: Section 7 Release Plan places JAX in R2. P0 = must ship in R1 Alpha; JAX does not.)* | 3 | No |
| ML-03 | P0 | W&B, MLflow, and TensorBoard integration for experiment tracking with dataset lineage | Auto-logs dataset ID, version, composition query, quality statistics, episode count at training start. One-line setup: `tq.integrations.wandb.init()`. | 3 | No |
| ML-04 | P1 | Policy checkpoint registry with dataset provenance: trace any model to exact training episodes | `tq.checkpoint.save(policy, dataset='pick_v3')`. `tq.trace('policy_v3')` returns complete lineage. Searchable registry. | 3 | No |
| ML-05 | P1 | Training job orchestration: launch training runs directly from composed datasets | `tq.train(dataset='pick_v3', config='act_policy.yaml')`. Supports local and cloud (SLURM, K8s) backends. Auto-logs to experiment tracker. | 3 | No |
| ML-06 | P2 | Automated evaluation pipeline: run trained policies against held-out episodes | `tq.evaluate(policy, holdout='pick_eval_v1')` reports success rate, trajectory similarity, and quality delta. | 3 | No |
| ML-07 | P2 | Cross-embodiment action space normalization for foundation model training | Automatic normalization of heterogeneous action spaces. Validated: normalized cross-embodiment datasets train policies that transfer across 3+ robot types. | 3 | No |
| ML-08 | P3 | Distributed training data serving: optimized loading for multi-GPU/multi-node training | Distributed data serving with locality-aware shard assignment. Supports 8+ GPU nodes. Validated on A100 and H100 clusters. | 3 | No |

### 4.5 Query Engine & Data Discovery (Tier 1–2)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| QE-01 | P0 | Natural language search with training-specific semantics: 'successful grasps with quality > 0.8 on ALOHA-2' | Converts natural language to structured queries with >85% accuracy against a benchmark of 50 annotated query-result pairs (benchmark to be defined during architecture phase covering task, quality, embodiment, date, and success filter patterns). Response time <2s. Supports follow-up refinement. | 1+ | Partial |
| QE-02 | P0 | Structured query API: `tq.query(task='pick', quality_min=0.8, embodiment='aloha2')` | Returns matching episodes in <1s for 100K episodes. Supports chaining. Results are lazy-evaluated iterators. Index-accelerated on task, embodiment, quality, date fields. | 2 | No |
| QE-03 | P1 | Task × Embodiment × Quality heatmap for dataset coverage analysis | Interactive 3D heatmap. Cell click expands to episode list. Highlights underrepresented cells. Updates in real-time as episodes are added. | 3 | No |
| QE-04 | P1 | Similar episode retrieval: 'find 50 episodes most similar to this successful grasp' | Returns top-K similar episodes based on visual or state embeddings with <500ms latency for 100K episodes. | 1+ | Partial |
| QE-05 | P2 | Dataset gap analysis: 'what tasks/conditions are underrepresented in my training data?' | Reports top 5 gaps with specific collection recommendations. Integrates with composition engine to auto-fill gaps from available pool. | 3 | No |

### 4.6 Collaboration & Cloud (Tier 1–2)

| ID | Priority | Requirement | Acceptance Criteria | Tier | Alloy Overlap |
|---|---|---|---|---|---|
| CC-01 | P1 | Team workspace with role-based access control (admin, researcher, annotator) | Create workspaces, invite members via email, set per-dataset read/write/admin permissions. 3 default roles with configurable custom roles. Activity log per workspace. | 1 | Yes |
| CC-02 | P1 | Dataset sharing: publish curated datasets with documentation and quality reports | `tq.publish(dataset, readme='...', license='CC-BY-4.0')` creates shareable dataset page with auto-generated quality report. DOI assignment for academic citation. | 2 | No |
| CC-03 | P2 | Episode annotation and review workflows (approve/reject/flag for re-collection) | Assign episodes to reviewers. Batch annotation support. Review status tracked in episode metadata. | 2 | No |
| CC-04 | P2 | Cross-institution dataset federation for collaborative research | Federated queries across institutions without centralizing data. Each institution retains data ownership. Supports HIPAA-compatible isolation. | 3 | No |
| CC-05 | P3 | Marketplace: share and discover community-contributed datasets | Public dataset directory with search, preview, and one-click import. Quality badges. Revenue share model (15% platform fee). | 3 | No |

### 4.7 Package & Distribution (Naming Spec)

| ID | Priority | Requirement | Acceptance Criteria |
|---|---|---|---|
| PD-01 | P0 | SDK installable as `pip install torq-robotics` with Python import `import torq as tq` | `pip install torq-robotics` succeeds on Python 3.9+. `import torq as tq` imports in under 2 seconds. `tq.__version__` returns the current version string. No user ever types `import torque`. |
| PD-02 | P0 | All SDK documentation, README examples, and acceptance criteria use `import torq as tq` | `grep -r 'import torque'` in the repository returns zero matches. All code examples use `import torq as tq`. PyPI page shows correct install command. |

### 4.8 Gravity Well SDK Requirements

> Cloud Platform: https://www.datatorq.ai

| ID | Priority | Requirement | Acceptance Criteria | Maps To |
|---|---|---|---|---|
| GW-SDK-01 | P0 | After `tq.quality.score()` completes, print a gravity well prompt directing users to datatorq.ai for community benchmark comparison. | Prompt fires on every successful `quality.score()` call. Text includes computed score and datatorq.ai URL. Fires after successful completion, never on error. Suppressible via `tq.config.quiet=True`. | GW-01 |
| GW-SDK-02 | P0 | After `tq.compose()` returns a non-empty dataset, print a gravity well prompt directing users to datatorq.ai for dataset sharing. | Prompt fires on every successful `compose()` call returning >0 episodes. Text includes episode count and datatorq.ai URL. Suppressible via `tq.config.quiet=True`. | GW-02 |
| GW-SDK-03 | P1 | When `tq.DataLoader()` is initialised on a dataset exceeding 50GB, print a gravity well prompt for cloud streaming. | Prompt fires when dataset size > 50GB at DataLoader init time. Suppressible via `tq.config.quiet=True`. | GW-03 |
| GW-SDK-04 | P0 | `tq.cloud()` prints a waitlist prompt and URL. Called explicitly or when any cloud-only keyword argument is passed. | `tq.cloud()` always prints the datatorq.ai URL and waitlist message. Cloud-only kwargs on local functions trigger the same message and do not raise an unhandled exception. | GW-04 |
| GW-SDK-05 | P1 | When `tq.query()` or `tq.compose()` returns fewer than 5 episodes, print a gravity well prompt for community dataset discovery. | Prompt fires when result set < 5 episodes. Text includes the task and embodiment queried and datatorq.ai URL. Suppressible via `tq.config.quiet=True`. | GW-05 |
| GW-SDK-06 | P0 | All gravity well prompts use a single internal `_gravity_well()` helper with consistent formatting. | All 5 gravity wells produce output via the same helper. Consistent format: `💡 {message}\n   → {url}\n`. Suppressible via `tq.config.quiet=True`. No network calls in R1. | GW-01–GW-05 |
| GW-SDK-07 | P1 | Opt-in telemetry: captures gravity well trigger event, SDK version, Python version, and OS platform only. Deferred to R2. | *(R2 — deferred from R1 due to zero-network-calls constraint in Section 0.2. Telemetry requires an outbound HTTP call which violates R1 scope.)* First gravity well prompt in R2 includes one-time consent request. Default: no telemetry. Preference stored in `~/.torq/config.toml`. Never captures file paths, scores, episode content, or IP address. | GW-01–GW-05 |

---

## 4.9 Non-Functional Requirements

> Each NFR follows BMAD format: "The system shall [metric] [condition] [measurement method]." All NFRs are testable with specific criteria.

### Performance

| ID | Requirement | Scope |
|---|---|---|
| NFR-P01 | The SDK shall ingest a 1GB MCAP file in under 10 seconds as measured by wall-clock time on a standard developer machine (16GB RAM, 4-core CPU, SSD storage). | SDK — R1 |
| NFR-P02 | The system shall return metadata query results in under 1 second for datasets containing 100,000 or more episodes, as measured from query call to result returned. **Architecture dependency:** JSON index must support indexed lookups on task, quality, embodiment, and date fields without full-scan. Index structure must be designed explicitly during architecture phase — a flat JSON file cannot meet this target at 100K scale without indexing. | SDK — R1 |
| NFR-P03 | The PyTorch DataLoader shall sustain throughput of 1,000 or more episodes per second during training iteration, as measured by batch iteration wall-clock timing on a GPU-equipped machine. | SDK — R1 |
| NFR-P04 | The cloud DataLoader shall deliver the first training batch within 30 seconds of DataLoader initialisation for remotely-stored datasets, as measured from `tq.DataLoader()` call to first batch available. | Cloud — R2 |
| NFR-P05 | The SDK shall complete `import torq` in under 2 seconds on a standard developer laptop, as measured by wall-clock time from Python import statement to module ready. No heavy dependencies loaded at import time. | SDK — R1 |

### Scalability

| ID | Requirement | Scope |
|---|---|---|
| NFR-S01 | The system shall support datasets up to 10TB containing 1 million or more episodes without degradation in query performance below the NFR-P02 threshold, as validated by load testing. | Cloud — R3 |
| NFR-S02 | The cloud platform shall support 100 or more concurrent users per workspace without API response latency exceeding 500ms at the 95th percentile, as measured by APM monitoring. | Cloud — R3 |
| NFR-S03 | The cloud storage backend shall scale transparently to petabyte-scale data via S3/GCS without requiring application-level changes, as validated by architecture review and provider SLA documentation. | Cloud — R4 |

### Reliability

| ID | Requirement | Scope |
|---|---|---|
| NFR-R01 | The cloud platform shall maintain 99.9% uptime during scheduled business hours as measured by cloud provider SLA monitoring and independent uptime tracking. | Cloud — R2 |
| NFR-R02 | The system shall provide 99.999999999% (11 nines) data durability for stored episodes via S3/GCS multi-region replication, as guaranteed by cloud provider SLA. | Cloud — R2 |
| NFR-R03 | The system shall encrypt all stored episode data at rest using AES-256 and all data in transit using TLS 1.3 minimum, validated by security audit before cloud launch. | Cloud — R2 |
| NFR-R04 | The SDK shall produce no silent data corruption. Any file that fails to parse must log a warning with the file path and error reason and continue processing remaining files, never dropping data silently. Validated by adversarial test fixtures (corrupt MCAP, truncated HDF5). | SDK — R1 |
| NFR-R05 | The cloud platform shall achieve SOC 2 Type II certification within 18 months of commercial launch (target: M18), as evidenced by completed audit report. | Cloud — R4 |

### Usability

| ID | Requirement | Scope |
|---|---|---|
| NFR-U01 | A new user shall complete the full 5-line workflow (ingest → score → compose → DataLoader → training loop) in under 20 minutes from `pip install`, with real data on hand, as measured by timed observation during user validation sessions (UJ-01). | SDK — R1 |
| NFR-U02 | Every SDK exception shall include a human-readable message stating what went wrong and what the user should do to resolve it. No bare exceptions. Validated by code review and edge case test suite. | SDK — R1 |
| NFR-U03 | All CLI commands shall support a `--json` flag for machine-readable output and return non-zero exit codes on any failure, enabling headless use in CI/CD pipelines. Validated by CLI integration tests. | SDK — R1 |

### Compatibility

| ID | Requirement | Scope |
|---|---|---|
| NFR-C01 | The SDK shall support Python 3.9 and above on Linux (primary), macOS (development), and Windows via WSL2 (limited support), as validated by CI matrix across all three platforms. | SDK — R1 |
| NFR-C02 | The SDK shall support PyTorch 2.0 and above including DistributedDataParallel across multiple GPUs and nodes, as validated by integration tests on multi-GPU test hardware. | SDK — R1 |
| NFR-C03 | The SDK shall support JAX 0.4 and above with `jax.pmap` for multi-device training, as validated by JAX-specific integration tests. | SDK — R2 |
| NFR-C04 | The SDK shall support ROS 2 distributions Humble (LTS), Iron, and Jazzy for MCAP ingestion, as validated by ingestion tests against each distribution's message schema. | SDK — R1 |
| NFR-C05 | The SDK core (`import torq as tq`) shall function without torch, jax, or any ML framework installed. Framework imports occur only inside the `serve/` layer and only when explicitly requested. Validated by install test with core dependencies only. | SDK — R1 |

---

## 5. System Architecture

**Design Principle:** Architecture designed with competitive defensibility as a first-class concern. Layers 1–2 overlap with Alloy. Layers 3–8 are Torque-specific and represent the defensible product surface.

**Value Principle:** Value accretes upward. Layers 1–2 are table stakes. Layers 5–8 are where Torque creates irreplaceable value. The test is: 'Does this feature require understanding of robot learning, not just robot data?'

### 8-Layer Architecture Stack

| Layer | Name | Function | Alloy Overlap | Defensibility |
|---|---|---|---|---|
| 8 | Policy Pipeline | Training orchestration, checkpoint registry, evaluation | NONE | DEEP MOAT |
| 7 | ML Integration | DataLoaders, W&B/MLflow integration, experiment tracking | NONE | DEEP MOAT |
| 6 | Composition Engine | Dataset construction, sim-to-real mixing, stratified sampling | NONE | DEEP MOAT |
| 5 | Quality Engine | Episode scoring, quality gates, distribution analysis | NONE | STRONG |
| 4 | Query Engine | NL search, structured queries, coverage heatmaps | PARTIAL | MODERATE |
| 3 | Episode Store | Canonical episode format, versioning, provenance | LOW | STRONG |
| 2 | Format Bridge | MCAP, HDF5, RLDS, LeRobot, Zarr, EBML conversion | MCAP only | MODERATE |
| 1 | Ingestion | File upload, streaming capture, episode boundary detection | HIGH | LOW |

> Non-Functional Requirements: See Section 4.9 above.

### SDK API Reference

```python
import torq as tq          # pip install torq-robotics

# ── Connect ──
client = tq.connect("https://app.datatorq.ai", api_key="tq_...")

# ── Ingest (Tier 1-2: Same front door, different processing) ──
episodes = tq.ingest("demo_session.mcap",
  auto_segment=True,       # Episode boundary detection
  quality_score=True,      # Inline quality scoring
  format='auto'            # MCAP, HDF5, RLDS, LeRobot, Zarr
)

# ── Quality (Tier 2-3: Core moat) ──
report = tq.quality.score(episodes[0])
print(report.smoothness)          # 0.87
print(report.task_completion)     # True
print(report.kinematic_feasible)  # True
print(report.overall)             # 0.82

# ── Compose (Tier 3: Deepest moat) ──
dataset = tq.compose(
  tasks=['pick', 'place', 'pour'],
  quality_min=0.75,
  balance='stratified',
  sim_ratio=0.3,
  embodiments=['aloha2', 'franka'],
  version='pick_place_v3'
)

# ── Train (Tier 3: Full pipeline) ──
loader = tq.DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
  obs, actions = batch['observations'], batch['actions']
  loss = policy(obs, actions)
  tq.log(loss=loss.item())

# ── Trace (Tier 3: Full lineage) ──
tq.checkpoint.save(policy, dataset='pick_place_v3',
  metrics={'success_rate': 0.91})
lineage = tq.trace('policy_v3')
```

---

## 5.5 Gravity Well Conversion Model

> Every SDK feature has a moment where local execution succeeds but leaves an unanswered question that only aggregate, collaborative, or cloud-scale data can resolve. Torq SDK solves the immediate problem. datatorq.ai solves the next one. The SDK earns trust; the cloud converts it. Gravity wells fire after successful task completion — never interrupting a failed or in-progress operation.

**Cloud Platform:** https://www.datatorq.ai

### Classification Rules

| Rule | Definition | Verdict | Example |
|---|---|---|---|
| Airplane Test | Works fully offline, delivers complete value to one user | SDK — no gravity well | `tq.quality.score()` runs offline. Score returned locally. |
| Single-Player Test | Full value delivered to a single user with no collaboration | SDK — no gravity well | `tq.compose()` builds a local dataset. Single researcher, complete value. |
| Multi-Player Test | Value requires sharing, collaboration, or team coordination | Cloud — gravity well fires | Sharing a composed dataset with a lab collaborator requires cloud. |
| Impossible Locally Test | Requires aggregate data, community benchmarks, or cloud-scale compute | Cloud — gravity well fires | Comparing your quality score against community median is impossible locally. |

### Touchpoints

#### GW-01 — Quality Benchmark Comparison
| Field | Value |
|---|---|
| **Trigger Function** | `tq.quality.score()` |
| **Trigger Condition** | Scoring completes successfully on any episode or episode list. |
| **Local Ceiling** | User receives quality scores (0.0–1.0) but has no reference point. Is 0.71 good for a pick task? This question cannot be answered without aggregate community data. |
| **Cloud Resolution** | Community quality benchmarks aggregated across datatorq.ai users, broken down by task type and embodiment. Anonymised, opt-in contribution. |
| **Classification** | Impossible Locally |
| **Strength** | STRONGEST — unanswerable without aggregate data |

**Prompt:**
```
💡 Your episodes scored {score:.2f}. Want to see how this compares to the community?
   → https://www.datatorq.ai
```
*Note: No invented benchmark numbers in prompt. Thresholds calibrated during Phase 1 validation.*

---

#### GW-02 — Dataset Sharing and Publication
| Field | Value |
|---|---|
| **Trigger Function** | `tq.compose()` |
| **Trigger Condition** | Dataset composition completes and dataset contains >0 episodes. |
| **Local Ceiling** | User has a composed dataset locally but cannot share it with collaborators, publish it with a DOI, or version it with team-visible provenance. |
| **Cloud Resolution** | Dataset sharing, team workspaces, DOI assignment for academic citation, published quality reports (CC-01, CC-02). |
| **Classification** | Multi-Player |
| **Strength** | STRONG — academic users have a natural sharing need |

**Prompt:**
```
💡 Dataset '{dataset_name}' composed: {n} episodes, quality avg {score:.2f}.
   Share with your lab or publish with a citable DOI
   → https://www.datatorq.ai
```

---

#### GW-03 — Cloud Streaming for Large Datasets
| Field | Value |
|---|---|
| **Trigger Function** | `tq.DataLoader()` |
| **Trigger Condition** | Dataset size exceeds 50GB at DataLoader initialisation. |
| **Local Ceiling** | Local disk and RAM become the bottleneck. Downloading TBs of data before training is impractical. Multi-GPU training requires distributed data serving. |
| **Cloud Resolution** | Cloud-streaming DataLoader without local download. Locality-aware shard assignment for multi-node runs (ML-01, ML-08). |
| **Classification** | Impossible Locally |
| **Strength** | STRONG — hits at scale, exactly when users need it most |

**Prompt:**
```
💡 Dataset is {size:.1f}GB. Stream directly to your training loop without downloading
   → https://www.datatorq.ai
```

---

#### GW-04 — Explicit Cloud Access
| Field | Value |
|---|---|
| **Trigger Function** | `tq.cloud()` |
| **Trigger Condition** | Explicit user call, or any cloud-only keyword argument passed to a local function (e.g. `tq.compose(..., publish=True)`). |
| **Cloud Resolution** | Full cloud platform: dashboard, team workspaces, community benchmarks. |
| **Classification** | Explicit |
| **Strength** | BASELINE — always-on waitlist entry point |

**Prompt:**
```
💡 Torq Cloud is in early access.
   Join the waitlist → https://www.datatorq.ai
```

---

#### GW-05 — Community Dataset Discovery
| Field | Value |
|---|---|
| **Trigger Function** | `tq.query()` / `tq.compose()` |
| **Trigger Condition** | Result set returns fewer than 5 episodes. |
| **Local Ceiling** | User's local collection is insufficient for the task. 'Where do I get more pick demonstrations for ALOHA-2?' cannot be answered from local data alone. |
| **Cloud Resolution** | Community dataset pool and marketplace (CC-05, QE-05). |
| **Classification** | Impossible Locally |
| **Strength** | MODERATE — fires when users most feel the data scarcity problem |

**Prompt:**
```
💡 Only {n} episodes found for task='{task}', embodiment='{embodiment}'.
   Discover community episodes for this task
   → https://www.datatorq.ai
```

---

### Implementation Spec

```python
def _gravity_well(
    message: str,
    feature: str,
    url: str = "https://www.datatorq.ai"
) -> None:
    """Print a cloud escalation prompt after successful local task completion."""
    if not _config.quiet:
        print(f"\n💡 {message}")
        print(f"   → {url}\n")
    _telemetry_ping(feature=feature)  # opt-in only
```

**Telemetry (Opt-In):**

First gravity well prompt includes one-time consent request:
> "Help improve Torq by sharing anonymous usage data? [y/N]"

Preference stored in `~/.torq/config.toml`. Default: N (no telemetry).

| Captured | Not Captured |
|---|---|
| Gravity well ID (e.g. GW-01) | File paths, dataset names, episode content |
| Timestamp (UTC) | Quality scores or any user data |
| Torq SDK version | IP address |
| Python version | — |
| OS platform | — |

**Ordering Principle:** GW-01 (quality benchmarks) is the highest-priority well to activate at cloud launch. It fires for every user who runs quality scoring — the core SDK workflow — and poses a question that is genuinely impossible to answer locally. This is the primary conversion mechanism from free SDK to first paying cloud feature.

---

## 6. Alloy Intervention Scenarios & Mitigation

| ID | Scenario | Probability | Timeline | Impact | Mitigation |
|---|---|---|---|---|---|
| A | **Alloy Adds Training Data Export** | 90% | 3–6 months | Low — low-sophistication users may not adopt Torq | Accelerate quality scoring and composition features. The message: 'Alloy exports data. Torq composes training datasets.' Run benchmarks showing quality-scored, composed datasets train better policies with fewer demonstrations. |
| B | **Alloy Adds Episode Segmentation** | 40% | 12–18 months | Medium — Alloy can claim 'episode management' | By this point, Torq's episode model must be so deeply integrated with quality scoring, composition, and training that 'episode' means something fundamentally richer in Torq. |
| C | **Alloy Pivots to Training Infrastructure** | 15% | 18–24 months | High — direct competition with better funding ($3M vs. bootstrapped), existing customers, and VC backing. Existential scenario. | Speed. If Torq has 12–18 months of training-specific features and SDK adoption, Alloy starts from scratch in a space where Torq has established conventions. The W&B playbook proves developer tool switching costs are real once teams integrate at the SDK level. |
| D | **Airtree Funds Both** | 60% | — | Mixed — validates complementary positioning but creates board-level information flow risk (see R-03, updated to MEDIUM). If Airtree funds both, portfolio visibility into Torq's implementation roadmap is near-certain. | Frame the Alloy–Torq relationship as 'Datadog + Weights & Biases for robotics.' Prepare 1-page diagram: Torq (training) → deployed robot → Alloy (monitoring) → Torq (failure-to-training feedback loop). Product mitigation: present only vision in investor materials, never implementation detail. Request portfolio conflict clause or information barrier commitment in term sheet. |
| E | **Neuracore Accelerates** | 40% | 6–12 months | Medium-High — Neuracore is the most strategically similar competitor (see Section 8). If they close the 12–18 month execution gap through additional funding or a key hire, they enter Torq's primary market with academic credibility. | Speed is the primary mitigation. Torq's advantage is current execution lead — Neuracore is pre-seed with a tiny team. Establish SDK conventions (episode format, quality scoring API) as community standards before Neuracore can. Prioritise academic design partner relationships at CoRL/RSS/ICRA to make Torq the default tool before Neuracore has a product to show. |

---

## 7. Release Plan (Competitive-Accelerated)

> v3.0 change: Quality scoring and composition features moved from R2 to R1 (Alpha) because they are now the primary competitive differentiators.

| Release | Timeline | Deliverables | Competitive Purpose |
|---|---|---|---|
| **R1: Alpha** | Month 3 | SDK (`tq.ingest`, `tq.quality.score`, `tq.compose`), Episode store, Quality scoring engine, Basic composition, PyTorch DataLoader, CLI | Establish training-specific tooling before Alloy extends. Land 3–5 academic design partners. |
| **R2: Beta** | Month 6 | Cloud dashboard, NL search (training semantics), Sim-to-real mixing, Dataset versioning, W&B integration, Team workspaces | Make Torq sticky for teams. Integration depth creates switching costs. |
| **R3: GA** | Month 9 | Cross-embodiment normalization, Policy experiment tracking, Dataset marketplace (beta), Enterprise SSO/RBAC | Expand to foundation model training teams. Begin enterprise sales. |
| **R4: Scale** | Month 12 | Distributed data serving, Auto-composition ML, Alloy integration (production-to-training feedback loop), Custom quality plugins | Platform maturity. Consider strategic Alloy partnership. |

---

## 8. Competitive Threat Matrix

> v3.0 update: Alloy elevated from LOW-MED to HIGH. Neuracore elevated to MEDIUM-HIGH.

| Competitor | Threat | Key Strength | Critical Gap |
|---|---|---|---|
| **Alloy** | HIGH | $3M funded, 4 design partners, MCAP ingestion, NL search, 'Scenarios to Training Data', Blackbird+Airtree, same geography | No episode model, no quality scoring, no composition, no training pipeline, no sim. Mission/scenario ontology structurally different. |
| **Foxglove** | HIGH | MCAP standard creator, cloud platform, enterprise customers, strong brand | No ML pipeline, no episode management, no training integration. Visualization tool, not data platform. |
| **LeRobot/HF** | MED-HIGH | Community standard, open format, policy library, HW support | No cloud infra, no quality tools, no team features, no composition engine. |
| **Neuracore** | MED-HIGH | Most similar vision, academic credibility, recent seed round | Pre-seed ($3M), tiny team, 12–18 months behind in execution. See Section 6, Scenario E for mitigation plan. Alloy threat timelines (Section 6, Scenarios A–D) apply independently. |
| **Scale AI** | MEDIUM | Resources ($29B), Physical AI Data Engine announced | Managed service model, not developer tooling. Extremely expensive. |
| **NVIDIA Isaac** | MEDIUM | Full sim stack, OSMO orchestration, ecosystem gravity | Explicitly lacks dashboards, artifact registry, experiment tracking. Sim-only. |
| **Rerun** | MEDIUM | 9.3K GitHub stars, elegant data model, dataframe API | No cloud platform yet, no ML features, no training integration. |
| **W&B** | LOW | Developer experience gold standard, massive adoption | CoreWeave-owned, focused on LLMs. No robotics-specific features. |

---

## 9. Risk Matrix

| ID | Tag | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|---|
| R-01 | NEW | Alloy extends into pre-deployment training data (scenario export, basic DataLoader) | HIGH | MED | Ship quality scoring + composition before Alloy can. Benchmark: composed datasets train 30% better. |
| R-02 | NEW | Alloy pivots core product to compete directly with Torq | LOW | CRIT | Speed. 12–18 months of features + SDK adoption creates switching costs. |
| R-03 | NEW | Airtree shares Torq's roadmap with Alloy (portfolio conflict) | MEDIUM | HIGH | Share vision, not implementation details. Use 'complementary portfolio' frame. Request information barrier clause in term sheet. *(Probability updated LOW→MEDIUM: Scenario D puts Airtree funding both at 60% — if they fund both, board-level information flow is near-certain.)* |
| R-04 | — | Format fragmentation accelerates | HIGH | MED | Extensible format plugin system. Canonical internal representation insulates from format churn. |
| R-05 | — | Academic users don't pay (free-tier dependency) | HIGH | MED | Freemium SDK with cloud limits. Enterprise upsell. Community creates evangelists. |
| R-06 | — | NVIDIA enters with integrated solution | MED | HIGH | Hardware-agnostic positioning. NVIDIA lock-in is a weakness for customers. |
| R-07 | — | Solo founder execution risk | HIGH | CRIT | Prioritize ruthlessly. SDK + quality + composition in R1. Hire first engineer by Month 2 (see SC-1.5 — hiring is now a measurable Phase 1 gate). |
| R-08 | NEW | Customer overlap: teams using Alloy for ops resist second platform | MED | MED | Position as complementary (Datadog + W&B). Build Alloy import: pull production data into Torq for retraining. |

---

## 10. Open Technical Decisions

### Decision 1 — Canonical Episode Format

**Status: CLOSED** — Section 0.2 Product Scope already specifies *"Parquet + MP4 + JSON storage with master index"* as the R1 storage format. Decision is implicitly made: Arrow/Parquet (LeRobot-aligned) for non-image data, MP4 for image sequences, JSON for the master index. This decision is recorded here for audit trail only.

**Decision:** Parquet + MP4 + JSON index (LeRobot-aligned).
**Rationale:** Best ecosystem compatibility, aligns with the dominant community standard, reduces friction for LeRobot users importing existing datasets.

| Option | Pros | Competitive Consideration |
|---|---|---|
| ~~Arrow/Parquet (LeRobot-aligned)~~ | ~~Best ecosystem compatibility~~ | **Selected.** Alignment with LeRobot increases interop and accelerates academic adoption. |
| Zarr | HDF5-compatible, better for large trajectories | Deferred — revisit for R2 large-trajectory optimization. |
| Custom binary | Maximum control, highest switching costs | Rejected — risks ecosystem rejection at this stage. |

### Decision 2 — SDK Openness

**Question:** How open should the SDK be?

| Option | Pros | Competitive Consideration |
|---|---|---|
| Fully open-source SDK | Maximum adoption, W&B playbook | Strongest adoption play but means Alloy could wrap Torq's SDK. |
| Source-available with cloud dependency | — | — |
| Proprietary | — | Cloud-only composition/training features create the monetization boundary. |

### Decision 3 — First Vertical

**Status: CLOSED** — All personas (Maya, Jake, Sarah Kim), all hardware references (ALOHA-2, Franka), all acceptance criteria examples (pick/place tasks), and all competitive analysis reference manipulation robotics. The PRD has already answered this implicitly throughout. Recording the decision explicitly.

**Decision:** Manipulation robotics first.
**Rationale:** Largest VLA/ACT research activity, Alloy's design partners are in orthogonal verticals (drones, maritime, agriculture, warehousing), maximises competitive distance in Phase 1.

| Option | Pros | Competitive Consideration |
|---|---|---|
| ~~Manipulation~~ | ~~Largest VLA market, most research activity~~ | **Selected.** ALOHA, ACT, VLA community is the primary beachhead. |
| Locomotion | Unitree ecosystem, growing fast | Deferred — R3 expansion candidate once manipulation is established. |
| Surgical robotics | Highest ACV, strongest compliance needs | Deferred — high regulatory burden incompatible with R1 speed requirements. |

---

*Torq PRD v3.1 — Centroid Foundry — Confidential*

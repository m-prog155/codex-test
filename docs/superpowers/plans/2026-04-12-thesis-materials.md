# Thesis Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize thesis-ready experiment materials for the vehicle and license plate system, including final training comparison data and reusable writing fragments.

**Architecture:** Keep the work documentation-only. Use the verified remote training outputs as the single source of truth, then translate them into a local thesis materials document and refresh the implementation notes so the paper and system state stay aligned.

**Tech Stack:** Markdown, existing remote experiment outputs, local docs

---

### Task 1: Freeze final experiment facts

**Files:**
- Read: `D:\Projects\Car\docs\thesis-implementation-notes.md`
- Read: `D:\Projects\Car\README.md`
- Read remote: `/root/autodl-tmp/car-project/runs/plate_detector/training_summaries.csv`
- Read remote: `/root/autodl-tmp/car-project/outputs/smoke_remote_mvp/*`
- Read remote: `/root/autodl-tmp/car-project/outputs/smoke_video_mvp/*`

- [ ] Read the final remote training summary and sample output paths
- [ ] Confirm the final quick vs mvp comparison values and sample artifact locations

### Task 2: Write thesis-ready experiment materials

**Files:**
- Create: `D:\Projects\Car\docs\thesis-experiment-materials.md`

- [ ] Write a compact training comparison table with final metrics
- [ ] Write image and video figure caption suggestions with artifact paths
- [ ] Write one reusable experiment analysis paragraph
- [ ] Write one reusable system effect description paragraph

### Task 3: Refresh implementation notes

**Files:**
- Modify: `D:\Projects\Car\docs\thesis-implementation-notes.md`

- [ ] Update the engineering status section so it reflects the finished plate detector training and final remote config state
- [ ] Remove outdated wording that still describes detector connection and formal experiments as future work

### Task 4: Verify local thesis docs

**Files:**
- Verify: `D:\Projects\Car\docs\thesis-experiment-materials.md`
- Verify: `D:\Projects\Car\docs\thesis-implementation-notes.md`

- [ ] Read both documents back to confirm metrics, wording, and artifact paths are consistent

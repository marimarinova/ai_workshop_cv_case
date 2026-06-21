# Concepts and definitions

This file gives the whole team a **shared vocabulary** and, most importantly, a
**precise, agreed definition** of the two events you are detecting. Read it
before touching the data. If two people on the team label the same clip
differently, it is almost always because they skipped this page.

---

## 1. What kind of problem is this?

The umbrella term is **event detection in video** (sometimes called *temporal
action detection*, *temporal action localization*, or *action spotting*).
A few neighbouring terms are easy to mix up:

- **Action recognition** — given a short clip that you already know contains
  *one* action, say *which* action it is. (Input is already trimmed.)
- **Temporal action localization / detection** — given a *long, untrimmed* clip,
  find *where* the actions are (start/end) **and** what they are. **This is your
  task.**
- **Action spotting** — find the single *instant* an event happens, rather than
  an interval. A `pickup` is arguably an instant (the moment of grabbing); you
  can treat it either as an instant or as a short interval — just be consistent.

You are doing **temporal action detection on untrimmed video**: the footage is
long and mostly uneventful, and you must output *both* the label *and* the time.

---

## 2. Video basics (for non-specialists)

- **Frame** — a single still image. Video is a sequence of frames.
- **FPS (frames per second)** — how many frames make up one second of video. A
  30 fps clip has 30 images per second. You will almost always **downsample**
  (e.g. work at 2–8 fps) because consecutive frames are nearly identical and full
  fps is wasteful.
- **Resolution** — frame size in pixels (e.g. 1920×1080). You will usually
  **resize down** (e.g. to 224–512 px on the short side) to fit memory and run
  faster.
- **Timecode / timestamp** — the position of a moment inside a clip, measured in
  seconds from the clip's start (e.g. `t = 4.20 s`). You can convert between a
  frame index and a timestamp: `timestamp = frame_index / fps`.
- **Trimmed vs. untrimmed** — a trimmed clip contains exactly one action and
  little else; an untrimmed clip is raw footage with long gaps. Yours are
  untrimmed.
- **Clip / segment** — a clip is one video file; a segment is a sub-interval of
  it (e.g. the interval where a `pickup` happens).

---

## 3. The two events — precise definitions

Treat these as the **operational definitions** the entire team commits to. The
copy you actually annotate against lives in
[`../templates/labeling-guidelines.md`](../templates/labeling-guidelines.md);
keep the two consistent.

### `pickup`
> A person **removes an item from a shelf or surface and takes it into their
> hand(s)**, so that the item leaves its resting place and is carried/held by the
> person.

The defining moment is the **transfer of the item from the shelf to the hand**.
Mark the event time at that transfer (or as a short interval bracketing it).

### `putdown`
> A person **places an item back onto a shelf or surface**,
> releasing it from their hand(s) so that it rests there again.

Note the word **"taken"**: a `putdown` is the *return* of an item the person was
holding. The defining moment is the **release of the item onto the surface**.

### A useful mental model
A `pickup` and a `putdown` are roughly **time-reverses** of each other: hand
approaches → contact → item moves with hand (pickup) *or* item separates from
hand and stays (putdown) → hand leaves. Because they look similar, **the
direction of motion in time is what tells them apart.** Any method that throws
away temporal order will struggle to separate the two — keep that in mind when
you design Layers 1 and 2.

---

## 4. What is *not* an event (negatives)

Be strict about these, or your labels will be noisy:

- **Touching / inspecting without removing** — the person touches an item and leaves it where it was, without moving it. If the item does not leave
  its place, it is **not** a `pickup`.
- **Just looking / reaching past** — no item leaves the shelf.
- **Walking by, browsing, standing.**
- **Restocking / stuff appearing on shelves that was never "taken"** — a
  `putdown` is specifically the return of a *taken* item, not generic placement
  of new goods. (If your footage never contains restocking, you can ignore this,
  but write down the decision.)
- **Empty clips / no person** — these are removed in Layer 0, not labeled as
  events.

---

## 5. Edge-case rules (use these as given)

These are the agreed rulings for the awkward cases. Apply them as written so the
whole team labels the same way. They are repeated in
[`../templates/labeling-guidelines.md`](../templates/labeling-guidelines.md).

- **Grabbing two items at once** — **two events** (one `pickup` per item).
- **Pick up then immediately put back** — **two events**: a `pickup` followed by
  a `putdown`.
- **Item or hand occluded by the body / out of frame** — **exclude** it (do not
  add it to the events table).
- **Multiple people acting at the same time** — **label all** of the events.
- **Very brief or very ambiguous motions** — keep them, but **set the confidence
  flag to `low`** instead of forcing a clean yes/no.
- **Instant vs. interval** — store an **interval `[t_start, t_end]`** for every
  event (`t_start` = the action begins, `t_end` = the item is settled).
- **Hard cases** (multiple people in shot, partially obscured but still
  labelable, unusual motions) — keep them and set a separate **`hard_case` flag**
  so they can be found and reviewed later.

---

## 6. Ground truth, splits, and leakage

- **Ground truth** — the labels you trust as "correct". In this case, *you*
  create the ground truth in Layer 0. Its quality caps how good any model can
  look, so invest in it.
- **Train / validation / test split** — you partition your labeled clips into
  three groups: one to train on, one to tune on, one to *only* test on at the
  end. Keep the test set untouched until you are done.
- **Leakage** — when information from the test set sneaks into training, making
  results look better than they are. The classic trap here: **near-duplicate
  frames or the same recording session ending up in both train and test.** Split
  **by whole clip** (and, if you can tell, by person/session), never by random
  individual frames.
- **Class imbalance** — events are rare compared with "nothing happening". A
  detector that predicts "no event" everywhere can look deceptively accurate. Use
  metrics that account for this (below), and consider sampling windows around
  events when training.

---

## 7. How to tell whether a detector works

You compare predicted events against your ground-truth events. Because events are
**intervals** and exact agreement is unrealistic, a predicted event counts as
correct (a match) when it has the **right type** *and* its interval lines up with
a true interval of that type. Two common ways to decide "lines up":

- **Temporal IoU (tIoU)** — the overlap of the two intervals divided by their
  union; count a match when `tIoU ≥ 0.5` (or whatever threshold you pick).
- **Midpoint tolerance** — simpler: count a match when the interval midpoints are
  within, say, ±1 second of each other.

From the matched / unmatched events you get:

- **Precision** — of the events you predicted, how many were real?
- **Recall** — of the real events, how many did you find?
- **F1** — the balance of the two.
- A **confusion** count between `pickup` and `putdown` (how often the type is
  flipped), which is the interesting failure mode here.

If you want one number that sweeps the threshold, the standard choice in this
field is **mean Average Precision at a temporal IoU threshold (mAP@tIoU)** —
optional, and heavier to implement. Precision / recall / F1 at a fixed tolerance
is enough to learn a great deal.

> This evaluation is a **scientific measurement you run on your own model** so you
> know whether it is improving. It is not a grade and not a target to game —
> report it honestly, including where it fails.

---

## 8. A note on the people in the footage

The clips contain images of people. Handle them respectfully: keep the data
within your working storage, do not redistribute it, and do not try to identify
individuals (which is also explicitly out of scope). If you publish examples in a
write-up, prefer blurred faces or synthetic illustrations.
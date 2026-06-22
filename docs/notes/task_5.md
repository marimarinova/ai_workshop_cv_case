Yes—15 candidates for roughly 7–8 real events is reasonable for the later stages.

For candidate generation, prioritize high recall over precision. A practical target is around 1.5–3 candidates per true event. So for 7–8 events, roughly 10–25 candidates is acceptable.

Incorrect candidates should:

be marked rejected, not added to events.csv;
be retained as labeled false positives / hard negatives;
be used to train or calibrate the later classifier/VLM verifier.

Prioritize false positives that look plausibly like pickup/putdown. Obvious irrelevant candidates can be downsampled rather than keeping every one.
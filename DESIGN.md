# Visual Match UI System

## Product direction

Visual Match is a calm, professional workspace for finding an object or a person in an image or video. The interface should explain each step in everyday language and make the reference, target, and result easy to understand at a glance. The primary surface is a light canvas with restrained green accents, crisp typography, generous spacing, and clear upload states.

Keep all search capabilities intact: Object and Person modes, body appearance and face matching, image and video targets, reference selection, overlays, result selection, and detailed response data. Technical scores and raw response data belong under the collapsed **Advanced details** disclosure. Never expose internal service or model names in user-facing copy.

## Color tokens

| Role | Token | Value | Usage |
|---|---|---|---|
| Canvas | `--canvas` | `#f4f7f5` | Page background |
| Surface | `--surface` | `#ffffff` | Workflow and result cards |
| Soft surface | `--surface-soft` | `#f7f9f8` | Upload zones and quiet panels |
| Ink | `--ink` | `#1d2925` | Main text |
| Muted | `--muted` | `#68756f` | Supporting text |
| Faint | `--faint` | `#8b9791` | Metadata |
| Border | `--border` | `#dfe7e2` | Card and control outlines |
| Accent | `--accent` | `#16735c` | Primary actions and selected states |
| Accent dark | `--accent-dark` | `#105744` | Hover and active accent |
| Accent soft | `--accent-soft` | `#e8f4ef` | Selected and privacy surfaces |
| Error | `--error` | `#a93c38` | Input and request errors |

Use green for meaningful actions and active selections. Keep large surfaces neutral. Do not communicate state with color alone.

## Typography and layout

Use the system sans-serif stack; do not rely on remote font services. Use a 4px spacing base. The desktop content width is capped at 1120px. Keep a two-column workflow for the reference and target steps, with a visible locked target placeholder until a reference item is selected. Stack the workflow into one column below 760px and tighten spacing on small phones.

Use a compact uppercase label for step context, a clear card heading, then one short instruction. Display text is 32–42px, card headings 17px, body text 12–14px, controls 10–12px, and metadata 9–11px. Keep line lengths short and avoid dense technical explanations in the main flow.

## Workflow components

### Search mode

Show Object and Person as two selectable cards with a title and a short explanation. Use native buttons, visible selected state, and `aria-pressed`. Switching modes clears stale detections and target media so results cannot be mistaken for the previous mode.

### Reference photo

Accept one JPEG, PNG, WEBP, GIF, or BMP image up to 20 MB. The picker supports browse, drag-and-drop, keyboard activation, preview, and replacement. Before selection, show the upload prompt. While analyzing, explain that the step is in progress. After analysis, list detected objects or people as selectable options and highlight the selected region on the preview.

Object choices may show a short location label. Person choices may indicate whether a body or face is visible. Do not show detection percentages in the primary workflow.

### Target and person matching

Reveal the target step after the user selects one reference item. Offer Image, Video, and Folder controls, with image selected by default. Person mode adds Body appearance and Face match controls with clear guidance about the kind of reference photo that works best. If a selected person cannot be matched with the chosen method, explain what needs to change and keep the search action disabled.

Images accept JPEG, PNG, WEBP, GIF, or BMP up to 20 MB. Videos accept MP4, MOV, and WEBM up to 200 MB. Show a local preview, filename, and replacement control. The primary action stays disabled until the required reference, selection, and target are ready.

Folder selection includes supported image and video files recursively. Where browser folder selection is unavailable, allow multiple files to be selected instead. Show supported and skipped file counts before the scan begins. Process one file at a time, report progress, continue after an individual file error, and allow the user to stop after the current file. Preserve each file’s folder-relative path in the results.

### Results

Use a persistent results panel with an empty prompt, a concise progress state, or selectable result cards. Result cards use plain labels such as “Top match” and “Match 2”; do not put numeric similarity scores, meters, thresholds, or technical scoring explanations in the main result view. Selecting a candidate updates the paired reference and target previews and their highlighted regions.

For video, show each appearance as a time range and frame count. Selecting one seeks to the relevant moment and shows the highlighted region. Do not show numeric similarity scores in the appearance list.

For folder scans, group each file’s result by relative path and show its scan state and matches. Keep file results individually expandable so large folders remain navigable. Retain image comparison and video appearance controls within each expanded result.

### Advanced details

Keep the disclosure hidden until a search has completed. It starts collapsed. Put numeric scores, raw JSON, and copy controls inside it. Explain briefly that detailed scores are estimates. Reset the disclosure to collapsed whenever a new search starts or the inputs change.

## Accessibility and motion

- Use semantic headings, native buttons, and descriptive labels for every picker and preview.
- Expose selected modes and result choices with `aria-pressed`.
- Announce search progress, completion, and errors through a polite live region.
- Keep focus indicators visible, maintain adequate text contrast, and pair status colors with text.
- Use transitions around 150ms for borders and surfaces. Animate only transform and opacity; respect `prefers-reduced-motion`.
- Never autoplay a video. Seeking follows an explicit appearance selection.

# Design System: Real-Time Object Detection & Tracking

## 1. Visual Theme & Atmosphere
A cockpit-calibrated, precision computer vision telemetry interface with high visual density and clinical clarity. The visual language evokes an aerospace mission control console rather than a generic tech demo. Visual Density is calibrated at 7/10, Design Variance at 6/10, and Motion Intensity at 6/10.

## 2. Color Palette & Roles
- **Obsidian Void** (`#09090B`) — Deep charcoal canvas background (Zinc-950)
- **Console Surface** (`#121215`) — Panel and card surface elevation (Zinc-900)
- **Subtle Frontier** (`#27272A`) — 1px structural borders and dividers (Zinc-800)
- **Telemetry Emerald** (`#10B981`) — Primary accent, active track indicator, confident detections
- **Amber Warning** (`#F59E0B`) — Secondary telemetry status, intermediate confidence detections
- **Signal Blue** (`#3B82F6`) — Bounding box track vectors and trajectory trails
- **Primary Text** (`#F4F4F5`) — Primary text and headline display (Zinc-100)
- **Muted Readout** (`#A1A1AA`) — Secondary metrics and descriptions (Zinc-400)

## 3. Typography Rules
- **Display & Section Headers:** `Satoshi`, `Geist Sans`, or system `-apple-system, BlinkMacSystemFont`, tracking-tighter, font-weight 600.
- **Body Text:** `Geist Sans` or `Segoe UI`, relaxed leading, max line width 65ch.
- **Telemetry & Metrics:** `JetBrains Mono` or `SF Mono`, tabular numbers for FPS, frame numbers, object IDs, and confidence ratios.
- **Banned:** `Inter`, comic fonts, default generic serif (`Times New Roman`, `Georgia`).

## 4. Component Stylings
- **Video Viewport:** Generously rounded container (`rounded-2xl`), 1px subtle frontier border (`#27272A`), enclosed HUD status pills at the top corner.
- **Control Sliders & Selectors:** Clean horizontal rails, `#10B981` active track fill, clear numeric label positioned strictly above controls.
- **Bounding Boxes:** Subdued 2px rounded bounding boxes with semi-transparent corner brackets. Pill badges for class and track ID, eliminating visual clutter.
- **Telemetry Cards:** 1px border dividers instead of heavy card stacking. Tabular numbers with fixed widths to prevent layout jitter during live inference.
- **Empty & Idle States:** Composed empty states with SVG wireframe guides showing camera/video readiness.

## 5. Layout Principles
- Split-screen telemetry layout: Left pane for video input stream and real-time bounding box annotations; right pane for metric telemetry, object class breakdown, and tracking history table.
- Mobile collapse: Automatically collapses to single-column vertical stack under 768px with video priority on top.
- Viewport containment: `max-w-[1400px]` centered with strict avoidance of viewport overflow.

## 6. Motion & Interaction
- Smooth 60fps tracking vector trails with fading alpha gradients over the last 30 frames.
- Tactile button press feedback (`active:scale-[0.98]`).
- Shimmer skeleton loaders during model weight initialization.
- Hardware-accelerated CSS transforms (`transform`, `opacity`) without layout shifts.

## 7. Anti-Patterns (Strictly Enforced)
- Zero emojis in UI labels, documentation, and console logs.
- No neon purple/cyan gradient glow cliches.
- No pure black (`#000000`) surfaces.
- No generic filler statistics or fake random mock data.
- No centered hero sections or floating ungrounded UI components.

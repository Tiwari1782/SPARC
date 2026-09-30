# SPARC Icon Guide

SPARC uses **Font Awesome icons through `react-icons`**. No emoji appears anywhere in the project: not in the UI, README, commit messages, code comments, console output or documentation.

---

## 1. Rules

1. Use only the Font Awesome 6 set from `react-icons/fa6`.
2. Never use emoji characters or emoji-style symbols in source files, strings, logs or docs.
3. Import icons individually (tree-shaken), never the whole package.
4. Icons never carry meaning alone. Pair every status icon with a text label and a colour, so the UI is readable without colour vision.
5. Keep one icon per concept across the whole app.
6. Default size is 16 px in tables, 20 px in buttons, 28 px or more in panel headers.

## 2. Install

```bash
cd frontend
npm install react-icons
```

## 3. Usage

```jsx
import { FaBatteryHalf, FaBolt } from "react-icons/fa6";

export function BatteryLabel({ soc }) {
  return (
    <span className="battery-label">
      <FaBatteryHalf size={20} aria-hidden="true" />
      <span>{soc.toFixed(0)}%</span>
    </span>
  );
}
```

Use `aria-hidden="true"` on decorative icons and an `aria-label` on icon-only buttons.

## 4. Central Icon Map (recommended)

Keep all icons in one file so they can be changed in one place.

```jsx
// frontend/src/icons.js
export {
  FaBatteryFull,
  FaBatteryThreeQuarters,
  FaBatteryHalf,
  FaBatteryQuarter,
  FaBatteryEmpty,
  FaBolt,
  FaPlugCircleBolt,
  FaTriangleExclamation,
  FaCircleExclamation,
  FaCircleCheck,
  FaFlagCheckered,
  FaCarSide,
  FaRoute,
  FaMapLocationDot,
  FaChartLine,
  FaGaugeHigh,
  FaPlay,
  FaPause,
  FaForwardStep,
  FaRotateLeft,
  FaCloudRain,
  FaTemperatureHigh,
  FaShieldHalved,
  FaUsers,
  FaCodeCompare,
  FaDatabase,
  FaClockRotateLeft,
  FaTrophy,
  FaBrain,
  FaSliders,
  FaCircleInfo,
  FaBell,
} from "react-icons/fa6";
```

## 5. Icon Mapping

### Battery state

| UI element | Icon | Used for |
|---|---|---|
| Battery 75 to 100 percent | `FaBatteryFull` | Healthy SoC |
| Battery 50 to 75 percent | `FaBatteryThreeQuarters` | Comfortable SoC |
| Battery 25 to 50 percent | `FaBatteryHalf` | Watch |
| Battery 10 to 25 percent | `FaBatteryQuarter` | Low |
| Battery below 10 percent | `FaBatteryEmpty` | Clip imminent |

### Energy flow

| UI element | Icon | Used for |
|---|---|---|
| Deploy zone | `FaBolt` | Spending battery on a straight |
| Harvest zone | `FaPlugCircleBolt` | Recovering energy under braking |
| Clip point marker | `FaTriangleExclamation` | Detected or predicted clipping |

### Severity

| State | Icon | Colour token |
|---|---|---|
| GREEN | `FaCircleCheck` | `--sparc-green` |
| AMBER | `FaCircleExclamation` | `--sparc-amber` |
| RED | `FaTriangleExclamation` | `--sparc-red` |
| CRITICAL | `FaTriangleExclamation` with pulse animation | `--sparc-critical` |

### Navigation and views

| View | Icon |
|---|---|
| Race view | `FaFlagCheckered` |
| Track map | `FaMapLocationDot` |
| Driver deep dive | `FaCarSide` |
| Energy battle | `FaCodeCompare` |
| Driver compare | `FaUsers` |
| Charts | `FaChartLine` |
| Alert log | `FaBell` |
| History | `FaClockRotateLeft` |
| Race energy story | `FaTrophy` |
| Model info | `FaBrain` |
| Settings and calibration | `FaSliders` |

### Controls

| Control | Icon |
|---|---|
| Play | `FaPlay` |
| Pause | `FaPause` |
| Next lap | `FaForwardStep` |
| Reset | `FaRotateLeft` |
| Mode toggle (data) | `FaDatabase` |

### Injection panel

| Injection | Icon |
|---|---|
| Safety car | `FaShieldHalved` |
| Rain | `FaCloudRain` |
| Cooling fault | `FaTemperatureHigh` |
| Harvest failure | `FaPlugCircleBolt` |
| Rival attack | `FaBolt` |
| Low-grip track | `FaRoute` |

## 6. Colour Tokens

Define once in the global stylesheet and reuse. Provide light and dark values.

```css
:root {
  --sparc-green: #1f9d55;
  --sparc-amber: #d98e04;
  --sparc-red: #d64545;
  --sparc-critical: #a10d2b;
}
```

## 7. Icons Outside the UI

- README, docs and commit messages: plain text only, no icons or emoji.
- Console and log output: use text prefixes such as `[INFO]`, `[WARN]`, `[ERROR]`.
- Favicon and logo: a simple vector mark (SVG) of a spark or lightning shape, created separately from the icon set.

## 8. Checklist Before Merging UI Changes

- [ ] Only `react-icons/fa6` imports, all from `icons.js`
- [ ] No emoji characters in code, strings, comments or docs
- [ ] Every status icon has a text label
- [ ] Icon-only buttons have an `aria-label`
- [ ] Colours come from the tokens above
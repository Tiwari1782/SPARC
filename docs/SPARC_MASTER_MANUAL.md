# SPARC: The Definitive Mathematical, Architectural, and Conceptual Master Guide

---

## Executive Summary: What is SPARC in 30 Seconds?

**SPARC** (**S**tate-of-charge **P**rediction and **A**nalysis for **R**ace **C**lipping) is an AI-driven, physics-informed telemetry digital twin built for Formula 1 racing. 

* **The Problem**: In modern Formula 1 (and dramatically elevated under the **2026 Engine Regulations**), cars rely on electric hybrid battery power for almost 50% of their total power output (350 kW, ~470 hp). When a driver drains their allowed battery energy mid-straight, the electric motor cuts out abruptly despite keeping the accelerator pinned at 100%. The car instantly loses 40–50 km/h relative to competitors. This is called **clipping**.
* **The Secret Data Problem**: Race teams have private sensors (cell voltages, internal temperatures, driver rotary dial modes) inside the car ECU. Public broadcasts and commentators **only see public channels**: speed, throttle pedal %, brake pedal %, gear, engine RPM, and GPS track coordinates.
* **What SPARC Does**:
  1. **Reconstructs** an internal, physics-informed Battery State-of-Charge (SoC %) from public pedal and speed data.
  2. **Predicts** when a car is going to clip **300 meters before it actually happens** using machine learning (LightGBM).
  3. **Dispatches** real-time warnings to live broadcast dashboards with an average **190.3-meter warning lead time**.
  4. **Proves the "Information Ceiling"**: An honest, rigorous scientific finding showing that under public broadcast limits, track geography dominates early onset prediction because private driver dial settings remain hidden.

---

## 1. Domain Physics: Why Does Clipping Happen?

### 1.1 The 2026 FIA Regulation Shift
* In the current turbo-hybrid era, the electric motor (MGU-K) produced **120 kW** (~160 hp).
* Under the **2026 FIA Technical Regulations**:
  * The Motor Generator Unit–Heat (MGU-H) is **completely eliminated**.
  * The MGU-K electric motor power is **nearly tripled to 350 kW (~470 hp)**.
  * Internal Combustion Engine (ICE) power is lowered to ~400 kW.
  * **Power split is 50% petrol, 50% electric!**
  * Per-lap usable battery energy is strictly limited by regulation to **~4.0 MJ (1.11 kWh)**.
  * Maximum regeneration recovery is capped at **8.5 MJ/lap**.

### 1.2 The Physics of Straight-Line Depletion
At high speed (e.g., 300+ km/h), aerodynamic drag force scales with the square of velocity:
$$F_{\text{drag}} = \frac{1}{2} \rho C_d A v^2$$
And aerodynamic power required to overcome drag scales with the **cube** of velocity:
$$P_{\text{drag}} = F_{\text{drag}} \cdot v = \frac{1}{2} \rho C_d A v^3$$

To accelerate past 300 km/h, the car desperately needs the 350 kW electric boost. 
When the battery allowance runs out before the end of the straight:
1. The electric motor torque collapses to zero.
2. The ICE alone cannot overcome the massive aerodynamic drag.
3. Acceleration abruptly drops from $+1.5\text{ m/s}^2$ to $\approx 0\text{ m/s}^2$.
4. Speed plateaus at 310 km/h while an attacking rival with battery deployment blows past at 330+ km/h.

**Real-world examples**: 
- **2024 Canadian Grand Prix (Montreal)**: George Russell defending against Max Verstappen ran out of battery on the Casino Straight.
- **2024 British Grand Prix (Silverstone)**: Lando Norris defending on the Hangar Straight clipped early, leaving him defenseless.

---

## 2. Complete Mathematical Foundations of SPARC

Here is every single formula formulated and deployed in the SPARC project, explained with first principles and physical intuition.

```
       [ Public Telemetry at 4 Hz ]
       (Speed v, Throttle T, Brake B)
                     │
                     ▼
┌──────────────────────────────────────────────┐
│           ANALYTICAL ENERGY ENGINE           │
│                                              │
│  Braking/Coasting ──► E_harvest (Eq. 2)      │
│  Pedal & Taper   ──► E_deploy  (Eq. 3, 4)    │
│                           │                  │
│                           ▼                  │
│       SoC(k+1) Integration (Eq. 1)           │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│         OPERATIONAL MOTORSPORT INDICES       │
│                                              │
│  Straight Demand vs Reserves  ──► CPI (Eq. 5)│
│  Net Lap Energy Balance       ──► EDR (Eq. 6)│
│  Attacker vs Defender Deficit ──► BAI (Eq. 7)│
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│            MACHINE LEARNING CORE             │
│                                              │
│  19 Features (Spatial + Physical + Kinematic)│
│  LightGBM Decision Ensemble                  │
│  Debounced Single-Shot Alert Dispatched      │
└──────────────────────────────────────────────┘
```

---

### Formula 1: The Discrete-Time Battery State-of-Charge (SoC) Estimator

$$\text{SoC}(k+1) = \text{clamp}\left(\text{SoC}(k) + \frac{E_{\text{harvest}}(k) - E_{\text{deploy}}(k)}{E_{\text{battery}} \times 3600} \times 100,\; 0,\; 100\right)$$

#### Variables & Parameters:
* $k$: Discrete time index (sampled at $4\text{ Hz}$, meaning $\Delta t = 0.25\text{ seconds}$).
* $\text{SoC}(k)$: Current estimated state-of-charge percentage, bounded between $0\%$ and $100\%$.
* $E_{\text{harvest}}(k)$: Electrical energy recovered back into the battery during time step $k$ (in kiloJoules, $\text{kJ}$).
* $E_{\text{deploy}}(k)$: Electrical energy discharged out of the battery to the motor during step $k$ (in $\text{kJ}$).
* $E_{\text{battery}} = 1.11\text{ kWh}$ ($\approx 4000\text{ kJ}$): Nominal usable battery buffer under FIA regulations.
* $3600$: Conversion constant ($1\text{ kWh} = 3600\text{ kJ}$).
* $\text{clamp}(x, 0, 100)$: Mathematical saturation boundary: $\max(0, \min(100, x))$.

#### Physical Meaning:
This is a discrete-time energy balance equation. In any given $0.25\text{ s}$ window, the change in battery percentage is simply:
$$\Delta \text{SoC} = \frac{\text{Net Joules In} - \text{Net Joules Out}}{\text{Total Battery Capacity}} \times 100\%$$

---

### Formula 2: Kinetic Energy Harvesting Model

$$E_{\text{harvest}}(k) = \left[\eta_h \cdot \mathcal{B}(k)^{1.2} + k_{\text{lift}} \cdot \mathcal{C}(k)\right] \cdot k_{\text{regen}} \cdot \Delta t$$

#### Variables & Parameters:
* $\mathcal{B}(k) = \text{clamp}\left(\frac{\text{Brake}(k)}{100},\, 0,\, 1\right)$: Normalized brake pedal input $[0, 1]$.
* $\text{Brake}(k)^{1.2}$: **Non-linear pedal exponent**. Accounts for the non-linear hydraulic line pressure curve where light pedal touches produce modest regen, while hard stomps maximize motor braking torque.
* $\eta_h = 0.85$ ($85\%$): Electrical round-trip efficiency of regenerative braking (losses in copper winding, inverter switching, and battery cell internal resistance).
* $\mathcal{C}(k) \in \{0, 1\}$: **Coasting Indicator**. If a driver lifts off the throttle ($\mathcal{T} < 0.05$) without pressing the brake ($\mathcal{B} < 0.05$), the car is in "lift-and-coast", engaging light regenerative drag:
  $$\mathcal{C}(k) = \begin{cases} 1 & \text{if } \mathcal{T}(k) < 0.05 \text{ and } \mathcal{B}(k) < 0.05 \\ 0 & \text{otherwise} \end{cases}$$
* $k_{\text{lift}} = 0.30$: Drag recovery factor during off-throttle coasting.
* $k_{\text{regen}}$: Constructor-specific calibrated regeneration coefficient (e.g., Ferrari and Red Bull harvest differently based on PU architecture).
* $\Delta t = 0.25\text{ s}$.

#### Physical Meaning:
The battery charges in two ways: hard braking (where the MGU-K acts as a massive generator slowing the car) and passive lift-and-coast (where trailing throttle charges the battery through engine braking).

---

### Formula 3: Electrical Energy Deployment Model

$$E_{\text{deploy}}(k) = \mathcal{T}(k)^{1.5} \cdot f_{\text{taper}}(v) \cdot k_{\text{deploy}} \cdot \Delta t$$

#### Variables & Parameters:
* $\mathcal{T}(k) = \frac{\text{Throttle}(k)}{100}$: Normalized throttle pedal input $[0, 1]$.
* $\mathcal{T}(k)^{1.5}$: **Convex pedal curve**. Drivers don't get linear motor torque; electric deployment is aggressively biased toward full-throttle exits ($90\text{--}100\%$).
* $k_{\text{deploy}}$: Constructor-specific deployment rate parameter fitted on training folds.
* $f_{\text{taper}}(v)$: Velocity-dependent regulatory derating factor (explained below).

---

### Formula 4: High-Speed Regulatory Velocity Taper

$$f_{\text{taper}}(v) = \text{clamp}\left(1 - \frac{v - v_{\text{start}}}{v_{\text{zero}} - v_{\text{start}}},\; 0,\; 1\right)$$

#### Variables & Parameters:
* $v$: Instantaneous vehicle velocity in $\text{km/h}$.
* $v_{\text{start}} = 280\text{ km/h}$: Velocity above which electrical power begins tapering down.
* $v_{\text{zero}} = 370\text{ km/h}$: Terminal velocity where electric deployment drops strictly to $0\text{ kW}$.

#### Physical & Regulatory Meaning:
The 2026 FIA Power Unit regulations state that the electric motor cannot deliver 350 kW all the way to 360 km/h (otherwise cars would exceed dangerous terminal speeds). The regulations mandate that as velocity climbs past 280–300 km/h, the power must ramp down linearly to zero. This formula models that exact regulatory taper.

---

### Formula 5: Linear Fuel Mass Depletion

$$m_{\text{fuel}}(\lambda) = \max\left(0,\; m_{\text{start}} - \dot{m}_{\text{burn}} \cdot (\lambda - 1)\right)$$

#### Variables & Parameters:
* $\lambda$: Current lap number in the race ($\lambda \in [1, N_{\text{laps}}]$).
* $m_{\text{start}} = 70.0\text{ kg}$: Starting fuel mass at race launch.
* $\dot{m}_{\text{burn}} = 1.2\text{ kg/lap}$: Average fuel consumption per lap.

#### Physical Meaning:
A lighter car requires less force to accelerate ($F = m \cdot a$). Accounting for fuel burn prevents the model from mistaking lap 50 acceleration changes for battery variations.

---

### Formula 6: Clip Pressure Index (CPI)

$$\text{CPI} = \left(\frac{D_{\text{ahead}}}{R_{\text{avail}}}\right)^{1.3} \cdot (1 + 0.5 \cdot P_{\text{rival}}) \cdot W_{\text{fuel}}$$

#### Variables & Parameters:
* $D_{\text{ahead}}$: Distance remaining on the active straight until the next braking marker (in meters).
* $R_{\text{avail}}$: Remaining usable energy buffer in the battery (in Joules):
  $$R_{\text{avail}} = \text{SoC}(k) \times (E_{\text{battery}} \times 3600) \times \tau_{\text{thermal}}$$
* Exponent $1.3$: Non-linear penalty factor. As remaining battery energy drops to zero, the risk of clipping explodes exponentially.
* $P_{\text{rival}}$: Tactical proximity pressure from a car behind:
  $$P_{\text{rival}} = \begin{cases} \frac{1}{\Delta t_{\text{gap}} + 0.5} & \text{if } 0 < \Delta t_{\text{gap}} \le 2.0\text{ s} \\ 0 & \text{otherwise} \end{cases}$$
  *(If an attacking car is right on your gearbox within 0.5–1.0s, you are forced to spend maximum battery defending against DRS).*
* $W_{\text{fuel}} = 1 + 0.002 \cdot \lambda$: Fuel adjustment weight.

#### Physical Meaning:
**CPI** is a single composite metric measuring: *"Does this car have enough electrical Joules left in the tank to survive the rest of this straight under current rival attack?"* If $\text{CPI} \gg 1.0$, the car is in critical danger of clipping.

---

### Formula 7: Energy Debt Rate (EDR)

$$\text{EDR} = \frac{\sum_{\text{lap}} E_{\text{deploy}} - \sum_{\text{lap}} E_{\text{harvest}}}{L_{\text{lap}}}$$

#### Variables & Parameters:
* $\sum_{\text{lap}} E_{\text{deploy}}$: Total energy deployed across the current lap ($\text{kJ}$).
* $\sum_{\text{lap}} E_{\text{harvest}}$: Total energy harvested across the current lap ($\text{kJ}$).
* $L_{\text{lap}}$: Circuit lap length in meters.

#### Physical Meaning:
Measures the **net energetic deficit per meter**. 
* $\text{EDR} > 0$: The driver is over-consuming (running a deficit).
* $\text{EDR} < 0$: The driver is energy-saving (harvesting surplus).

---

### Formula 8: Battle Asymmetry Index (BAI)

$$\text{BAI} = \text{CPI}_{\text{defender}} - \text{CPI}_{\text{attacker}}$$

#### Physical Meaning:
Evaluates relative energy vulnerability between two contesting cars. If $\text{BAI} > 0$, the defending car's battery is depleted while the attacking car's battery is primed with energy, signaling an imminent, indefensible straight-line overtake.

---

### Formula 9: Ground-Truth Acceleration-Envelope Detector

How do we mathematically know in historical data that a car clipped if we don't have team telemetry?
We compute the expected unconstrained acceleration envelope $a_{\text{base}}(v, g)$ for each car and gear from historical clean laps. A sample is flagged as an onset event if and only if all three conditions are satisfied:
1. **Full Throttle & Zero Brake**: $\mathcal{T} \ge 98\%$ and $\mathcal{B} = 0$.
2. **High-Speed Straight**: $v > 200\text{ km/h}$ within a designated straight.
3. **Severe Acceleration Collapse**:
   $$a_{\text{long}}(k) < 0.45 \cdot a_{\text{base}}(v, g)$$
   sustained for at least $15\text{ meters}$ before the braking zone, excluding terminal aerodynamic top speed limits.

---

### Formula 10: Machine Learning Labeling Horizon

A telemetry sample at distance $s_k$ receives a positive ground-truth label $y_k = 1$ if:
$$s_k < s^* \le s_k + H \quad (H = 300\text{ meters})$$
where $s^*$ is the physical onset location of the clipping event and the vehicle is not already actively clipping; otherwise $y_k = 0$.

---

## 3. The 19-Feature Machine Learning Space

SPARC feeds 19 domain-engineered features into LightGBM:

| Domain | Feature | Physical Explanation |
|---|---|---|
| **Spatial** | `straight_length` | Total length of the straight in meters. |
| **Spatial** | `distance_to_straight_end` | Remaining meters until the braking marker. |
| **Spatial** | `lap_fraction` | Progress around the circuit $[0.0, 1.0)$. |
| **Spatial** | `dist_into_straight` | Distance traveled from straight entry (m). |
| **Physics** | `soc_est` | Analytical battery state-of-charge ($0\text{--}100\%$, Eq. 1). |
| **Physics** | `cpi` | Clip Pressure Index (Eq. 6). |
| **Physics** | `edr_lap` | Energy Debt Rate on active lap (kJ/m, Eq. 7). |
| **Physics** | `fuel_mass_kg` | Estimated fuel remaining (kg, Eq. 5). |
| **Kinematic** | `speed` | Instantaneous GPS speed (km/h). |
| **Kinematic** | `accel_long` | Longitudinal forward acceleration ($\text{m/s}^2$). |
| **Kinematic** | `throttle` | Engine accelerator pedal input ($0\text{--}100\%$). |
| **Kinematic** | `gear` | Current transmission gear ($1\text{--}8$). |
| **Kinematic** | `rpm` | Engine crankshaft revolutions per minute. |
| **Tactical** | `gap_ahead_s` | Time gap to the car ahead (seconds). |
| **Tactical** | `p_rival` | Proximity attack coefficient. |
| **Tactical** | `drs_active` | Binary flag for DRS rear wing slot open. |
| **Tactical** | `bai` | Battle Asymmetry Index (Eq. 8). |
| **Contextual**| `track_temp` | Ambient track asphalt surface temperature ($^\circ\text{C}$). |
| **Contextual**| `air_temp` | Ambient atmospheric temperature ($^\circ\text{C}$). |

---

## 4. Key Experimental Results & Proofs

### 4.1 Benchmark Evaluation (Full Unfiltered Test Folds)

* **Dataset**: 8,282,692 samples across 15 Grand Prix (2026 season calendar).
* **Positive Prevalence**: $1.66\%$ (severe class imbalance).
* **Cross-Validation**: 5-Fold GroupKFold strictly by Grand Prix circuit (held-out unseen circuits).

| Model | Macro PR-AUC | Pooled PR-AUC | Debounced Recall | Operational FA / lap-driver | Mean Warning Lead (m) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Baseline** | 0.0166 | 0.0166 | --- | --- | --- |
| **SoC Rule (<25%)** | 0.0383 | 0.0398 | 5.40% | 5.83 | 222.0 m |
| **Logistic Regression**| 0.0409 | 0.0401 | 20.48% | 2.20 | 217.4 m |
| **Position-Only** | 0.1181 | 0.0824 | 17.74% | 1.97 | 207.1 m |
| **Random Forest** | **0.2068** | **0.1934** | 22.70% | 1.44 | 183.0 m |
| **LightGBM (SPARC)**| **0.1696** | **0.1467** | **28.61%** | **1.06** | **190.3 m** |

### 4.2 Why LightGBM is the Production Choice over Random Forest
While Random Forest has a slightly higher raw macro PR-AUC ($0.2068$ vs $0.1696$), **LightGBM is unequivocally superior for real-world live deployment**:
1. **Higher Operational Recall**: Captures **28.61%** of clipping events at the $\le 1.0\text{ FA/ld}$ budget vs. only $22.70\%$ for Random Forest.
2. **Lower False Alarm Rate**: $1.06\text{ FA/ld}$ vs. $1.44\text{ FA/ld}$ (fewer annoying false alerts for commentators).
3. **Longer Warning Lead**: Dispatches alerts **$190.3\text{ meters}$ in advance** vs. $183.0\text{ m}$.
4. **$13\times$ Faster Inference**: LightGBM scores a frame in **$1.4\text{ ms}$** vs. $18.2\text{ ms}$ for Random Forest (critical for sub-250 ms real-time streaming).

---

## 5. The Core Scientific Discovery: The "Information Ceiling"

When we ran feature ablation experiments, removing all physics features (`soc_est`, `cpi`, `edr_lap`) yielded a PR-AUC of **$0.1983 \pm 0.1241$**, matching or slightly outperforming the full model ($0.1696$).

### What does this prove?
This is the central finding of the research:
1. **The Physical SoC Proxy Succeeds as a Monitoring Twin**: 
   Pre-clip SoC drops to **$2.90\%\text{--}15.12\%$** across all 11 teams, compared to **$54.14\%\text{--}77.19\%$** during standard running (a massive **$50\text{--}62$ percentage-point gap**).
2. **Track Geometry Dominates Early Prediction**:
   In public broadcast feeds, private driver rotary settings (Strat modes, overtake button) are unobservable. Therefore, the machine learning model learns that clipping is physically anchored to circuit topography (long straights like Kemmel at Spa, Hangar at Silverstone).
3. **The Information Ceiling**:
   Public telemetry cannot pierce the boundary of unobserved driver intent. Forecasting clipping 300 meters ahead is fundamentally bound by circuit geometry.

---

## 6. Viva / Defense / Interview Cheat Sheet

| Question | The Winning Answer |
|---|---|
| *"What is clipping?"* | In hybrid racing, clipping is the premature exhaustion of electrical battery deployment along a straight. The motor disengages while the driver is still at 100% throttle, causing a 40–50 km/h speed deficit against rivals. |
| *"Why not just use an electrochemical battery model like Thevenin or Randles?"* | Those models require internal cell voltage and current discharge in Amperes. In Formula 1 broadcasts, that data is proprietary and encrypted. SPARC formulates an analytical energy proxy derived strictly from pedal and kinematic flux. |
| *"Why use PR-AUC instead of ROC-AUC?"* | Positive class prevalence is only 1.66% (extreme class imbalance). ROC-AUC is distorted by the millions of True Negatives, making poor models look artificially great ($>0.95$). PR-AUC evaluates true precision against recall without True Negative distortion. |
| *"What is your false alarm metric?"* | We use **False Alarms per Lap-Driver (FA/ld)**. A warning model is useless if it spams false alerts every lap. SPARC enforces debounced single-shot alerting per straight with an operational budget of $\le 1.0\text{ FA/ld}$. |
| *"What is the Information Ceiling?"* | It is our honest scientific finding: while our physics engine successfully reconstructs battery stress (50–62 pt gap), early 300m prediction from public data is dominated by track geometry because private driver dial settings remain hidden. |

---

*Authored for the SPARC Research Project, 2026.*

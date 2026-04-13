/**
 * app.js — EMBER wildfire risk assessor frontend
 *
 * Handles form submission, chat conversation, and result rendering.
 */

/* ─── Helpers ────────────────────────────────────────────────────────────── */

/**
 * Append a new chat bubble to the conversation window.
 * @param {"system"|"user"} role
 * @param {HTMLElement|string} content  - HTML string or element
 * @returns {HTMLElement} the inserted bubble div
 */
function appendBubble(role, content) {
  const win = document.getElementById("chatWindow");

  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}-avatar`;
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "system" ? "🤖" : "👤";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}-bubble`;

  if (typeof content === "string") {
    bubble.innerHTML = content;
  } else {
    bubble.appendChild(content);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  win.appendChild(row);

  // Scroll to bottom
  win.scrollTop = win.scrollHeight;
  return bubble;
}

/** Remove a specific element from the DOM (used to remove the spinner). */
function removeBubble(el) {
  if (el && el.closest(".bubble-row")) {
    el.closest(".bubble-row").remove();
  }
}

/** Format a number with one decimal place. */
function fmt1(n) {
  return Number(n).toFixed(1);
}

/** Return a hex color string for a risk level string. */
/** Default color used when a risk level is not recognised. */
const DEFAULT_RISK_COLOUR = "#f97316";

function riskColor(level) {
  const map = {
    LOW:      "#22c55e",
    MODERATE: "#eab308",
    HIGH:     DEFAULT_RISK_COLOUR,
    EXTREME:  "#ef4444",
  };
  return map[level] ?? DEFAULT_RISK_COLOUR;
}

/** Return a human-readable description for the risk level. */
function riskDescription(level, score) {
  const descriptions = {
    LOW:      `This fire shows a <strong>${fmt1(score)}%</strong> estimated probability of escalating to a large fire (&gt;200 ha). Current conditions are relatively benign.`,
    MODERATE: `This fire shows a <strong>${fmt1(score)}%</strong> estimated probability of escalating to a large fire. Conditions warrant monitoring and pre-positioning of resources.`,
    HIGH:     `This fire shows a <strong>${fmt1(score)}%</strong> estimated probability of escalating to a large fire. Aggressive initial attack is recommended.`,
    EXTREME:  `This fire shows a <strong>${fmt1(score)}%</strong> estimated probability of escalating to a large fire. Conditions are severe — immediate full suppression response advised.`,
  };
  return descriptions[level] ?? "";
}

/**
 * Build a SVG ring gauge.
 * @param {number} pct  0–100
 * @param {string} color  hex/rgb
 */
function buildGauge(pct, color) {
  const R = 28;                         // radius of the arc
  const C = 2 * Math.PI * R;            // circumference ≈ 175.9
  const offset = C - (pct / 100) * C;

  const wrap = document.createElement("div");
  wrap.className = "risk-gauge-wrap";

  wrap.innerHTML = `
    <svg viewBox="0 0 76 76" aria-hidden="true">
      <circle class="gauge-bg"  cx="38" cy="38" r="${R}" />
      <circle class="gauge-arc" cx="38" cy="38" r="${R}"
              stroke="${color}"
              stroke-dasharray="${C}"
              stroke-dashoffset="${C}" />
    </svg>
    <div class="gauge-pct-text" style="color:${color}">${Math.round(pct)}%</div>
  `;

  // Animate after a brief delay so the transition fires
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const arc = wrap.querySelector(".gauge-arc");
      if (arc) arc.style.strokeDashoffset = offset;
    });
  });

  return wrap;
}

/**
 * Build a single SHAP factor bar.
 * @param {{ name:string, contribution:number }} factor
 * @param {number} maxContrib  the highest contribution in the list (for scaling)
 */
function buildFactorBar(factor, maxContrib) {
  const pct = maxContrib > 0 ? (factor.contribution / maxContrib) * 100 : 0;

  const item = document.createElement("div");
  item.className = "factor-item";
  item.innerHTML = `
    <div class="factor-label-row">
      <span class="factor-name">${factor.name}</span>
      <span class="factor-pct">${fmt1(factor.contribution)}%</span>
    </div>
    <div class="factor-bar-bg">
      <div class="factor-bar-fill" style="width:0%" data-target="${pct}"></div>
    </div>
  `;
  return item;
}

/** Animate all factor bar fills to their data-target widths. */
function animateFactorBars(container) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".factor-bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.target + "%";
      });
    });
  });
}

/* ─── User message summary ───────────────────────────────────────────────── */

function labelFor(selectId) {
  const sel = document.getElementById(selectId);
  return sel ? sel.options[sel.selectedIndex].text : "—";
}

function buildUserSummary(data) {
  return `
    <strong>Conditions submitted:</strong>
    <ul style="margin-top:.4rem;padding-left:1.1rem;font-size:.82rem;line-height:1.8">
      <li>Fire size: <strong>${data.fire_size} ha</strong></li>
      <li>Spread rate: <strong>${labelFor("spreadRate")}</strong></li>
      <li>Wind speed: <strong>${data.wind_speed} km/h</strong></li>
      <li>Temperature: <strong>${data.temperature} °C</strong></li>
      <li>Relative humidity: <strong>${data.humidity}%</strong></li>
      <li>Ignition cause: <strong>${labelFor("ignitionCause")}</strong></li>
      <li>Forest region: <strong>${labelFor("forestRegion")}</strong></li>
    </ul>
  `;
}

/* ─── Result card builder ────────────────────────────────────────────────── */

function buildResultCard(result) {
  const { risk_score, risk_level, factors } = result;
  const color = riskColor(risk_level);

  const card = document.createElement("div");
  card.className = "result-card";

  // ── Score row (gauge + badge + label)
  const scoreRow = document.createElement("div");
  scoreRow.className = "risk-score-row";

  const gauge = buildGauge(risk_score, color);
  scoreRow.appendChild(gauge);

  const labelWrap = document.createElement("div");
  labelWrap.innerHTML = `
    <span class="risk-level-badge badge-${risk_level}">${risk_level}</span>
    <div class="risk-label-text" style="color:${color}">
      Escalation Risk
    </div>
    <div class="risk-sublabel">
      ${riskDescription(risk_level, risk_score)}
    </div>
  `;
  scoreRow.appendChild(labelWrap);
  card.appendChild(scoreRow);

  // ── SHAP factor bars
  const title = document.createElement("p");
  title.className = "factors-title";
  title.textContent = "Key driving factors (SHAP contributions)";
  card.appendChild(title);

  const maxContrib = factors.length > 0 ? factors[0].contribution : 1;

  factors.forEach((f) => {
    card.appendChild(buildFactorBar(f, maxContrib));
  });

  animateFactorBars(card);

  // ── "Assess Another Fire" button
  const btn = document.createElement("button");
  btn.className = "reassess-btn";
  btn.innerHTML = "🔄 Assess Another Fire";
  btn.addEventListener("click", () => {
    document.getElementById("assessForm").reset();
    // Show the form panel on mobile if it was scrolled away
    document.querySelector(".form-section").scrollIntoView({ behavior: "smooth" });
  });
  card.appendChild(btn);

  return card;
}

/* ─── Form submission ────────────────────────────────────────────────────── */

document.getElementById("assessForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const form = e.target;
  const btn  = document.getElementById("submitBtn");

  // ── Validate
  let valid = true;
  form.querySelectorAll("input, select").forEach((el) => {
    el.classList.remove("invalid");
    if (!el.value && el.required) {
      el.classList.add("invalid");
      valid = false;
    }
  });
  if (!valid) return;

  // ── Gather values
  const data = {
    fire_size:      parseFloat(document.getElementById("fireSize").value),
    spread_rate:    document.getElementById("spreadRate").value,
    wind_speed:     parseFloat(document.getElementById("windSpeed").value),
    temperature:    parseFloat(document.getElementById("temperature").value),
    humidity:       parseFloat(document.getElementById("humidity").value),
    ignition_cause: document.getElementById("ignitionCause").value,
    forest_region:  document.getElementById("forestRegion").value,
  };

  // Reject NaN values that could slip past the basic required check
  const numericIds = ["fireSize", "windSpeed", "temperature", "humidity"];
  numericIds.forEach((id) => {
    const el = document.getElementById(id);
    if (isNaN(parseFloat(el.value))) {
      el.classList.add("invalid");
      valid = false;
    }
  });
  if (!valid) return;

  // ── Post user summary to chat
  appendBubble("user", buildUserSummary(data));

  // ── Show spinner
  const spinnerBubble = appendBubble(
    "system",
    `<div class="thinking">
       <span></span><span></span><span></span>
     </div>`
  );

  btn.disabled = true;

  try {
    const resp = await fetch("/api/assess", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
    });

    const result = await resp.json();

    removeBubble(spinnerBubble);

    if (!resp.ok) {
      appendBubble(
        "system",
        `<p>⚠️ Error: ${result.error ?? "Unknown error. Please try again."}</p>`
      );
      return;
    }

    appendBubble("system", buildResultCard(result));

  } catch (err) {
    removeBubble(spinnerBubble);
    appendBubble(
      "system",
      `<p>⚠️ Network error: ${err.message}. Please check your connection and try again.</p>`
    );
  } finally {
    btn.disabled = false;
  }
});

/* ─── Clear invalid state on input ──────────────────────────────────────── */
document.querySelectorAll("#assessForm input, #assessForm select").forEach((el) => {
  el.addEventListener("input", () => el.classList.remove("invalid"));
  el.addEventListener("change", () => el.classList.remove("invalid"));
});

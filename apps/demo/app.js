const metricLabels = {
  rar: "Residual Attack Ratio (aggregate evaluator metric)",
  hidden_source_recall: "Hidden-source recall (aggregate evaluator metric)",
  precision: "Extension precision (aggregate evaluator metric)",
  benign_weight_loss: "Benign weight loss (aggregate evaluator metric)",
  utility_retention: "Utility retention (aggregate evaluator metric)",
  control_item_drift: "Control-item drift (aggregate evaluator metric)",
  ndcg_at_20: "NDCG@20 (aggregate evaluator metric)",
  recall_at_20: "Recall@20 (aggregate evaluator metric)",
  topk_jaccard_reference: "Top-K Jaccard vs. clean reference (aggregate)",
  topk_jaccard_retrain: "Top-K Jaccard vs. retraining (aggregate)",
  cost_ratio: "Cost ratio vs. retraining (aggregate)",
  membership_auc_gap: "Membership AUC gap (aggregate evaluator metric)",
};

const scenarioSelect = document.querySelector("#scenario-select");
const methodSelect = document.querySelector("#method-select");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
let bundle = null;

function setMessage(title, detail, isError = false) {
  message.hidden = false;
  message.classList.toggle("error", isError);
  message.querySelector(".message-icon").textContent = isError ? "!" : "i";
  message.querySelector("h2").textContent = title;
  message.querySelector("p").textContent = detail;
  results.hidden = true;
}

function setText(selector, text) {
  document.querySelector(selector).textContent = text;
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function formatRate(value) {
  return Number.isFinite(value) ? `${formatNumber(value)} (${formatNumber(value * 100)}%)` : "Not recorded";
}

function formatNumber(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}

function rankLabel(rank, topK) {
  if (rank === null) return `Not present in Top-${topK} (full rank unavailable)`;
  if (rank > topK) return `${rank} (outside Top-${topK})`;
  return `#${rank}`;
}

function renderList(selector, entries) {
  const list = document.querySelector(selector);
  list.replaceChildren();
  if (entries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-list";
    empty.textContent = "No items were saved for this list.";
    list.append(empty);
    return;
  }
  for (const entry of entries) {
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.className = "item-label";
    label.textContent = entry.label;
    row.append(label);
    if (entry.is_target) {
      const badge = document.createElement("span");
      badge.className = "target-badge";
      badge.textContent = "Target";
      row.append(badge);
    }
    list.append(row);
  }
}

function renderMetrics(method) {
  const container = document.querySelector("#metrics");
  container.replaceChildren();
  const entries = Object.entries(method.aggregate_metrics);
  document.querySelector("#no-metrics").hidden = entries.length > 0;
  for (const [key, value] of entries) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const label = document.createElement("span");
    label.textContent = metricLabels[key] || key;
    const number = document.createElement("strong");
    number.textContent = formatNumber(value);
    card.append(label, number);
    container.append(card);
  }
  const elapsedCard = document.createElement("div");
  elapsedCard.className = "metric-card elapsed";
  const elapsedLabel = document.createElement("span");
  elapsedLabel.textContent = "Elapsed time for selected method";
  const elapsedValue = document.createElement("strong");
  elapsedValue.textContent = `${formatNumber(method.elapsed_seconds)} s`;
  elapsedCard.append(elapsedLabel, elapsedValue);
  container.append(elapsedCard);
}

function renderSelection() {
  if (!bundle) return;
  const scenario = bundle.scenarios[Number(scenarioSelect.value)];
  if (!scenario) return;
  const method = scenario.methods[Number(methodSelect.value)];
  if (!method) return;

  message.hidden = true;
  results.hidden = false;
  setText("#scenario-title", scenario.label);
  setText("#scenario-summary", "Completed offline result · Deployable tier");
  setText("#dataset", scenario.setting.dataset);
  setText("#construction", scenario.setting.hypergraph_construction);
  setText("#attack", scenario.setting.attack);
  setText(
    "#identification",
    `${formatRate(scenario.setting.identification.rate)} rate · ` +
      `${formatRate(scenario.setting.identification.false_positive_share)} false-positive share · ` +
      `${scenario.setting.identification.strategy}`,
  );
  setText(
    "#seeds",
    Object.entries(scenario.setting.seeds).map(([name, value]) => `${name}: ${value}`).join(" · "),
  );
  setText("#target-name", scenario.target_item_label);
  setText("#user-name", scenario.example.user_label);
  setText("#method-heading", method.label);
  setText("#rank-before", rankLabel(scenario.example.target_rank_before, scenario.example.top_k));
  setText("#rank-after", rankLabel(method.target_rank_after, scenario.example.top_k));
  renderList("#list-before", scenario.example.before);
  renderList("#list-after", method.after);
  renderMetrics(method);
}

function populateMethods(scenarioIndex) {
  const scenario = bundle.scenarios[Number(scenarioIndex)];
  methodSelect.replaceChildren();
  if (!scenario) {
    methodSelect.disabled = true;
    return;
  }
  scenario.methods.forEach((method, index) => addOption(methodSelect, String(index), method.label));
  methodSelect.disabled = false;
  methodSelect.value = "0";
}

async function loadBundle() {
  try {
    const response = await fetch("/api/bundle", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The local demo bundle could not be loaded.");
    if (data.schema_version !== 1 || !Array.isArray(data.scenarios)) {
      throw new Error("The local demo bundle uses an unsupported data contract.");
    }
    bundle = data;
    if (bundle.scenarios.length === 0) {
      scenarioSelect.replaceChildren();
      addOption(scenarioSelect, "", "No saved scenarios");
      scenarioSelect.disabled = true;
      methodSelect.replaceChildren();
      addOption(methodSelect, "", "No saved methods");
      methodSelect.disabled = true;
      setMessage(
        "No exported experiment results yet",
        "Run the offline experiment pipeline and export a completed-run file to populate this read-only viewer. No recommendations or metrics have been fabricated.",
      );
      return;
    }
    scenarioSelect.replaceChildren();
    bundle.scenarios.forEach((scenario, index) => addOption(scenarioSelect, String(index), scenario.label));
    scenarioSelect.disabled = false;
    populateMethods("0");
    renderSelection();
  } catch (error) {
    scenarioSelect.disabled = true;
    methodSelect.disabled = true;
    const detail = error instanceof TypeError
      ? "Could not contact the local demo server. Start it from the repository root and reload this page."
      : error.message;
    setMessage("Saved results could not be loaded", detail, true);
  }
}

scenarioSelect.addEventListener("change", () => {
  populateMethods(scenarioSelect.value);
  renderSelection();
});
methodSelect.addEventListener("change", renderSelection);
loadBundle();

"use strict";

const METRIC_LABELS = {
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

const TRACE_LABELS = {
  pipeline: {
    dataset_preprocessing: "Dataset preprocessing",
    interaction_matrix: "User–item interaction matrix",
    hypergraph_construction: "Hypergraph construction",
    recommendation_baseline: "Recommendation baseline",
  },
  poisoning: {
    attack_configuration: "Attack configuration",
    poisoning_simulation: "Poisoning simulation",
    identification: "Source identification",
    unlearning: "Unlearning stage",
    evaluation: "Post-unlearning evaluation",
  },
};

const PAGE_SIZE = 12;
const state = {
  view: "shop",
  bundle: { schema_version: 2, scenarios: [] },
  bundleError: "",
  catalog: [],
  catalogError: "",
  cart: new Map(),
  category: "",
  query: "",
  page: 0,
  traceIndexes: { pipeline: -1, poisoning: -1 },
  traceTimers: { pipeline: null, poisoning: null },
};

const scenarioSelect = document.querySelector("#scenario-select");
const methodSelect = document.querySelector("#method-select");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setText(selector, text) {
  document.querySelector(selector).textContent = String(text);
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function formatNumber(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "Price unavailable";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 }).format(value);
}

function rankLabel(rank, topK) {
  if (rank === null) return `Not present in Top-${topK} (full rank unavailable)`;
  if (rank > topK) return `${rank} (outside Top-${topK})`;
  return `#${rank}`;
}

function selectedScenario() {
  if (!state.bundle.scenarios.length || scenarioSelect.disabled) return null;
  return state.bundle.scenarios[Number(scenarioSelect.value)] || null;
}

function selectedMethod(scenario = selectedScenario()) {
  if (!scenario || !scenario.methods.length || methodSelect.disabled) return null;
  return scenario.methods[Number(methodSelect.value)] || null;
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".step-link").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "step" : "false");
  });
  document.querySelectorAll(".view-section").forEach((section) => {
    const active = section.id === `view-${view}`;
    section.hidden = !active;
    section.classList.toggle("active", active);
  });
  document.querySelector("#run-controls").hidden = view === "shop";
  stopPlayback("pipeline");
  stopPlayback("poisoning");
  if (view === "pipeline") renderTrace("pipeline");
  if (view === "poisoning") renderTrace("poisoning");
  if (view === "results") renderResults();
}

function renderItems(listSelector, entries) {
  const list = document.querySelector(listSelector);
  list.replaceChildren();
  if (!entries.length) {
    list.append(element("li", "empty-list", "No items were saved for this list."));
    return;
  }
  entries.forEach((entry) => {
    const row = element("li", "recommendation-row");
    row.append(element("span", "item-label", entry.label));
    if (entry.is_target) row.append(element("span", "target-badge", "Target"));
    list.append(row);
  });
}

function renderMetrics(method) {
  const container = document.querySelector("#metrics");
  container.replaceChildren();
  const metrics = Object.entries(method.aggregate_metrics);
  document.querySelector("#no-metrics").hidden = metrics.length > 0;
  metrics.forEach(([key, value]) => {
    const card = element("div", "metric-card");
    card.append(element("span", "", METRIC_LABELS[key] || key), element("strong", "", formatNumber(value)));
    container.append(card);
  });
  const elapsedCard = element("div", "metric-card elapsed");
  elapsedCard.append(element("span", "", "Elapsed time for selected method"), element("strong", "", `${formatNumber(method.elapsed_seconds)} s`));
  container.append(elapsedCard);
}

function renderResults() {
  const scenario = selectedScenario();
  const method = selectedMethod(scenario);
  const empty = document.querySelector("#results-empty");
  const content = document.querySelector("#results-content");
  if (!scenario || !method) {
    content.hidden = true;
    empty.hidden = false;
    empty.querySelector("strong").textContent = state.bundleError ? "Saved results could not be loaded" : "No exported experiment results yet";
    empty.querySelector("p").textContent = state.bundleError || "Complete the offline research pipeline and export a reviewed completed-run file. This viewer does not invent recommendations, rankings, traces, or metrics.";
    return;
  }
  empty.hidden = true;
  content.hidden = false;
  setText("#scenario-title", scenario.label);
  setText("#scenario-summary", "Precomputed completed offline result · deployable tier");
  setText("#dataset", scenario.setting.dataset);
  setText("#construction", scenario.setting.hypergraph_construction);
  setText("#attack", scenario.setting.attack);
  setText(
    "#identification",
    `${formatNumber(scenario.setting.identification.rate * 100)}% rate · ` +
      `${formatNumber(scenario.setting.identification.false_positive_share * 100)}% false-positive share · ` +
      scenario.setting.identification.strategy,
  );
  setText("#seeds", Object.entries(scenario.setting.seeds).map(([name, value]) => `${name}: ${value}`).join(" · "));
  setText("#target-name", scenario.target_item_label);
  setText("#user-name", scenario.example.user_label);
  setText("#method-heading", method.label);
  setText("#rank-before", rankLabel(scenario.example.target_rank_before, scenario.example.top_k));
  setText("#rank-after", rankLabel(method.target_rank_after, scenario.example.top_k));
  renderItems("#list-before", scenario.example.before);
  renderItems("#list-after", method.after);
  renderMetrics(method);
}

function renderSettingStrip(selector, scenario) {
  const strip = document.querySelector(selector);
  strip.replaceChildren();
  strip.hidden = !scenario;
  if (!scenario) return;
  const parts = [
    ["DATASET", scenario.setting.dataset],
    ["HYPERGRAPH", scenario.setting.hypergraph_construction],
    ["ATTACK", scenario.setting.attack],
    ["METHOD", selectedMethod(scenario)?.label || "Not selected"],
  ];
  parts.forEach(([label, value]) => {
    const cell = element("div", "setting-strip-cell");
    cell.append(element("small", "", label), element("strong", "", value));
    strip.append(cell);
  });
}

function eventDetail(event) {
  const detail = element("div", "trace-event-detail");
  const countEntries = Object.entries(event.counts);
  if (countEntries.length) {
    const list = element("div", "trace-counts");
    countEntries.forEach(([key, value]) => {
      const chip = element("span", "count-chip");
      chip.append(element("strong", "", formatNumber(value)), document.createTextNode(` ${key.replaceAll("_", " ")}`));
      list.append(chip);
    });
    detail.append(list);
  } else {
    detail.append(element("span", "muted", "No aggregate counts recorded"));
  }
  detail.append(element("span", "trace-elapsed", event.elapsed_seconds === null ? "Elapsed time not recorded" : `${formatNumber(event.elapsed_seconds)} s saved`));
  return detail;
}

function stopPlayback(kind) {
  if (state.traceTimers[kind] !== null) {
    window.clearInterval(state.traceTimers[kind]);
    state.traceTimers[kind] = null;
  }
}

function renderTrace(kind) {
  const scenario = selectedScenario();
  const timeline = document.querySelector(`#${kind}-timeline`);
  const empty = document.querySelector(`#${kind}-empty`);
  const current = document.querySelector(`#${kind}-current`);
  const events = scenario ? scenario[`${kind}_trace`] : [];
  timeline.replaceChildren();
  renderSettingStrip(`#${kind}-setting`, scenario);
  empty.hidden = Array.isArray(events) && events.length > 0;
  if (!scenario) {
    empty.querySelector("strong").textContent = state.bundleError ? "Saved run could not be loaded" : kind === "pipeline" ? "No pipeline trace in this run" : "No attack trace in this run";
    empty.querySelector("p").textContent = state.bundleError || "Select an exported completed run with saved trace events. Nothing is generated by this page.";
  }
  const index = state.traceIndexes[kind];
  if (Array.isArray(events)) {
    events.forEach((event, eventIndex) => {
      const row = element("li", "trace-event");
      row.classList.toggle("complete", eventIndex < index);
      row.classList.toggle("current", eventIndex === index);
      const node = element("span", "trace-node", eventIndex < index ? "✓" : String(eventIndex + 1).padStart(2, "0"));
      const body = element("div", "trace-event-body");
      const heading = element("div", "trace-event-heading");
      heading.append(element("strong", "", TRACE_LABELS[kind][event.step] || "Saved experiment stage"));
      const status = eventIndex < index ? "PLAYED" : eventIndex === index ? "NOW PLAYING" : "SAVED";
      heading.append(element("span", "trace-event-status", status));
      body.append(heading, eventDetail(event));
      row.append(node, body);
      timeline.append(row);
    });
  }
  current.replaceChildren();
  current.hidden = !Array.isArray(events) || events.length === 0;
  if (!current.hidden) {
    const message = index < 0 ? "Ready · play or advance through saved events" : index >= events.length - 1 ? "Playback complete · all events came from the saved run" : `Showing saved event ${index + 1} of ${events.length}`;
    current.append(element("span", "status-dot"), element("span", "", message));
  }
  document.querySelectorAll(`[data-trace="${kind}"]`).forEach((button) => {
    button.disabled = !Array.isArray(events) || events.length === 0;
    if (button.dataset.traceAction === "play") {
      button.textContent = state.traceTimers[kind] === null ? "▶ Play saved trace" : "Ⅱ Pause playback";
    }
  });
}

function advanceTrace(kind) {
  const scenario = selectedScenario();
  const events = scenario ? scenario[`${kind}_trace`] : [];
  if (!Array.isArray(events) || !events.length) return;
  state.traceIndexes[kind] = Math.min(state.traceIndexes[kind] + 1, events.length - 1);
  renderTrace(kind);
}

function togglePlayback(kind) {
  if (state.traceTimers[kind] !== null) {
    stopPlayback(kind);
    renderTrace(kind);
    return;
  }
  const scenario = selectedScenario();
  const events = scenario ? scenario[`${kind}_trace`] : [];
  if (!Array.isArray(events) || !events.length) return;
  if (state.traceIndexes[kind] >= events.length - 1) state.traceIndexes[kind] = -1;
  state.traceTimers[kind] = window.setInterval(() => {
    if (state.traceIndexes[kind] + 1 >= events.length) {
      stopPlayback(kind);
      renderTrace(kind);
      return;
    }
    state.traceIndexes[kind] += 1;
    if (state.traceIndexes[kind] >= events.length - 1) {
      stopPlayback(kind);
      renderTrace(kind);
      return;
    }
    renderTrace(kind);
  }, 1100);
  renderTrace(kind);
}

function handleTraceAction(event) {
  const button = event.target.closest("button[data-trace-action]");
  if (!button) return;
  const kind = button.dataset.trace;
  const action = button.dataset.traceAction;
  if (action === "reset") {
    stopPlayback(kind);
    state.traceIndexes[kind] = -1;
    renderTrace(kind);
  } else if (action === "step") {
    stopPlayback(kind);
    advanceTrace(kind);
  } else if (action === "play") {
    togglePlayback(kind);
  }
}

function filteredProducts() {
  const query = state.query.trim().toLocaleLowerCase();
  return state.catalog.filter((product) => {
    const matchesCategory = !state.category || product.category.split(" / ")[0] === state.category;
    const matchesQuery = !query || `${product.name} ${product.category}`.toLocaleLowerCase().includes(query);
    return matchesCategory && matchesQuery;
  });
}

function renderProducts() {
  const grid = document.querySelector("#product-grid");
  const message = document.querySelector("#catalog-message");
  const products = filteredProducts();
  const pageCount = Math.max(1, Math.ceil(products.length / PAGE_SIZE));
  state.page = Math.min(state.page, pageCount - 1);
  const start = state.page * PAGE_SIZE;
  const visible = products.slice(start, start + PAGE_SIZE);
  grid.replaceChildren();
  setText("#catalog-count", `${products.length.toLocaleString()} items`);
  setText("#catalog-page-label", products.length ? `Showing ${start + 1}–${Math.min(start + PAGE_SIZE, products.length)} of ${products.length.toLocaleString()}` : "No products to show");
  document.querySelector("#products-prev").disabled = state.page <= 0;
  document.querySelector("#products-next").disabled = start + PAGE_SIZE >= products.length;
  if (!state.catalog.length) {
    message.hidden = false;
    message.classList.toggle("error", Boolean(state.catalogError));
    message.textContent = state.catalogError || "No local catalog found. Import the supplied sample archive to populate the storefront.";
    return;
  }
  message.hidden = products.length > 0;
  message.classList.remove("error");
  message.textContent = "No products match those filters.";

  visible.forEach((product) => {
    const card = element("article", "product-card");
    const visual = element("div", "product-visual");
    visual.append(element("span", "product-monogram", product.category.split(" / ").at(-1).slice(0, 2).toUpperCase()));
    visual.append(element("span", "product-category", product.category.split(" / ").at(-1)));
    const info = element("div", "product-info");
    info.append(element("p", "product-category-text", product.category.split(" / ").slice(0, 2).join(" / ")));
    info.append(element("h4", "", product.name));
    const ratingText = product.rating === null ? "Rating not recorded" : `★ ${formatNumber(product.rating)} · ${product.rating_count === null ? "count not recorded" : `${product.rating_count.toLocaleString()} ratings`}`;
    info.append(element("span", "product-rating", ratingText));
    const priceLine = element("div", "product-price-line");
    priceLine.append(element("strong", "product-price", formatMoney(product.price)));
    if (product.list_price !== null && product.list_price > (product.price || 0)) priceLine.append(element("s", "list-price", formatMoney(product.list_price)));
    info.append(priceLine);
    const add = element("button", "add-button", product.price === null ? "Price unavailable" : "Add to cart +");
    add.type = "button";
    add.dataset.addProduct = product.id;
    add.disabled = product.price === null;
    info.append(add);
    card.append(visual, info);
    grid.append(card);
  });
}

function renderCart() {
  const cartList = document.querySelector("#cart-list");
  cartList.replaceChildren();
  let totalQuantity = 0;
  let subtotal = 0;
  state.cart.forEach(({ product, quantity }) => {
    totalQuantity += quantity;
    if (product.price !== null) subtotal += product.price * quantity;
    const row = element("div", "cart-row");
    const monogram = element("span", "cart-monogram", product.name.slice(0, 1).toUpperCase());
    const body = element("div", "cart-row-main");
    body.append(element("strong", "cart-product-name", product.name), element("span", "cart-product-price", formatMoney(product.price)));
    const controls = element("div", "quantity-controls");
    const decrement = element("button", "quantity-button", "−");
    decrement.type = "button";
    decrement.setAttribute("aria-label", `Decrease quantity of ${product.name}`);
    decrement.dataset.cartAction = "decrement";
    decrement.dataset.productId = product.id;
    controls.append(decrement, element("span", "quantity", String(quantity)));
    const increment = element("button", "quantity-button", "+");
    increment.type = "button";
    increment.setAttribute("aria-label", `Increase quantity of ${product.name}`);
    increment.dataset.cartAction = "increment";
    increment.dataset.productId = product.id;
    controls.append(increment);
    const remove = element("button", "remove-button", "Remove");
    remove.type = "button";
    remove.dataset.cartAction = "remove";
    remove.dataset.productId = product.id;
    body.append(controls, remove);
    row.append(monogram, body);
    cartList.append(row);
  });
  if (!state.cart.size) cartList.append(element("p", "cart-empty", "Your cart is empty. Add an item to see it here."));
  setText("#cart-count", String(totalQuantity));
  setText("#cart-items-label", `(${totalQuantity})`);
  setText("#cart-subtotal", formatMoney(subtotal));
  document.querySelector("#checkout-button").disabled = totalQuantity === 0;
}

function updateCategories() {
  const select = document.querySelector("#category-filter");
  select.replaceChildren();
  addOption(select, "", "All categories");
  const categories = [...new Set(state.catalog.map((product) => product.category.split(" / ")[0]))].sort((a, b) => a.localeCompare(b));
  categories.forEach((category) => addOption(select, category, category));
}

async function loadCatalog() {
  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The local catalog could not be loaded.");
    if (data.schema_version !== 1 || !Array.isArray(data.products)) throw new Error("The local catalog uses an unsupported data contract.");
    state.catalog = data.products;
    updateCategories();
  } catch (error) {
    state.catalogError = error instanceof TypeError ? "Could not contact the local demo server. Start it and reload this page." : error.message;
  }
  renderProducts();
}

async function loadBundle() {
  try {
    const response = await fetch("/api/bundle", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The saved experiment bundle could not be loaded.");
    if (data.schema_version !== 2 || !Array.isArray(data.scenarios)) throw new Error("The saved experiment bundle uses an unsupported data contract.");
    state.bundle = data;
  } catch (error) {
    state.bundleError = error instanceof TypeError ? "Could not contact the local demo server. Start it and reload this page." : error.message;
  }
  scenarioSelect.replaceChildren();
  if (!state.bundle.scenarios.length) {
    addOption(scenarioSelect, "", "No saved scenarios");
    scenarioSelect.disabled = true;
    methodSelect.replaceChildren();
    addOption(methodSelect, "", "No saved methods");
    methodSelect.disabled = true;
  } else {
    state.bundle.scenarios.forEach((scenario, index) => addOption(scenarioSelect, String(index), scenario.label));
    scenarioSelect.disabled = false;
    populateMethods();
  }
  renderTrace("pipeline");
  renderTrace("poisoning");
  renderResults();
}

function populateMethods() {
  const scenario = state.bundle.scenarios[Number(scenarioSelect.value)];
  methodSelect.replaceChildren();
  if (!scenario) {
    methodSelect.disabled = true;
    addOption(methodSelect, "", "No saved methods");
    return;
  }
  scenario.methods.forEach((method, index) => addOption(methodSelect, String(index), method.label));
  methodSelect.disabled = false;
  methodSelect.value = "0";
}

document.querySelectorAll(".step-link").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  setView("shop");
});
document.querySelectorAll(".trace-toolbar").forEach((toolbar) => toolbar.addEventListener("click", handleTraceAction));
scenarioSelect.addEventListener("change", () => {
  populateMethods();
  state.traceIndexes = { pipeline: -1, poisoning: -1 };
  renderTrace("pipeline");
  renderTrace("poisoning");
  renderResults();
});
methodSelect.addEventListener("change", () => {
  renderTrace("pipeline");
  renderTrace("poisoning");
  renderResults();
});

document.querySelector("#product-search").addEventListener("input", (event) => {
  state.query = event.target.value;
  state.page = 0;
  renderProducts();
});
document.querySelector("#category-filter").addEventListener("change", (event) => {
  state.category = event.target.value;
  state.page = 0;
  renderProducts();
});
document.querySelector("#products-prev").addEventListener("click", () => {
  state.page = Math.max(0, state.page - 1);
  renderProducts();
});
document.querySelector("#products-next").addEventListener("click", () => {
  state.page += 1;
  renderProducts();
});
document.querySelector("#product-grid").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-add-product]");
  if (!button) return;
  const product = state.catalog.find((item) => item.id === button.dataset.addProduct);
  if (!product || product.price === null) return;
  const current = state.cart.get(product.id);
  state.cart.set(product.id, { product, quantity: Math.min((current?.quantity || 0) + 1, 99) });
  document.querySelector("#checkout-note").textContent = "";
  renderCart();
});
document.querySelector("#cart-list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-cart-action]");
  if (!button) return;
  const id = button.dataset.productId;
  const current = state.cart.get(id);
  if (!current) return;
  if (button.dataset.cartAction === "remove" || (button.dataset.cartAction === "decrement" && current.quantity <= 1)) state.cart.delete(id);
  else if (button.dataset.cartAction === "decrement") state.cart.set(id, { ...current, quantity: current.quantity - 1 });
  else if (button.dataset.cartAction === "increment") state.cart.set(id, { ...current, quantity: Math.min(current.quantity + 1, 99) });
  renderCart();
});
document.querySelector("#checkout-button").addEventListener("click", () => {
  document.querySelector("#checkout-note").textContent = "Demo only — no order or payment was created.";
});

renderCart();
loadCatalog();
loadBundle();

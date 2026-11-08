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

const PAGE_SIZE = 18;
const state = {
  view: "shop",
  bundle: { schema_version: 2, scenarios: [] },
  bundleError: "",
  movies: [],
  catalogError: "",
  movieDataset: null,
  watchlist: new Map(),
  watched: new Set(),
  category: "",
  query: "",
  movieSort: "popular",
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

function filteredMovies() {
  const query = state.query.trim().toLocaleLowerCase();
  const movies = state.movies.filter((movie) => {
    const matchesCategory = !state.category || movie.genres.includes(state.category);
    const matchesQuery = !query || `${movie.title} ${movie.genres.join(" ")} ${movie.year || ""}`.toLocaleLowerCase().includes(query);
    return matchesCategory && matchesQuery;
  });
  if (state.movieSort === "newest") {
    movies.sort((left, right) => (right.year || 0) - (left.year || 0) || right.popularity - left.popularity || left.id - right.id);
  } else if (state.movieSort === "title") {
    movies.sort((left, right) => left.title.localeCompare(right.title) || left.id - right.id);
  } else {
    movies.sort((left, right) => right.popularity - left.popularity || left.id - right.id);
  }
  return movies;
}

function renderMovies() {
  const grid = document.querySelector("#product-grid");
  const message = document.querySelector("#catalog-message");
  const movies = filteredMovies();
  const pageCount = Math.max(1, Math.ceil(movies.length / PAGE_SIZE));
  state.page = Math.min(state.page, pageCount - 1);
  const start = state.page * PAGE_SIZE;
  const visible = movies.slice(start, start + PAGE_SIZE);
  grid.replaceChildren();
  setText("#catalog-count", `${movies.length.toLocaleString()} titles`);
  setText("#catalog-page-label", movies.length ? `Showing ${start + 1}–${Math.min(start + PAGE_SIZE, movies.length)} of ${movies.length.toLocaleString()}` : "No titles to show");
  document.querySelector("#products-prev").disabled = state.page <= 0;
  document.querySelector("#products-next").disabled = start + PAGE_SIZE >= movies.length;
  if (!state.movies.length) {
    message.hidden = false;
    message.classList.toggle("error", Boolean(state.catalogError));
    message.textContent = state.catalogError || "No MovieLens catalog is available yet. Generate results/movie-catalog.json from an approved MovieLens archive.";
    return;
  }
  message.hidden = movies.length > 0;
  message.classList.remove("error");
  message.textContent = "No films match those filters.";

  visible.forEach((movie, index) => {
    const card = element("article", "movie-card");
    const visual = element("div", `poster-art poster-tone-${(start + index) % 6}`);
    visual.append(element("span", "poster-mark", movie.genres[0] || "CINEMA"));
    visual.append(element("strong", "poster-title", movie.title.replace(/\s*\(\d{4}\)\s*$/, "")));
    visual.append(element("span", "poster-year", movie.year || "MOVIELENS"));
    const info = element("div", "movie-info");
    info.append(element("h4", "", movie.title));
    info.append(element("p", "movie-meta", `${movie.year || "Year not listed"} · ${movie.genres.join(" / ") || "Genre not listed"}`));
    info.append(element("span", "movie-popularity", `Clean training popularity · ${movie.popularity.toLocaleString()}`));
    const actions = element("div", "movie-actions");
    const watch = element("button", "add-button", state.watchlist.has(movie.id) ? "In watchlist ✓" : "Add to watchlist +");
    watch.type = "button";
    watch.dataset.movieAction = "watchlist";
    watch.dataset.movieId = String(movie.id);
    const seen = element("button", "seen-button", state.watched.has(movie.id) ? "Watched ✓" : "Mark watched");
    seen.type = "button";
    seen.dataset.movieAction = "watched";
    seen.dataset.movieId = String(movie.id);
    actions.append(watch, seen);
    info.append(actions);
    card.append(visual, info);
    grid.append(card);
  });
}

function renderWatchlist() {
  const cartList = document.querySelector("#cart-list");
  cartList.replaceChildren();
  state.watchlist.forEach((movie) => {
    const row = element("div", "cart-row");
    const monogram = element("span", "cart-monogram", movie.title.slice(0, 1).toUpperCase());
    const body = element("div", "cart-row-main");
    body.append(element("strong", "cart-product-name", movie.title), element("span", "cart-product-price", movie.year || "Film"));
    const remove = element("button", "remove-button", "Remove");
    remove.type = "button";
    remove.dataset.movieAction = "remove-watchlist";
    remove.dataset.movieId = String(movie.id);
    body.append(remove);
    row.append(monogram, body);
    cartList.append(row);
  });
  if (!state.watchlist.size) cartList.append(element("p", "cart-empty", "Your watchlist is empty. Add a film to save it here."));
  setText("#cart-count", String(state.watchlist.size));
  setText("#cart-items-label", `(${state.watchlist.size})`);
}

function updateCategories() {
  const select = document.querySelector("#category-filter");
  select.replaceChildren();
  addOption(select, "", "All genres");
  const genres = [...new Set(state.movies.flatMap((movie) => movie.genres))].sort((a, b) => a.localeCompare(b));
  genres.forEach((genre) => addOption(select, genre, genre));
}

async function loadMovies() {
  try {
    const response = await fetch("/api/movies", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The MovieLens catalog could not be loaded.");
    if (data.schema_version !== 1 || !Array.isArray(data.movies)) throw new Error("The MovieLens catalog uses an unsupported data contract.");
    state.movies = data.movies;
    state.movieDataset = data.dataset ? `${data.dataset} · ${data.dataset_role}` : null;
    updateCategories();
    if (state.movieDataset) {
      setText("#movie-source-note", `${state.movieDataset} · Popularity is counted from clean training interactions before attack. MovieLens supplies titles and genres, not poster images.`);
    }
  } catch (error) {
    state.catalogError = error instanceof TypeError ? "Could not contact the local demo server. Start it and reload this page." : error.message;
  }
  renderMovies();
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
  renderMovies();
});
document.querySelector("#category-filter").addEventListener("change", (event) => {
  state.category = event.target.value;
  state.page = 0;
  renderMovies();
});
document.querySelector("#products-prev").addEventListener("click", () => {
  state.page = Math.max(0, state.page - 1);
  renderMovies();
});
document.querySelector("#products-next").addEventListener("click", () => {
  state.page += 1;
  renderMovies();
});
document.querySelector("#product-grid").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-movie-action]");
  if (!button) return;
  const movie = state.movies.find((item) => item.id === Number(button.dataset.movieId));
  if (!movie) return;
  if (button.dataset.movieAction === "watchlist") {
    if (state.watchlist.has(movie.id)) state.watchlist.delete(movie.id);
    else state.watchlist.set(movie.id, movie);
    renderWatchlist();
  } else if (button.dataset.movieAction === "watched") {
    if (state.watched.has(movie.id)) state.watched.delete(movie.id);
    else state.watched.add(movie.id);
  }
  renderMovies();
});
document.querySelector("#cart-list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-movie-action='remove-watchlist']");
  if (!button) return;
  state.watchlist.delete(Number(button.dataset.movieId));
  renderWatchlist();
  renderMovies();
});
document.querySelectorAll("[data-movie-sort]").forEach((button) => button.addEventListener("click", () => {
  state.movieSort = button.dataset.movieSort;
  state.page = 0;
  document.querySelectorAll("[data-movie-sort]").forEach((option) => option.classList.toggle("active", option === button));
  renderMovies();
}));

renderWatchlist();
loadMovies();
loadBundle();

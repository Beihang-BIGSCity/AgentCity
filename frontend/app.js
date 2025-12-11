const API_BASE = "../api";

const TRANSLATIONS = {
  en: {
    "lang.toggle": "English / 中文",
    "hero.title": "LibCity Agent",
    "hero.description":
      "Monitor the Claude-driven automation that scouts papers, ports models into LibCity, validates runs, and reports metrics—styled in the spirit of the LibCity landing page.",
    "hero.subtitle":
      "Autonomous pipelines cover literature searches, model migrations, and verification with full transparency.",
    "hero.cta.script": "View Pipeline Script",
    "hero.cta.guide": "Contributor Guide",
    "hero.badge": "Autonomous Workflow",
    "hero.badge_subtext": "2025 Editions: ICLR · ICML · NeurIPS · KDD · ICDE",
    "search.title": "Literature Search & Downloads",
    "search.description":
      "Enter keywords to surface the latest traffic-forecasting papers. Use the filters to scope the PaSa agent by year and conference.",
    "search.keywords.label": "Keywords",
    "search.keywords.placeholder": "e.g. graph transformer, PEMS, diffusion",
    "search.year.label": "Year",
    "search.year.this": "This Year",
    "search.year.last": "Last Year",
    "search.year.all": "All",
    "search.conference.label": "Conference (Customizable)",
    "search.conference.placeholder": "e.g. ICLR, ICML, NeurIPS",
    "search.actions.submit": "Search & Download",
    "search.actions.reset": "Reset",
    "migration.title": "Migration Batch Runner",
    "migration.description":
      "Select downloaded papers with code, export a migration plan, then run migration and tuning as two separate steps.",
    "migration.selection.title": "Select Papers",
    "migration.selection.description":
      "Only papers with a PDF (saved or link) and repository are listed here.",
    "migration.actions.stage_label": "Step 1: Migration",
    "migration.actions.start": "Start Migration",
    "migration.actions.export": "Export Plan",
    "migration.helper":
      "The exported JSON includes repository + model metadata and can be passed to <code>python claude_client.py</code>.",
    "tuning.actions.stage_label": "Step 2: Tuning",
    "tuning.actions.start": "Start Tuning",
    "tuning.helper":
      "Run tuning after migration finishes; it reuses the same selection to search hyperparameters.",
    "jobs.title": "Job Queue",
    "jobs.description": "Monitor literature-search, migration, and tuning jobs in real time.",
    "stage.title": "Stage Timeline",
    "stage.description":
      "Each block reflects a pipeline stage and highlights the latest transcript snippets.",
    "articles.title": "Archived Articles",
    "articles.description":
      "Browse every saved PDF and metadata entry captured by the catalog_article tool.",
    "footer.note":
      "Data sources refresh automatically after every run. Serve this folder via <code>python -m http.server</code> to load local JSON from <code>data/</code>.",
    "stages.empty": "No pipeline run recorded yet.",
    "stages.stage_id": "Stage ID: {{id}}",
    "stages.summary_fallback":
      "Stage completed without textual summary. Check raw transcript below.",
    "stages.transcript_label": "Transcript ({{count}} messages)",
    "stages.no_snippets": "No text snippets captured.",
    "articles.empty": "No articles cataloged yet. Run the pipeline to populate this grid.",
    "articles.datasets": "Datasets: {{value}}",
    "articles.datasets.unknown": "Unknown",
    "articles.model": "Model: {{value}}",
    "articles.model.pending": "Model: Pending catalog update",
    "articles.repo.link": "Repository",
    "articles.repo.missing": "No repository shared",
    "articles.pdf.local": "Open PDF",
    "articles.pdf.remote": "View PDF Online",
    "articles.pdf.missing": "PDF not saved yet",
    "articles.notes.missing": "No additional notes.",
    "labels.untitled": "Untitled",
    "labels.na": "N/A",
    "search.no_results": "No papers matched the filters. Try different keywords.",
    "search.no_summary": "No summary yet. Open the PDF below for more details.",
    "search.repo.available": "GitHub: repository provided",
    "search.repo.missing": "GitHub: not available yet",
    "search.pdf.download": "Download PDF",
    "search.pdf.online": "View PDF Online",
    "search.pdf.missing_btn": "PDF not saved",
    "search.repo.button": "Open Repository",
    "search.repo.button_missing": "No repository",
    "search.added": "Added",
    "search.add_to_migration": "Add to migration list",
    "search.add_unavailable": "Locked",
    "selection.empty":
      "No papers currently satisfy “PDF (saved or link) + GitHub available”. Run a literature job first.",
    "selection.model_missing": "Model not specified",
    "migration.feedback.none_selected":
      "Selected papers have no migration summaries yet. Run the migration stage and refresh.",
    "migration.feedback.none_available":
      "No migration summaries recorded. Execute a migration workflow to populate this feed.",
    "migration.feedback.model": "Model: {{value}}",
    "migration.feedback.open_repo": "Open Repository",
    "migration.feedback.no_repo": "No repository",
    "migration.feedback.view_pdf": "View PDF",
    "migration.feedback.view_summary": "View Summary",
    "migration.feedback.summary_missing": "Summary not generated",
    "migration.feedback.summary_empty":
      "No summary excerpt available yet. Re-run the migration stage for a fresh summary.",
    "migration.feedback.updated": "Last updated: {{value}}",
    "jobs.empty": "No jobs yet. Submit a literature search, migration, or tuning run to see status here.",
    "jobs.time.created": "Created",
    "jobs.time.started": "Started",
    "jobs.time.finished": "Finished",
    "jobs.meta.paper": "Paper: {{value}}",
    "jobs.meta.model": "Model: {{value}}",
    "jobs.meta.error": "Error: {{value}}",
    "jobs.status.running": "Running",
    "jobs.status.succeeded": "Completed",
    "jobs.status.failed": "Failed",
    "jobs.status.pending": "Queued",
    "alerts.loading": "Processing...",
    "alerts.submit_literature_loading": "Submitting...",
    "alerts.start_migration_loading": "Launching...",
    "alerts.start_tuning_loading": "Starting tuning...",
    "alerts.literature_failed": "Failed to submit literature search job",
    "alerts.migration_failed": "Failed to submit migration job",
    "alerts.tuning_failed": "Failed to submit tuning job",
    "alerts.no_selection": "Select at least one paper before starting a run.",
    "alerts.export_no_selection": "Select at least one paper before exporting the migration plan.",
    "alerts.migration_started_single": "Started 1 migration job.",
    "alerts.migration_started_multi": "Started {{count}} migration jobs.",
    "alerts.tuning_started_single": "Started 1 tuning job.",
    "alerts.tuning_started_multi": "Started {{count}} tuning jobs.",
    "errors.dashboard_load": "Failed to load dashboard data",
    "search.reset": "Reset",
    "jobs.form.paper_meta": "Paper:",
    "jobs.form.model_meta": "Model:",
    "jobs.form.error_label": "Error:",
    "alerts.submit_required": "Please enter at least one keyword.",
    "export.download": "Download migration plan JSON",
  },
  zh: {
    "lang.toggle": "中文 / English",
    "hero.title": "LibCity Agent",
    "hero.description":
      "监控 Claude 驱动的自动化流程，覆盖论文检索、模型迁移、指标验证与报告，整体风格与 LibCity 官网保持一致。",
    "hero.subtitle":
      "通过智能代理完成论文扫描、模型迁移、参数验证，流程可追溯、结果可复现。",
    "hero.cta.script": "查看运行脚本",
    "hero.cta.guide": "贡献者指南",
    "hero.badge": "全自动工作流",
    "hero.badge_subtext": "2025 重点会议：ICLR · ICML · NeurIPS · KDD · ICDE",
    "search.title": "文献搜索 & 下载",
    "search.description":
      "输入关键词即可筛选最新的交通预测论文，并可限制年份与会议；PaSa 代理将自动搜索并保存结果。",
    "search.keywords.label": "关键词",
    "search.keywords.placeholder": "例如：graph transformer, PEMS, diffusion",
    "search.year.label": "年份",
    "search.year.this": "今年",
    "search.year.last": "去年",
    "search.year.all": "全部",
    "search.conference.label": "会议（可自定义）",
    "search.conference.placeholder": "例如：ICLR, ICML, NeurIPS",
    "search.actions.submit": "搜索并下载",
    "search.actions.reset": "重置",
    "migration.title": "模型迁移批处理",
    "migration.description":
      "从已下载的论文中选择目标，导出迁移计划，迁移完成后再单独触发调参阶段。",
    "migration.selection.title": "选择论文",
    "migration.selection.description": "仅展示提供 PDF（本地或链接）且有 GitHub 仓库的论文。",
    "migration.actions.stage_label": "步骤 1：迁移",
    "migration.actions.start": "启动迁移",
    "migration.actions.export": "导出迁移计划",
    "migration.helper":
      "导出的 JSON 包含仓库与模型信息，可直接交由 <code>python claude_client.py</code> 执行。",
    "tuning.actions.stage_label": "步骤 2：调参",
    "tuning.actions.start": "启动调参",
    "tuning.helper": "迁移结束后再执行，复用同一批论文进行超参搜索。",
    "jobs.title": "任务队列",
    "jobs.description": "实时关注文献搜索、模型迁移与调参任务的状态。",
    "stage.title": "阶段时间线",
    "stage.description": "展示各阶段概览与最新对话片段，便于追踪自动化过程。",
    "articles.title": "论文归档",
    "articles.description": "查看 catalog_article 工具保存的全部 PDF 与元信息记录。",
    "footer.note":
      "每次运行结束后数据都会自动刷新。可通过 <code>python -m http.server</code> 直接预览 <code>data/</code> 中的 JSON。",
    "stages.empty": "当前暂无运行记录，执行一次自动化管线即可生成。",
    "stages.stage_id": "阶段 ID：{{id}}",
    "stages.summary_fallback": "该阶段未生成文本摘要，可展开下方对话记录查看。",
    "stages.transcript_label": "对话记录（{{count}} 条消息）",
    "stages.no_snippets": "暂无文本片段。",
    "articles.empty": "还没有论文被收录，先运行一次文献搜索吧。",
    "articles.datasets": "数据集：{{value}}",
    "articles.datasets.unknown": "未提供",
    "articles.model": "模型：{{value}}",
    "articles.model.pending": "模型：等待 catalog 更新",
    "articles.repo.link": "打开仓库",
    "articles.repo.missing": "尚未提供仓库链接",
    "articles.pdf.local": "打开本地 PDF",
    "articles.pdf.remote": "在线查看 PDF",
    "articles.pdf.missing": "PDF 尚未保存",
    "articles.notes.missing": "暂无补充说明。",
    "labels.untitled": "未命名",
    "labels.na": "暂无",
    "search.no_results": "未找到匹配的论文，请尝试其他关键词。",
    "search.no_summary": "暂无摘要，可通过下方 PDF 了解详情。",
    "search.repo.available": "GitHub：已提供仓库链接",
    "search.repo.missing": "GitHub：尚未收录",
    "search.pdf.download": "下载 PDF",
    "search.pdf.online": "在线查看 PDF",
    "search.pdf.missing_btn": "PDF 未保存",
    "search.repo.button": "打开仓库",
    "search.repo.button_missing": "暂无仓库",
    "search.added": "已添加",
    "search.add_to_migration": "加入迁移列表",
    "search.add_unavailable": "不可添加",
    "selection.empty": "暂无满足“提供 PDF 或链接且有 GitHub 仓库”的论文，请先运行文献搜索。",
    "selection.model_missing": "未指定模型",
    "migration.feedback.none_selected": "所选论文尚未生成迁移 Summary，运行迁移阶段后刷新即可。",
    "migration.feedback.none_available": "还没有迁移 Summary，执行一次模型迁移即可查看。",
    "migration.feedback.model": "模型：{{value}}",
    "migration.feedback.open_repo": "打开仓库",
    "migration.feedback.no_repo": "暂无仓库",
    "migration.feedback.view_pdf": "查看 PDF",
    "migration.feedback.view_summary": "查看 Summary",
    "migration.feedback.summary_missing": "Summary 未生成",
    "migration.feedback.summary_empty": "暂无 Summary 内容，可重新运行迁移阶段生成。",
    "migration.feedback.updated": "最近更新：{{value}}",
    "jobs.empty": "暂无任务，提交文献搜索、模型迁移或调参以查看状态。",
    "jobs.time.created": "创建",
    "jobs.time.started": "开始",
    "jobs.time.finished": "结束",
    "jobs.meta.paper": "论文：{{value}}",
    "jobs.meta.model": "模型：{{value}}",
    "jobs.meta.error": "错误：{{value}}",
    "jobs.status.running": "运行中",
    "jobs.status.succeeded": "完成",
    "jobs.status.failed": "失败",
    "jobs.status.pending": "排队中",
    "alerts.loading": "处理中...",
    "alerts.submit_literature_loading": "提交中...",
    "alerts.start_migration_loading": "启动中...",
    "alerts.start_tuning_loading": "启动调参...",
    "alerts.literature_failed": "提交文献搜索任务失败",
    "alerts.migration_failed": "提交模型迁移任务失败",
    "alerts.tuning_failed": "提交调参任务失败",
    "alerts.no_selection": "请选择至少一篇论文再启动任务。",
    "alerts.export_no_selection": "请选择至少一篇论文再导出迁移计划。",
    "alerts.migration_started_single": "已启动 1 个迁移任务。",
    "alerts.migration_started_multi": "已启动 {{count}} 个迁移任务。",
    "alerts.tuning_started_single": "已启动 1 个调参任务。",
    "alerts.tuning_started_multi": "已启动 {{count}} 个调参任务。",
    "errors.dashboard_load": "加载仪表盘数据失败",
  },
};

const state = {
  stages: [],
  stagePayload: { stages: [] },
  articles: [],
  migrationCatalog: [],
  migrationLookup: new Map(),
  jobs: [],
  selectedPapers: new Set(),
  completedJobs: new Set(),
  jobPoller: null,
};

let currentLanguage = localStorage.getItem("appLanguage") || "zh";

function t(key, vars = {}) {
  const catalog = TRANSLATIONS[currentLanguage] || {};
  const template = catalog[key] ?? key;
  return template.replace(/\{\{(\w+)\}\}/g, (_, token) =>
    vars[token] !== undefined ? vars[token] : ""
  );
}

function updateStaticText() {
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (!key) return;
    if (element.dataset.i18nHtml === "true") {
      element.innerHTML = t(key);
    } else {
      element.textContent = t(key);
    }
    if (element.dataset.originalText !== undefined) {
      element.dataset.originalText = element.textContent;
    }
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    const key = element.dataset.i18nPlaceholder;
    if (!key) return;
    element.placeholder = t(key);
  });
}

function setLanguage(lang) {
  currentLanguage = lang === "en" ? "en" : "zh";
  localStorage.setItem("appLanguage", currentLanguage);
  document.body.dataset.lang = currentLanguage;
  updateStaticText();
  refreshAllViews();
}

function toggleLanguage() {
  setLanguage(currentLanguage === "en" ? "zh" : "en");
}

async function fetchJSON(path, options = {}) {
  const requestOptions = { ...options };
  if (requestOptions.body && typeof requestOptions.body !== "string") {
    requestOptions.body = JSON.stringify(requestOptions.body);
    requestOptions.headers = {
      "Content-Type": "application/json",
      ...(requestOptions.headers || {}),
    };
  }
  const response = await fetch(`${API_BASE}${path}`, requestOptions);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.status === 204 ? null : await response.json();
}

function setButtonLoading(button, isLoading, loadingText = t("alerts.loading")) {
  if (!button) return;
  if (!button.dataset.originalText) {
    button.dataset.originalText = button.textContent;
  }
  button.dataset.loading = isLoading ? "true" : "false";
  if (isLoading) {
    button.textContent = loadingText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText;
    button.disabled = false;
  }
}

function handleError(message, error) {
  console.error(message, error);
  alert(`${message}\n${error?.message || error}`);
}

function getArticleId(article) {
  return `${article.title || "Untitled"}__${article.conference || "N/A"}`;
}

function normalize(text) {
  return (text || "").toString().toLowerCase();
}

function truncateText(text, limit = 220) {
  if (!text) return "";
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function normalizeLocalAssetPath(rawPath) {
  if (!rawPath) return "";
  const normalized = rawPath.replace(/\\/g, "/");
  if (/^https?:\/\//i.test(normalized)) {
    return normalized;
  }
  const dataMarker = "/data/";
  const markerIndex = normalized.indexOf(dataMarker);
  if (markerIndex !== -1) {
    return normalized.slice(markerIndex);
  }
  const withoutCurrentDir = normalized.replace(/^\.\/+/, "");
  if (withoutCurrentDir.startsWith("data/")) {
    return `/${withoutCurrentDir}`;
  }
  const withoutParentDir = withoutCurrentDir.replace(/^(\.\.\/)+/, "");
  if (withoutParentDir.startsWith("data/")) {
    return `/${withoutParentDir}`;
  }
  if (withoutParentDir.startsWith("/")) {
    return withoutParentDir;
  }
  return withoutParentDir ? `/${withoutParentDir}` : "";
}

function getRepoUrl(article) {
  return article?.repo_url || article?.github || "";
}

function getPdfPath(article) {
  return article?.pdf_path || article?.pdf_file || "";
}

function getPdfHref(article) {
  const localPath = normalizeLocalAssetPath(getPdfPath(article));
  if (localPath) {
    return /^https?:\/\//i.test(localPath) ? localPath : localPath.startsWith("/")
        ? localPath
        : `/${localPath}`;
  }
  return article?.pdf_link || "";
}

function getSummaryHref(entry) {
  const rawPath = entry?.summary_path || entry?.summary_file || "";
  const normalized = normalizeLocalAssetPath(rawPath);
  if (!normalized) return "";
  return normalized.startsWith("/") ? normalized : `/${normalized}`;
}

function hasRepo(article) {
  return Boolean(getRepoUrl(article));
}

function hasSavedPdf(article) {
  return Boolean(getPdfPath(article));
}

function hasAvailablePdf(article) {
  return Boolean(getPdfHref(article));
}

function canSelectForMigration(article) {
  return hasAvailablePdf(article) && hasRepo(article);
}

function renderStages(stagePayload) {
  const container = document.getElementById("stage-grid");
  const stages = Array.isArray(stagePayload?.stages)
    ? stagePayload.stages
    : Array.isArray(stagePayload)
    ? stagePayload
    : [];

  if (stages.length === 0) {
    container.innerHTML = `<div class="empty-state">${t("stages.empty")}</div>`;
    return;
  }

  container.innerHTML = "";
  stages.forEach((stage, index) => {
    const card = document.createElement("article");
    card.className = "card";

    const header = document.createElement("h3");
    const workflowTag = stage.workflow || stage.key;
    header.innerHTML = `<span>${index + 1}. ${stage.title}</span><span class="tag">${workflowTag}</span>`;

    const subLabel = document.createElement("p");
    subLabel.className = "card__meta";
    subLabel.textContent = t("stages.stage_id", { id: stage.key });

    const summary = document.createElement("p");
    summary.className = "card__summary";
    summary.textContent =
      stage.summary?.trim() || t("stages.summary_fallback");

    const transcript = document.createElement("details");
    transcript.innerHTML = `<summary>${t("stages.transcript_label", {
      count: stage.messages.length,
    })}</summary>`;

    const list = document.createElement("ul");
    const snippets = stage.messages
      .filter((msg) => msg.type === "text")
      .slice(-3);
    if (snippets.length === 0) {
      const item = document.createElement("li");
      item.textContent = t("stages.no_snippets");
      list.appendChild(item);
    } else {
      snippets.forEach((chunk) => {
        const item = document.createElement("li");
        item.textContent = chunk.content;
        list.appendChild(item);
      });
    }
    transcript.appendChild(list);

    card.appendChild(header);
    card.appendChild(subLabel);
    card.appendChild(summary);
    card.appendChild(transcript);
    container.appendChild(card);
  });
}

function renderArticles(articles) {
  const container = document.getElementById("article-grid");
  if (!Array.isArray(articles) || articles.length === 0) {
    container.innerHTML = `<div class="empty-state">${t("articles.empty")}</div>`;
    return;
  }

  container.innerHTML = "";
  articles.forEach((article) => {
    const card = document.createElement("article");
    card.className = "card";
    const repoUrl = getRepoUrl(article);
    const pdfHref = getPdfHref(article);
    const localPdf = hasSavedPdf(article);

    const header = document.createElement("h3");
    header.innerHTML = `<span>${article.title || t("labels.untitled")}</span><span class="tag">${article.conference ||
      t("labels.na")}</span>`;

    const meta = document.createElement("p");
    meta.className = "card__meta";
    const datasetValue = Array.isArray(article.datasets)
      ? article.datasets.join(", ")
      : article.datasets || "";
    meta.textContent = t("articles.datasets", {
      value: datasetValue || t("articles.datasets.unknown"),
    });

    const modelMeta = document.createElement("p");
    modelMeta.className = "card__meta";
    modelMeta.textContent = article.model_name
      ? t("articles.model", { value: article.model_name })
      : t("articles.model.pending");

    const repoLink = document.createElement("a");
    repoLink.className = "article-link";
    if (repoUrl) {
      repoLink.href = repoUrl;
      repoLink.textContent = t("articles.repo.link");
      repoLink.target = "_blank";
      repoLink.rel = "noreferrer";
    } else {
      repoLink.textContent = t("articles.repo.missing");
      repoLink.classList.add("card__meta");
    }

    const pdfLink = document.createElement("a");
    pdfLink.className = "article-link";
    if (pdfHref) {
      pdfLink.href = pdfHref;
      pdfLink.textContent = localPdf
        ? t("articles.pdf.local")
        : t("articles.pdf.remote");
      pdfLink.target = "_blank";
      if (/^https?:\/\//i.test(pdfHref)) {
        pdfLink.rel = "noreferrer";
      }
    } else {
      pdfLink.textContent = t("articles.pdf.missing");
      pdfLink.classList.add("card__meta");
    }

    const notes = document.createElement("p");
    notes.className = "card__summary";
    notes.textContent = article.notes || t("articles.notes.missing");

    card.appendChild(header);
    card.appendChild(meta);
    card.appendChild(modelMeta);
    const linkRow = document.createElement("div");
    linkRow.className = "link-row";
    linkRow.appendChild(repoLink);
    linkRow.appendChild(pdfLink);
    card.appendChild(linkRow);
    card.appendChild(notes);
    container.appendChild(card);
  });
}

function filterArticlesByKeywords(keywords) {
  if (!keywords.length) return [...state.articles];
  return state.articles.filter((article) => {
    const haystack = normalize(
      `${article.title} ${article.notes} ${article.datasets} ${article.conference}`
    );
    return keywords.every((keyword) => haystack.includes(keyword));
  });
}

function renderSearchResults(results) {
  const container = document.getElementById("search-results");
  if (!Array.isArray(results) || results.length === 0) {
    container.innerHTML = `<div class="empty-state">${t("search.no_results")}</div>`;
    return;
  }

  container.innerHTML = "";
  results.forEach((article) => {
    const card = document.createElement("article");
    card.className = "card";
    const header = document.createElement("h3");
    header.innerHTML = `<span>${article.title || t("labels.untitled")}</span><span class="tag">${article.conference ||
      t("labels.na")}</span>`;

    const excerpt = document.createElement("p");
    excerpt.className = "card__summary";
    excerpt.textContent =
      truncateText(article.notes) || t("search.no_summary");

    const repoUrl = getRepoUrl(article);
    const pdfHref = getPdfHref(article);
    const localPdf = hasSavedPdf(article);
    const repoStatus = document.createElement("p");
    repoStatus.className = "card__meta repo-status";
    repoStatus.textContent = (Boolean(repoUrl) && String(repoUrl).toLowerCase() !== "not available")
      ? t("search.repo.available")
      : t("search.repo.missing");

    const actionRow = document.createElement("div");
    actionRow.className = "card__actions";

    const pdfBtn = document.createElement("a");
    pdfBtn.className = "btn btn--inline";
    if (pdfHref) {
      pdfBtn.href = pdfHref;
      pdfBtn.target = "_blank";
      pdfBtn.textContent = localPdf
        ? t("search.pdf.download")
        : t("search.pdf.online");
      if (/^https?:\/\//i.test(pdfHref)) {
        pdfBtn.rel = "noreferrer";
      }
    } else {
      pdfBtn.textContent = t("search.pdf.missing_btn");
      pdfBtn.classList.add("btn--disabled");
    }

    const repoBtn = document.createElement("a");
    repoBtn.className = "btn btn--inline btn--secondary";
    if (repoUrl) {
      repoBtn.href = repoUrl;
      repoBtn.target = "_blank";
      repoBtn.rel = "noreferrer";
      repoBtn.textContent = t("search.repo.button");
    } else {
      repoBtn.textContent = t("search.repo.button_missing");
      repoBtn.classList.add("btn--disabled");
    }

    const articleId = getArticleId(article);
    const alreadySelected = state.selectedPapers.has(articleId);
    const eligible = canSelectForMigration(article);

    actionRow.appendChild(pdfBtn);
    actionRow.appendChild(repoBtn);

    if (!eligible && !alreadySelected) {
      const note = document.createElement("p");
      note.className = "card__meta action-note";
      note.textContent = t("search.add_unavailable");
      actionRow.appendChild(note);
    } else {
      const addBtn = document.createElement("button");
      addBtn.type = "button";
      addBtn.className = "btn btn--inline btn--ghost";
      addBtn.textContent = alreadySelected
        ? t("search.added")
        : t("search.add_to_migration");
      addBtn.disabled = alreadySelected;
      addBtn.addEventListener("click", () => {
        if (!eligible) return;
        state.selectedPapers.add(articleId);
        renderMigrationSelection();
        renderMigrationFeedback();
        performSearch();
      });
      actionRow.appendChild(addBtn);
    }

    card.appendChild(header);
    card.appendChild(excerpt);
    card.appendChild(repoStatus);
    card.appendChild(actionRow);
    container.appendChild(card);
  });
}

function renderMigrationSelection() {
  const container = document.getElementById("migration-selection");
  const eligible = state.articles.filter(
    (article) => hasAvailablePdf(article) && hasRepo(article)
  );

  if (eligible.length === 0) {
    container.innerHTML = `<div class="empty-state">${t("selection.empty")}</div>`;
    return;
  }

  const validIds = new Set(eligible.map((article) => getArticleId(article)));
  state.selectedPapers.forEach((id) => {
    if (!validIds.has(id)) {
      state.selectedPapers.delete(id);
    }
  });
  const selectedIds = new Set(state.selectedPapers);

  container.innerHTML = "";
  eligible.forEach((article) => {
    const wrapper = document.createElement("label");
    wrapper.className = "selection-item";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    const articleId = getArticleId(article);
    checkbox.checked = selectedIds.has(articleId);
    checkbox.addEventListener("change", (event) => {
      if (event.target.checked) {
        state.selectedPapers.add(articleId);
      } else {
        state.selectedPapers.delete(articleId);
      }
      renderMigrationFeedback();
      performSearch();
    });

    const labelText = document.createElement("span");
    labelText.innerHTML = `<strong>${article.title ||
      t("labels.untitled")}</strong><small>${article.model_name ||
      t("selection.model_missing")}</small>`;

    wrapper.appendChild(checkbox);
    wrapper.appendChild(labelText);
    container.appendChild(wrapper);
  });
}

function getMigrationKey(entry) {
  if (!entry) return "";
  if (entry.paper_id) return entry.paper_id;
  const title = entry.title || entry.paper_title || "Untitled";
  const conference = entry.conference || "N/A";
  return `${title}__${conference}`;
}

function normalizeMigrationItems(payload) {
  const items = Array.isArray(payload?.items)
    ? payload.items
    : Array.isArray(payload)
    ? payload
    : [];
  return items.map((entry) => {
    const paperId = entry.paper_id || entry.paperId || getMigrationKey(entry);
    return {
      ...entry,
      paper_id: paperId,
      title: entry.title || entry.paper_title || entry.paperTitle || "Untitled",
      conference: entry.conference || entry.paper_conference || entry.conference_name || "N/A",
      summary_excerpt: entry.summary_excerpt || entry.summary || "",
      summary_path: entry.summary_path || entry.summaryPath || "",
      last_updated: entry.last_updated || entry.updated_at || entry.timestamp || "",
    };
  });
}

function renderMigrationFeedback() {
  const container = document.getElementById("migration-feedback");
  const entries = Array.isArray(state.migrationCatalog)
    ? [...state.migrationCatalog]
    : [];
  let filtered = entries;
  if (state.selectedPapers.size > 0) {
    const ids = new Set(state.selectedPapers);
    filtered = entries.filter((entry) => ids.has(getMigrationKey(entry)));
  }
  if (filtered.length === 0) {
    container.innerHTML =
      state.selectedPapers.size > 0
        ? `<div class="empty-state">${t("migration.feedback.none_selected")}</div>`
        : `<div class="empty-state">${t("migration.feedback.none_available")}</div>`;
    return;
  }

  container.innerHTML = "";
  filtered.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "card";

    const header = document.createElement("h3");
    header.innerHTML = `<span>${entry.title ||
      t("labels.untitled")}</span><span class="tag">${entry.conference ||
      t("labels.na")}</span>`;
    card.appendChild(header);

    const modelMeta = document.createElement("p");
    modelMeta.className = "card__meta";
    modelMeta.textContent = entry.model_name
      ? t("migration.feedback.model", { value: entry.model_name })
      : t("migration.feedback.model", { value: t("selection.model_missing") });
    card.appendChild(modelMeta);

    const linkRow = document.createElement("div");
    linkRow.className = "link-row";

    const repoLink = document.createElement("a");
    repoLink.className = "article-link";
    if (entry.repo_url) {
      repoLink.href = entry.repo_url;
      repoLink.target = "_blank";
      repoLink.rel = "noreferrer";
      repoLink.textContent = t("migration.feedback.open_repo");
    } else {
      repoLink.textContent = t("migration.feedback.no_repo");
      repoLink.classList.add("card__meta");
    }
    linkRow.appendChild(repoLink);

    const pdfHref = getPdfHref(entry);
    if (pdfHref) {
      const pdfLink = document.createElement("a");
      pdfLink.className = "article-link";
      pdfLink.href = pdfHref;
      pdfLink.target = "_blank";
      if (/^https?:/i.test(pdfHref)) {
        pdfLink.rel = "noreferrer";
      }
      pdfLink.textContent = t("migration.feedback.view_pdf");
      linkRow.appendChild(pdfLink);
    }

    const summaryHref = getSummaryHref(entry);
    const summaryLink = document.createElement("a");
    summaryLink.className = "article-link";
    if (summaryHref) {
      summaryLink.href = summaryHref;
      summaryLink.target = "_blank";
      summaryLink.textContent = t("migration.feedback.view_summary");
    } else {
      summaryLink.textContent = t("migration.feedback.summary_missing");
      summaryLink.classList.add("card__meta");
    }
    linkRow.appendChild(summaryLink);
    card.appendChild(linkRow);

    const excerpt = document.createElement("p");
    excerpt.className = "card__summary";
    excerpt.textContent =
      entry.summary_excerpt?.trim() ||
      t("migration.feedback.summary_empty");
    card.appendChild(excerpt);

    const updated = document.createElement("p");
    updated.className = "card__meta";
    updated.textContent = t("migration.feedback.updated", {
      value: entry.last_updated || t("labels.na"),
    });
    card.appendChild(updated);

    container.appendChild(card);
  });
}

function renderJobs() {
  const container = document.getElementById("job-grid");
  if (!container) return;
  if (!Array.isArray(state.jobs) || state.jobs.length === 0) {
    container.innerHTML = `<div class="empty-state">${t("jobs.empty")}</div>`;
    return;
  }

  container.innerHTML = "";
  state.jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "card job-card";

    const header = document.createElement("h3");
    header.innerHTML = `<span>${job.label}</span><span class="tag">${job.id.slice(
      0,
      6
    )}</span>`;

    const status = document.createElement("span");
    status.className = `status-pill job-status ${formatJobStatusClass(
      job.status
    )}`;
    status.textContent = formatJobStatusLabel(job.status);

    const times = document.createElement("div");
    times.className = "job-times";
    times.innerHTML = `
      <span>${t("jobs.time.created")}: ${job.created_at || "--"}</span>
      <span>${t("jobs.time.started")}: ${job.started_at || "--"}</span>
      <span>${t("jobs.time.finished")}: ${job.finished_at || "--"}</span>
    `;

    card.appendChild(header);
    card.appendChild(status);
    card.appendChild(times);
    if (job.paper_title || job.model_name) {
      const meta = document.createElement("p");
      meta.className = "card__meta";
      const parts = [];
      if (job.paper_title) {
        parts.push(t("jobs.meta.paper", { value: job.paper_title }));
      }
      if (job.model_name) {
        parts.push(t("jobs.meta.model", { value: job.model_name }));
      }
      meta.textContent = parts.join(" · ");
      card.appendChild(meta);
    }
    if (job.error) {
      const errorText = document.createElement("p");
      errorText.className = "card__summary";
      errorText.textContent = t("jobs.meta.error", { value: job.error });
      card.appendChild(errorText);
    }
    container.appendChild(card);
  });
}

function formatJobStatusClass(status) {
  switch (status) {
    case "running":
      return "info";
    case "succeeded":
      return "success";
    case "failed":
      return "error";
    default:
      return "pending";
  }
}

function formatJobStatusLabel(status) {
  switch (status) {
    case "running":
      return t("jobs.status.running");
    case "succeeded":
      return t("jobs.status.succeeded");
    case "failed":
      return t("jobs.status.failed");
    case "pending":
    default:
      return t("jobs.status.pending");
  }
}

function getCurrentKeywords() {
  const input = document.getElementById("keyword-input");
  if (!input) return [];
  return input.value
    .split(/[,，\s]+/)
    .map((word) => normalize(word))
    .filter(Boolean);
}

function getSelectedYearFilter() {
  const select = document.getElementById("year-filter");
  if (!select) return "this_year";
  return select.value || "this_year";
}

function getConferenceFilters() {
  const input = document.getElementById("conference-input");
  if (!input) return [];
  return input.value
    .split(/[,，\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function performSearch() {
  const keywords = getCurrentKeywords();
  const results = filterArticlesByKeywords(keywords);
  renderSearchResults(results);
}

async function requestLiteratureJob(keywords, button, yearFilter, conferences) {
  try {
    setButtonLoading(button, true, t("alerts.submit_literature_loading"));
    await fetchJSON("/literature/run", {
      method: "POST",
      body: {
        keywords,
        year_mode: yearFilter,
        conferences,
      },
    });
    await refreshJobs();
  } catch (error) {
    handleError(t("alerts.literature_failed"), error);
  } finally {
    setButtonLoading(button, false);
  }
}

async function requestMigrationJob(button) {
  if (state.selectedPapers.size === 0) {
    alert(t("alerts.no_selection"));
    return;
  }
  try {
    setButtonLoading(button, true, t("alerts.start_migration_loading"));
    const result = await fetchJSON("/migration/run", {
      method: "POST",
      body: { paper_ids: Array.from(state.selectedPapers) },
    });
    const jobs = Array.isArray(result?.items)
      ? result.items
      : result
      ? [result]
      : [];
    if (jobs.length > 0) {
      const message =
        jobs.length === 1
          ? t("alerts.migration_started_single")
          : t("alerts.migration_started_multi", { count: jobs.length });
      alert(message);
    }
    await refreshJobs();
  } catch (error) {
    handleError(t("alerts.migration_failed"), error);
  } finally {
    setButtonLoading(button, false);
  }
}

async function requestTuningJob(button) {
  if (state.selectedPapers.size === 0) {
    alert(t("alerts.no_selection"));
    return;
  }
  try {
    setButtonLoading(button, true, t("alerts.start_tuning_loading"));
    const result = await fetchJSON("/tuning/run", {
      method: "POST",
      body: { paper_ids: Array.from(state.selectedPapers) },
    });
    const jobs = Array.isArray(result?.items)
      ? result.items
      : result
      ? [result]
      : [];
    if (jobs.length > 0) {
      const message =
        jobs.length === 1
          ? t("alerts.tuning_started_single")
          : t("alerts.tuning_started_multi", { count: jobs.length });
      alert(message);
    }
    await refreshJobs();
  } catch (error) {
    handleError(t("alerts.tuning_failed"), error);
  } finally {
    setButtonLoading(button, false);
  }
}

function exportMigrationPlan() {
  if (state.selectedPapers.size === 0) {
    alert(t("alerts.export_no_selection"));
    return;
  }
  const payload = Array.from(state.selectedPapers).map((articleId) => {
    const article = state.articles.find(
      (item) => getArticleId(item) === articleId
    );
    return {
      title: article?.title || "Untitled",
      conference: article?.conference || "N/A",
      repo_url: getRepoUrl(article),
      model_name: article?.model_name || "",
      pdf_path: getPdfPath(article),
      datasets: article?.datasets || "",
    };
  });

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "migration_plan.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function loadDashboard() {
  try {
    const migrationPromise = fetchJSON("/migration/catalog").catch(async () => {
      try {
        return await fetchJSON("/migration/results");
      } catch (error) {
        console.warn("无法加载迁移 catalog，尝试旧版结果：", error);
        return { items: [] };
      }
    });
    const [stagePayload, articlePayload, migrationPayload] = await Promise.all([
      fetchJSON("/stages"),
      fetchJSON("/articles"),
      migrationPromise,
    ]);

    state.stages = Array.isArray(stagePayload?.stages)
      ? stagePayload.stages
      : [];
    state.stagePayload = { stages: state.stages };
    state.articles = Array.isArray(articlePayload?.items)
      ? articlePayload.items
      : [];
    state.migrationCatalog = normalizeMigrationItems(migrationPayload);
    state.migrationLookup = new Map(
      state.migrationCatalog
        .map((entry) => [getMigrationKey(entry), entry])
        .filter(([key]) => Boolean(key))
    );

    renderStages(state.stagePayload);
    renderArticles(state.articles);
    renderMigrationSelection();
    renderMigrationFeedback();
    performSearch();
  } catch (error) {
    handleError(t("errors.dashboard_load"), error);
  }
}

async function refreshJobs() {
  try {
    const jobPayload = await fetchJSON("/jobs");
    const jobs = Array.isArray(jobPayload?.items) ? jobPayload.items : [];
    const newlyFinished = jobs.filter(
      (job) =>
        job.finished_at &&
        ["succeeded", "failed"].includes(job.status) &&
        !state.completedJobs.has(job.id)
    );

    jobs
      .filter((job) => job.finished_at && ["succeeded", "failed"].includes(job.status))
      .forEach((job) => state.completedJobs.add(job.id));

    state.jobs = jobs;
    renderJobs();

    if (newlyFinished.length > 0) {
      await loadDashboard();
    }
  } catch (error) {
    console.warn("刷新任务列表失败:", error);
  }
}

function startJobPolling() {
  if (state.jobPoller) return;
  state.jobPoller = setInterval(refreshJobs, 5000);
}

function refreshAllViews() {
  renderStages(state.stagePayload || { stages: state.stages || [] });
  renderArticles(state.articles);
  renderMigrationSelection();
  renderMigrationFeedback();
  performSearch();
  renderJobs();
}

document.addEventListener("DOMContentLoaded", () => {
  setLanguage(currentLanguage);
  loadDashboard();
  refreshJobs();
  startJobPolling();
  const langToggle = document.getElementById("lang-toggle");
  if (langToggle) {
    langToggle.addEventListener("click", toggleLanguage);
  }

  const form = document.getElementById("search-form");
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const keywords = getCurrentKeywords();
      const submitBtn = form.querySelector('button[type="submit"]');
      const yearFilter = getSelectedYearFilter();
      const conferences = getConferenceFilters();
      if (keywords.length > 0) {
        await requestLiteratureJob(keywords, submitBtn, yearFilter, conferences);
      }
      performSearch();
    });
  }

  const input = document.getElementById("keyword-input");
  if (input) {
    input.addEventListener("input", performSearch);
  }

  const resetBtn = document.getElementById("reset-search");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (input) input.value = "";
      const yearSelect = document.getElementById("year-filter");
      if (yearSelect) yearSelect.value = "this_year";
      const conferenceInput = document.getElementById("conference-input");
      if (conferenceInput) conferenceInput.value = "";
      performSearch();
    });
  }

  const exportBtn = document.getElementById("export-plan");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportMigrationPlan);
  }

  const startBtn = document.getElementById("start-migration");
  if (startBtn) {
    startBtn.addEventListener("click", () => requestMigrationJob(startBtn));
  }

  const tuningBtn = document.getElementById("start-tuning");
  if (tuningBtn) {
    tuningBtn.addEventListener("click", () => requestTuningJob(tuningBtn));
  }
});

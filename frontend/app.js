const API_BASE = "../api";

const state = {
  stages: [],
  articles: [],
  migrationCatalog: [],
  migrationLookup: new Map(),
  jobs: [],
  selectedPapers: new Set(),
  completedJobs: new Set(),
  jobPoller: null,
};

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

function setButtonLoading(button, isLoading, loadingText = "处理中...") {
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

function renderStages(stagePayload) {
  const container = document.getElementById("stage-grid");
  const stages = Array.isArray(stagePayload?.stages)
    ? stagePayload.stages
    : Array.isArray(stagePayload)
    ? stagePayload
    : [];

  if (stages.length === 0) {
    container.innerHTML =
      '<div class="empty-state">No pipeline run recorded yet.</div>';
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
    subLabel.textContent = `Stage ID: ${stage.key}`;

    const summary = document.createElement("p");
    summary.className = "card__summary";
    summary.textContent =
      stage.summary?.trim() ||
      "Stage completed without textual summary. Check raw transcript below.";

    const transcript = document.createElement("details");
    transcript.innerHTML = `<summary>Transcript (${stage.messages.length} messages)</summary>`;

    const list = document.createElement("ul");
    const snippets = stage.messages
      .filter((msg) => msg.type === "text")
      .slice(-3);
    if (snippets.length === 0) {
      const item = document.createElement("li");
      item.textContent = "No text snippets captured.";
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
    container.innerHTML =
      '<div class="empty-state">No articles cataloged yet. Run the pipeline to populate this grid.</div>';
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
    header.innerHTML = `<span>${article.title || "Untitled"}</span><span class="tag">${article.conference ||
      "N/A"}</span>`;

    const meta = document.createElement("p");
    meta.className = "card__meta";
    meta.textContent = `Datasets: ${article.datasets || "Unknown"}`;

    const modelMeta = document.createElement("p");
    modelMeta.className = "card__meta";
    modelMeta.textContent = article.model_name
      ? `Model: ${article.model_name}`
      : "Model: Pending catalog update";

    const repoLink = document.createElement("a");
    repoLink.className = "article-link";
    if (repoUrl) {
      repoLink.href = repoUrl;
      repoLink.textContent = "Repository";
      repoLink.target = "_blank";
      repoLink.rel = "noreferrer";
    } else {
      repoLink.textContent = "No repository shared";
      repoLink.classList.add("card__meta");
    }

    const pdfLink = document.createElement("a");
    pdfLink.className = "article-link";
    if (pdfHref) {
      pdfLink.href = pdfHref;
      pdfLink.textContent = localPdf ? "Open PDF" : "View PDF Online";
      pdfLink.target = "_blank";
      if (/^https?:\/\//i.test(pdfHref)) {
        pdfLink.rel = "noreferrer";
      }
    } else {
      pdfLink.textContent = "PDF not saved yet";
      pdfLink.classList.add("card__meta");
    }

    const notes = document.createElement("p");
    notes.className = "card__summary";
    notes.textContent = article.notes || "No additional notes.";

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
    container.innerHTML =
      '<div class="empty-state">未找到匹配的论文，请尝试不同的关键词。</div>';
    return;
  }

  container.innerHTML = "";
  results.forEach((article) => {
    const card = document.createElement("article");
    card.className = "card";
    const header = document.createElement("h3");
    header.innerHTML = `<span>${article.title || "Untitled"}</span><span class="tag">${article.conference ||
      "N/A"}</span>`;

    const excerpt = document.createElement("p");
    excerpt.className = "card__summary";
    excerpt.textContent =
      truncateText(article.notes) ||
      "暂无摘要，可通过下方 PDF 了解详细内容。";

    const repoUrl = getRepoUrl(article);
    const pdfHref = getPdfHref(article);
    const localPdf = hasSavedPdf(article);
    const repoStatus = document.createElement("p");
    repoStatus.className = "card__meta repo-status";
    repoStatus.textContent = repoUrl
      ? "GitHub：已提供仓库链接"
      : "GitHub：尚未收录";

    const actionRow = document.createElement("div");
    actionRow.className = "card__actions";

    const pdfBtn = document.createElement("a");
    pdfBtn.className = "btn btn--inline";
    if (pdfHref) {
      pdfBtn.href = pdfHref;
      pdfBtn.target = "_blank";
      pdfBtn.textContent = localPdf ? "下载 PDF" : "查看在线 PDF";
      if (/^https?:\/\//i.test(pdfHref)) {
        pdfBtn.rel = "noreferrer";
      }
    } else {
      pdfBtn.textContent = "PDF 未保存";
      pdfBtn.classList.add("btn--disabled");
    }

    const repoBtn = document.createElement("a");
    repoBtn.className = "btn btn--inline btn--secondary";
    if (repoUrl) {
      repoBtn.href = repoUrl;
      repoBtn.target = "_blank";
      repoBtn.rel = "noreferrer";
      repoBtn.textContent = "打开仓库";
    } else {
      repoBtn.textContent = "暂无仓库";
      repoBtn.classList.add("btn--disabled");
    }

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn btn--inline btn--ghost";
    const articleId = getArticleId(article);
    const alreadySelected = state.selectedPapers.has(articleId);
    addBtn.textContent = alreadySelected ? "已添加" : "加入迁移列表";
    addBtn.disabled = alreadySelected;
    addBtn.addEventListener("click", () => {
      state.selectedPapers.add(articleId);
      renderMigrationSelection();
      renderMigrationFeedback();
      performSearch();
    });

    actionRow.appendChild(pdfBtn);
    actionRow.appendChild(repoBtn);
    actionRow.appendChild(addBtn);

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
    (article) => hasSavedPdf(article) && hasRepo(article)
  );

  if (eligible.length === 0) {
    container.innerHTML =
      '<div class="empty-state">暂无满足“已保存 PDF 且提供 GitHub 仓库”的论文。</div>';
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
      "Untitled"}</strong><small>${article.model_name || "未指定模型"}</small>`;

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
        ? '<div class="empty-state">所选论文尚未生成迁移 Summary，运行管线后刷新即可。</div>'
        : '<div class="empty-state">暂无迁移 Summary，执行一次模型迁移阶段后查看。</div>';
    return;
  }

  container.innerHTML = "";
  filtered.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "card";

    const header = document.createElement("h3");
    header.innerHTML = `<span>${entry.title ||
      "Untitled"}</span><span class="tag">${entry.conference || "N/A"}</span>`;
    card.appendChild(header);

    const modelMeta = document.createElement("p");
    modelMeta.className = "card__meta";
    modelMeta.textContent = entry.model_name
      ? `模型：${entry.model_name}`
      : "模型：未记录";
    card.appendChild(modelMeta);

    const linkRow = document.createElement("div");
    linkRow.className = "link-row";

    const repoLink = document.createElement("a");
    repoLink.className = "article-link";
    if (entry.repo_url) {
      repoLink.href = entry.repo_url;
      repoLink.target = "_blank";
      repoLink.rel = "noreferrer";
      repoLink.textContent = "打开仓库";
    } else {
      repoLink.textContent = "暂无仓库";
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
      pdfLink.textContent = "查看 PDF";
      linkRow.appendChild(pdfLink);
    }

    const summaryHref = getSummaryHref(entry);
    const summaryLink = document.createElement("a");
    summaryLink.className = "article-link";
    if (summaryHref) {
      summaryLink.href = summaryHref;
      summaryLink.target = "_blank";
      summaryLink.textContent = "查看 Summary";
    } else {
      summaryLink.textContent = "Summary 未生成";
      summaryLink.classList.add("card__meta");
    }
    linkRow.appendChild(summaryLink);
    card.appendChild(linkRow);

    const excerpt = document.createElement("p");
    excerpt.className = "card__summary";
    excerpt.textContent =
      entry.summary_excerpt?.trim() ||
      "暂无 Summary 内容，可重新运行迁移阶段生成。";
    card.appendChild(excerpt);

    const updated = document.createElement("p");
    updated.className = "card__meta";
    updated.textContent = `最近更新: ${entry.last_updated || "N/A"}`;
    card.appendChild(updated);

    container.appendChild(card);
  });
}

function renderJobs() {
  const container = document.getElementById("job-grid");
  if (!container) return;
  if (!Array.isArray(state.jobs) || state.jobs.length === 0) {
    container.innerHTML =
      '<div class="empty-state">暂无任务，提交文献搜索或模型迁移以查看状态。</div>';
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
      <span>创建: ${job.created_at || "--"}</span>
      <span>开始: ${job.started_at || "--"}</span>
      <span>结束: ${job.finished_at || "--"}</span>
    `;

    card.appendChild(header);
    card.appendChild(status);
    card.appendChild(times);
    if (job.paper_title || job.model_name) {
      const meta = document.createElement("p");
      meta.className = "card__meta";
      const parts = [];
      if (job.paper_title) {
        parts.push(`论文: ${job.paper_title}`);
      }
      if (job.model_name) {
        parts.push(`模型: ${job.model_name}`);
      }
      meta.textContent = parts.join(" · ");
      card.appendChild(meta);
    }
    if (job.error) {
      const errorText = document.createElement("p");
      errorText.className = "card__summary";
      errorText.textContent = `错误: ${job.error}`;
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
      return "运行中";
    case "succeeded":
      return "完成";
    case "failed":
      return "失败";
    case "pending":
    default:
      return "排队中";
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

function performSearch() {
  const keywords = getCurrentKeywords();
  const results = filterArticlesByKeywords(keywords);
  renderSearchResults(results);
}

async function requestLiteratureJob(keywords, button) {
  try {
    setButtonLoading(button, true, "提交中...");
    await fetchJSON("/literature/run", {
      method: "POST",
      body: { keywords },
    });
    await refreshJobs();
  } catch (error) {
    handleError("提交文献搜索任务失败", error);
  } finally {
    setButtonLoading(button, false);
  }
}

async function requestMigrationJob(button) {
  if (state.selectedPapers.size === 0) {
    alert("请选择至少一篇论文再启动迁移任务。");
    return;
  }
  try {
    setButtonLoading(button, true, "启动中...");
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
          ? "已启动 1 个迁移任务。"
          : `已启动 ${jobs.length} 个迁移任务。`;
      alert(message);
    }
    await refreshJobs();
  } catch (error) {
    handleError("提交模型迁移任务失败", error);
  } finally {
    setButtonLoading(button, false);
  }
}

function exportMigrationPlan() {
  if (state.selectedPapers.size === 0) {
    alert("请选择至少一篇论文再导出迁移计划。");
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
    state.articles = Array.isArray(articlePayload?.items)
      ? articlePayload.items
      : [];
    state.migrationCatalog = normalizeMigrationItems(migrationPayload);
    state.migrationLookup = new Map(
      state.migrationCatalog
        .map((entry) => [getMigrationKey(entry), entry])
        .filter(([key]) => Boolean(key))
    );

    renderStages(stagePayload);
    renderArticles(state.articles);
    renderMigrationSelection();
    renderMigrationFeedback();
    performSearch();
  } catch (error) {
    handleError("加载仪表盘数据失败", error);
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

document.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
  refreshJobs();
  startJobPolling();

  const form = document.getElementById("search-form");
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const keywords = getCurrentKeywords();
      const submitBtn = form.querySelector('button[type="submit"]');
      if (keywords.length > 0) {
        await requestLiteratureJob(keywords, submitBtn);
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
});

const state = {
  status: null,
  commands: [],
  reports: [],
  configs: [],
  activeReport: null,
  selectedJob: null,
  chatMessages: [
    {
      role: "assistant",
      content: "你好，我是灵犀。你可以像聊天一样直接说现在的状态，我会结合短期情绪记忆回应。",
      emotion: "平静",
      mode: "welcome",
    },
  ],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtDate(ts) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false });
}

function fmtSize(bytes) {
  const n = Number(bytes || 0);
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

function toast(message) {
  const box = $("#toast");
  box.textContent = message;
  box.classList.add("show");
  setTimeout(() => box.classList.remove("show"), 2400);
}

async function api(path, options = {}) {
  const init = { ...options };
  if (init.body && typeof init.body !== "string") {
    init.body = JSON.stringify(init.body);
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
  }
  const res = await fetch(path, init);
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) throw new Error(data.error || text || `HTTP ${res.status}`);
  return data;
}

function setView(name) {
  document.body.dataset.activeView = name;
  $$(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === name));
  const label = document.querySelector(`.nav-item[data-view="${name}"] span:last-child`)?.textContent || "总览";
  $("#view-title").textContent = label;
  if (name === "reports" && state.reports.length && !state.activeReport) loadReport(state.reports.find((r) => r.exists)?.path);
  if (name === "jobs") refreshJobs();
}

function counter(label, value, detail = "") {
  return `<div class="counter"><span class="value">${esc(value)}</span><span class="label">${esc(label)}</span><p class="subtle">${esc(detail)}</p></div>`;
}

function renderGpuList(gpus) {
  const box = $("#gpu-list");
  if (!gpus.length) {
    box.innerHTML = `<div class="notice boundary">未读取到 nvidia-smi 输出</div>`;
    return;
  }
  box.innerHTML = gpus
    .map((gpu) => {
      const memPct = Math.round((gpu.memory_used / gpu.memory_total) * 100);
      return `
        <div class="gpu-card">
          <strong>GPU ${esc(gpu.index)} · ${esc(gpu.name)}</strong>
          <div class="meter-line"><span>显存</span><div class="meter"><span style="width:${memPct}%"></span></div><span>${memPct}%</span></div>
          <div class="meter-line"><span>利用率</span><div class="meter amber"><span style="width:${gpu.utilization}%"></span></div><span>${gpu.utilization}%</span></div>
          <div class="meter-line"><span>温度</span><div class="meter coral"><span style="width:${Math.min(gpu.temperature, 100)}%"></span></div><span>${gpu.temperature}°C</span></div>
        </div>`;
    })
    .join("");
}

function renderDatasetCounters(data) {
  $("#dataset-counters").innerHTML = [
    counter("SFT train", data.train?.lines ?? 0, data.train?.exists ? "已生成" : "缺失"),
    counter("SFT valid", data.valid?.lines ?? 0, data.valid?.exists ? "已生成" : "缺失"),
    counter("DPO train", data.dpo_train?.lines ?? 0, data.dpo_train?.exists ? "偏好数据" : "缺失"),
    counter("评测提示", data.eval_prompts?.lines ?? 0, "固定问答集"),
  ].join("");
}

function renderArtifactCounters(experiments, reports) {
  const adapters = experiments.filter((item) => item.adapter_exists).length;
  const metrics = experiments.filter((item) => item.metrics_exists).length;
  const reportCount = reports.filter((item) => item.exists).length;
  $("#artifact-counters").innerHTML = [
    counter("Adapter", adapters, `共 ${experiments.length} 项`),
    counter("Metrics", metrics, "训练指标"),
    counter("Reports", reportCount, "实验报告"),
    counter("Memory", state.status?.memory_count ?? 0, "短期记忆"),
  ].join("");
}

function renderExperimentTable(target, rows) {
  const head = `
    <thead><tr>
      <th>实验</th><th>Adapter</th><th>Train loss</th><th>Eval loss</th><th>PPL</th>
      <th>Reward acc</th><th>Step/s</th><th>Runtime</th>
    </tr></thead>`;
  const body = rows
    .map((row) => {
      const ok = row.adapter_exists ? `<span class="ok">存在</span>` : `<span class="bad">缺失</span>`;
      return `
        <tr>
          <td><strong>${esc(row.label)}</strong><br><span class="subtle">${esc(row.path)}</span></td>
          <td>${ok}</td>
          <td>${fmtNumber(row.train_loss)}</td>
          <td>${fmtNumber(row.eval_loss)}</td>
          <td>${fmtNumber(row.perplexity)}</td>
          <td>${row.reward_accuracy === undefined || row.reward_accuracy === null ? "-" : fmtNumber(row.reward_accuracy, 3)}</td>
          <td>${fmtNumber(row.step_per_second, 3)}</td>
          <td>${row.runtime ? `${Number(row.runtime).toFixed(1)}s` : "-"}</td>
        </tr>`;
    })
    .join("");
  target.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function renderMetricBars(rows) {
  const scored = rows.filter((row) => Number.isFinite(Number(row.eval_loss)));
  if (!scored.length) {
    $("#metric-bars").innerHTML = `<div class="notice">暂无训练指标</div>`;
    return;
  }
  const losses = scored.map((row) => Number(row.eval_loss));
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  $("#metric-bars").innerHTML = scored
    .map((row) => {
      const loss = Number(row.eval_loss);
      const pct = max === min ? 100 : Math.max(12, Math.round((1 - (loss - min) / (max - min)) * 100));
      return `
        <div class="bar-row">
          <b>${esc(row.label)}</b>
          <div class="meter"><span style="width:${pct}%"></span></div>
          <span>${loss.toFixed(4)}</span>
        </div>`;
    })
    .join("");
}

function drawPipeline() {
  const canvas = $("#pipeline-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const nodes = [
    { x: 80, y: 72, t: "基础模型", s: "Qwen2.5-1.5B" },
    { x: 270, y: 72, t: "SFT", s: "LoRA / QLoRA" },
    { x: 460, y: 72, t: "消融", s: "rank · steps" },
    { x: 650, y: 72, t: "增强", s: "NEFTune · rsLoRA" },
    { x: 820, y: 72, t: "DPO", s: "偏好对齐" },
  ];
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#10a37f";
  ctx.fillStyle = "#10a37f";
  for (let i = 0; i < nodes.length - 1; i++) {
    ctx.beginPath();
    ctx.moveTo(nodes[i].x + 70, nodes[i].y + 35);
    ctx.lineTo(nodes[i + 1].x - 70, nodes[i + 1].y + 35);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(nodes[i + 1].x - 76, nodes[i + 1].y + 28);
    ctx.lineTo(nodes[i + 1].x - 64, nodes[i + 1].y + 35);
    ctx.lineTo(nodes[i + 1].x - 76, nodes[i + 1].y + 42);
    ctx.fill();
  }
  nodes.forEach((node, index) => {
    ctx.fillStyle = index === 4 ? "#fff8e6" : "#f7f7f8";
    ctx.strokeStyle = index === 4 ? "#a16207" : "#d9d9e3";
    roundRect(ctx, node.x - 70, node.y, 140, 70, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#202123";
    ctx.font = "700 18px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(node.t, node.x, node.y + 30);
    ctx.font = "13px Inter, sans-serif";
    ctx.fillStyle = "#6e6e80";
    ctx.fillText(node.s, node.x, node.y + 52);
  });
  ctx.fillStyle = "#202123";
  ctx.font = "700 15px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("输出：领域问答对比、超参数分析、安全边界、记忆陪伴接口", 80, 188);
  ctx.strokeStyle = "#e5e5e5";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(80, 205);
  ctx.lineTo(820, 205);
  ctx.stroke();
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function renderMemory(records) {
  $("#memory-count").textContent = `${records.length} 条`;
  $("#memory-timeline").innerHTML =
    records
      .slice()
      .reverse()
      .map(
        (item) => `
      <div class="memory-item">
        <div>
          <div class="memory-round">第 ${esc(item.round)} 轮</div>
          <span class="tag">${esc(item.emotion)}</span>
        </div>
        <div class="memory-text">
          <div><strong>用户：</strong>${esc(item.user_text)}</div>
          <div><strong>回复：</strong>${esc(item.robot_reply)}</div>
        </div>
      </div>`
      )
      .join("") || `<div class="notice">暂无记忆</div>`;
  drawMemoryCanvas(records);
}

function emotionClass(emotion) {
  const map = {
    平静: "calm",
    焦虑: "anxious",
    悲伤: "sad",
    孤独: "lonely",
    愤怒: "angry",
    疲惫: "tired",
    开心: "happy",
    危机: "crisis",
  };
  return map[emotion] || "calm";
}

function renderChatThread() {
  const thread = $("#chat-thread");
  if (!thread) return;
  thread.innerHTML = state.chatMessages
    .map((message) => {
      const roleLabel = message.role === "user" ? "你" : message.mode === "dry" ? "Prompt" : "灵犀";
      const meta = [message.emotion, message.mode === "safety" ? "安全边界" : "", message.mode === "manual" ? "手动写入" : ""]
        .filter(Boolean)
        .join(" · ");
      const stages = renderProcessStages(message.stages || [], message.streaming);
      const diagnostics = renderDiagnostics(message.diagnostics || []);
      return `
        <div class="chat-message ${message.role} ${message.streaming ? "streaming" : ""}">
          <div class="avatar ${message.role}">${message.role === "user" ? "你" : "LX"}</div>
          <div class="bubble ${message.role}">
            <div class="bubble-meta">
              <strong>${esc(roleLabel)}</strong>
              ${meta ? `<span>${esc(meta)}</span>` : ""}
            </div>
            <div class="bubble-content">${formatChatContent(message.content)}</div>
            ${stages}
            ${diagnostics}
          </div>
        </div>`;
    })
    .join("");
  requestAnimationFrame(() => {
    thread.scrollTop = thread.scrollHeight;
  });
}

function renderProcessStages(stages, streaming) {
  if (!stages.length) return "";
  return `
    <div class="process-rail">
      ${stages
        .map(
          (stage, index) => `
            <div class="process-step ${index === stages.length - 1 && streaming ? "active" : "done"}">
              <span class="process-dot"></span>
              <span class="process-text"><b>${esc(stage.title || stage.key || "处理")}</b>${stage.detail ? ` · ${esc(stage.detail)}` : ""}</span>
            </div>`
        )
        .join("")}
    </div>`;
}

function renderDiagnostics(items) {
  if (!items.length) return "";
  return `
    <details class="diagnostics">
      <summary>推理诊断 ${items.length}</summary>
      <pre>${esc(items.slice(-12).join("\n"))}</pre>
    </details>`;
}

function autoResizeComposer() {
  const input = $("#chat-user");
  if (!input) return;
  const maxHeight = 128;
  input.style.height = "auto";
  const nextHeight = Math.min(input.scrollHeight, maxHeight);
  input.style.height = `${nextHeight}px`;
  input.style.overflowY = input.scrollHeight > maxHeight ? "auto" : "hidden";
}

function formatChatContent(text) {
  return esc(text || "")
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

function addChatMessage(message) {
  state.chatMessages.push(message);
  renderChatThread();
}

function setLastAssistantContent(content, patch = {}) {
  for (let i = state.chatMessages.length - 1; i >= 0; i -= 1) {
    if (state.chatMessages[i].role === "assistant") {
      state.chatMessages[i] = { ...state.chatMessages[i], content, ...patch };
      renderChatThread();
      return;
    }
  }
}

function patchLastAssistant(patch = {}) {
  for (let i = state.chatMessages.length - 1; i >= 0; i -= 1) {
    if (state.chatMessages[i].role === "assistant") {
      state.chatMessages[i] = { ...state.chatMessages[i], ...patch };
      renderChatThread();
      return state.chatMessages[i];
    }
  }
  return null;
}

function appendLastAssistantStage(stage) {
  for (let i = state.chatMessages.length - 1; i >= 0; i -= 1) {
    if (state.chatMessages[i].role === "assistant") {
      const stages = [...(state.chatMessages[i].stages || []), stage];
      state.chatMessages[i] = { ...state.chatMessages[i], stages };
      renderChatThread();
      return;
    }
  }
}

function appendLastAssistantDiagnostic(text) {
  for (let i = state.chatMessages.length - 1; i >= 0; i -= 1) {
    if (state.chatMessages[i].role === "assistant") {
      const diagnostics = [...(state.chatMessages[i].diagnostics || []), text].slice(-40);
      state.chatMessages[i] = { ...state.chatMessages[i], diagnostics };
      renderChatThread();
      return;
    }
  }
}

function setChatBusy(busy) {
  state.chatBusy = busy;
  ["#chat-user", "#show-prompt", "#manual-toggle", "#new-chat"].forEach((selector) => {
    const el = $(selector);
    if (el) el.disabled = busy;
  });
  $$("[data-chat-mode], #chat-form button[type='submit']").forEach((btn) => {
    btn.disabled = busy;
  });
}

function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  block.split(/\r?\n/).forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  });
  const dataText = dataLines.join("\n");
  let data = {};
  if (dataText) {
    try {
      data = JSON.parse(dataText);
    } catch {
      data = { raw: dataText };
    }
  }
  return { event, data };
}

async function streamApi(path, payload, onEvent) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  if (!res.body) throw new Error("当前浏览器不支持流式读取");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index = buffer.indexOf("\n\n");
    while (index >= 0) {
      const block = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 2);
      if (block) onEvent(parseSseBlock(block));
      index = buffer.indexOf("\n\n");
    }
  }
  const rest = buffer.trim();
  if (rest) onEvent(parseSseBlock(rest));
}

function drawMemoryCanvas(records) {
  const canvas = $("#memory-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const emotions = ["平静", "开心", "疲惫", "孤独", "悲伤", "焦虑", "愤怒", "危机"];
  const colors = {
    平静: "#168244",
    开心: "#c58a00",
    疲惫: "#8e8ea0",
    孤独: "#2f6f9f",
    悲伤: "#4f6fb3",
    焦虑: "#a16207",
    愤怒: "#d92d45",
    危机: "#9f1239",
  };
  const plot = records.slice(-12);
  const pad = 24;
  const graphW = w - pad * 2;
  const graphH = h - 58;
  ctx.strokeStyle = "#e5e5e5";
  ctx.lineWidth = 1;
  emotions.forEach((emotion, index) => {
    const y = pad + (graphH / (emotions.length - 1)) * index;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
    ctx.fillStyle = "#6e6e80";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(emotion, 8, y + 4);
  });

  if (!plot.length) {
    ctx.fillStyle = "#6e6e80";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("暂无记忆", w / 2, h / 2);
    return;
  }

  const points = plot.map((item, index) => {
    const emotion = item.emotion || "平静";
    const yIndex = Math.max(0, emotions.indexOf(emotion));
    return {
      x: pad + (graphW / Math.max(plot.length - 1, 1)) * index,
      y: pad + (graphH / (emotions.length - 1)) * yIndex,
      emotion,
      round: item.round,
    };
  });

  ctx.strokeStyle = "#10a37f";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  points.forEach((point) => {
    ctx.fillStyle = colors[point.emotion] || "#10a37f";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#202123";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`#${point.round}`, point.x, h - 18);
  });
}

function renderAdapterSelect(rows) {
  const select = $("#chat-adapter");
  select.innerHTML = `<option value="">基础模型</option>`;
  rows
    .filter((row) => row.adapter_exists)
    .forEach((row) => {
      const option = document.createElement("option");
      option.value = row.path;
      option.textContent = row.label;
      if (row.label === "SFT 正式版") option.selected = true;
      select.appendChild(option);
    });
}

function renderCommands(commands) {
  const grouped = commands;
  $("#command-grid").innerHTML = grouped
    .map(
      (cmd) => `
        <div class="command-card">
          <div>
            <span class="tag">${esc(cmd.kind)}</span>
            <h4>${esc(cmd.label)}</h4>
            <p>${esc(cmd.description)}</p>
          </div>
          <button class="button secondary" data-job="${esc(cmd.key)}"><span class="icon">▶</span><span>启动</span></button>
        </div>`
    )
    .join("");
}

function renderReports(reports) {
  state.reports = reports;
  $("#report-list").innerHTML = reports
    .map(
      (item) => `
        <button class="report-item ${state.activeReport === item.path ? "active" : ""}" data-report="${esc(item.path)}" ${item.exists ? "" : "disabled"}>
          <h4>${esc(item.title)}</h4>
          <p>${esc(item.path)} · ${item.exists ? fmtSize(item.size) : "缺失"}</p>
        </button>`
    )
    .join("");
}

async function loadReport(path) {
  if (!path) return;
  const data = await api(`/api/report?path=${encodeURIComponent(path)}`);
  state.activeReport = data.path;
  $("#report-title").textContent = data.content.split("\n")[0].replace(/^#+\s*/, "") || "报告内容";
  $("#report-path").textContent = data.path;
  $("#report-viewer").innerHTML = renderMarkdown(data.content);
  renderReports(state.reports);
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let inCode = false;
  let code = [];
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${inlineMarkdown(paragraph.join(" "))}</p>`;
      paragraph = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("```")) {
      if (inCode) {
        html += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
        code = [];
        inCode = false;
      } else {
        flushParagraph();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      continue;
    }
    if (line.startsWith("|") && lines[i + 1]?.startsWith("|") && lines[i + 1].includes("---")) {
      flushParagraph();
      const tableLines = [line];
      i += 2;
      while (i < lines.length && lines[i].startsWith("|")) {
        tableLines.push(lines[i]);
        i += 1;
      }
      i -= 1;
      html += tableToHtml(tableLines);
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
      continue;
    }
    if (line.startsWith("- ")) {
      flushParagraph();
      const items = [line.slice(2)];
      while (lines[i + 1]?.startsWith("- ")) {
        i += 1;
        items.push(lines[i].slice(2));
      }
      html += `<ul>${items.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`;
      continue;
    }
    paragraph.push(line.trim());
  }
  flushParagraph();
  return html;
}

function inlineMarkdown(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function tableToHtml(lines) {
  const rows = lines.map((line) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim())
  );
  const head = rows[0];
  const body = rows.slice(1);
  return `<div class="table-wrap"><table><thead><tr>${head.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${body
    .map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`)
    .join("")}</tbody></table></div>`;
}

function renderDataFiles(data) {
  $("#data-files").innerHTML = Object.entries(data)
    .map(
      ([key, item]) => `
        <div class="file-card">
          <strong>${esc(key)}</strong>
          <p>${esc(item.path)}</p>
          <p>${item.exists ? `<span class="ok">存在</span>` : `<span class="bad">缺失</span>`} · ${item.lines ?? 0} 行 · ${fmtSize(item.size)}</p>
        </div>`
    )
    .join("");
}

async function loadConfigs() {
  const data = await api("/api/configs");
  state.configs = data.configs;
  const select = $("#config-select");
  select.innerHTML = data.configs
    .map((item) => `<option value="${esc(item.path)}">${esc(item.path)}${item.exists ? "" : "（缺失）"}</option>`)
    .join("");
  if (data.configs.length) loadConfig(data.configs[0].path);
}

async function loadConfig(path) {
  const data = await api(`/api/config?path=${encodeURIComponent(path)}`);
  $("#config-viewer").textContent = data.content;
}

async function loadSample(path) {
  const data = await api(`/api/sample?path=${encodeURIComponent(path)}&limit=5`);
  $("#sample-viewer").textContent = JSON.stringify(data.items, null, 2);
}

async function refreshMemory() {
  const data = await api("/api/memory");
  renderMemory(data.records);
}

function chatPayload(mode) {
  return {
    mode,
    user_text: $("#chat-user").value.trim(),
    scene: $("#chat-scene").value,
    emotion: $("#chat-emotion").value,
    adapter: $("#chat-adapter").value,
    window: Number($("#chat-window").value || 3),
    manual_reply: $("#manual-reply").value,
    no_save: $("#chat-no-save").checked,
    disable_safety: $("#chat-disable-safety").checked,
  };
}

async function previewPrompt() {
  const payload = chatPayload("dry");
  if (!payload.user_text) {
    toast("请输入内容");
    return;
  }
  $("#safety-box").innerHTML = "";
  setChatBusy(true);
  $("#prompt-drawer").hidden = false;
  $("#chat-prompt").textContent = "正在构建 Prompt...";
  try {
    const data = await api("/api/chat", { method: "POST", body: payload });
    if (data.safety?.triggered) {
      $("#safety-box").innerHTML = `<div class="notice ${esc(data.safety.level)}"><strong>安全边界触发：</strong>${esc(
        data.safety.level
      )} · ${esc((data.safety.matched_keywords || []).join("、"))}</div>`;
    }
    $("#chat-prompt").textContent = data.prompt || "";
    toast("Prompt 已更新");
  } finally {
    setChatBusy(false);
  }
}

async function handleChat(mode) {
  const input = $("#chat-user");
  const userText = input.value.trim();
  if (!userText) {
    toast("请输入内容");
    return;
  }
  $("#safety-box").innerHTML = "";
  const payload = chatPayload(mode);
  addChatMessage({ role: "user", content: userText, emotion: payload.emotion || "自动识别" });
  addChatMessage({
    role: "assistant",
    content: mode === "dry" ? "正在构建 Prompt..." : mode === "manual" ? "正在写入手动回复..." : "正在准备推理...",
    emotion: "处理中",
    mode,
    streaming: true,
    stages: [],
    diagnostics: [],
  });

  let streamedReply = "";
  let finalData = null;
  setChatBusy(true);
  try {
    await streamApi("/api/chat/stream", payload, ({ event, data }) => {
      if (event === "stage") {
        appendLastAssistantStage(data);
        if (!streamedReply && mode !== "dry") {
          patchLastAssistant({ content: data.title || "正在处理..." });
        }
      } else if (event === "safety") {
        if (data.triggered) {
          $("#safety-box").innerHTML = `<div class="notice ${esc(data.level)}"><strong>安全边界触发：</strong>${esc(
            data.level
          )} · ${esc((data.matched_keywords || []).join("、"))}</div>`;
        }
      } else if (event === "prompt") {
        $("#chat-prompt").textContent = data.prompt || "";
        patchLastAssistant({ emotion: data.emotion || payload.emotion || "平静" });
        if (mode === "dry") {
          patchLastAssistant({ content: data.prompt || "", mode: "dry" });
          $("#prompt-drawer").hidden = false;
        }
      } else if (event === "token") {
        streamedReply += data.text || "";
        patchLastAssistant({ content: streamedReply || "正在生成回复..." });
      } else if (event === "diagnostic") {
        appendLastAssistantDiagnostic(data.text || data.raw || "");
      } else if (event === "final") {
        finalData = data;
        patchLastAssistant({
          content: data.reply || streamedReply || "",
          emotion: data.emotion || payload.emotion || "平静",
          mode: data.mode || mode,
          streaming: false,
        });
        renderMemory(data.records || []);
      } else if (event === "error") {
        throw new Error(data.error || data.raw || "流式推理失败");
      }
    });
  } finally {
    setChatBusy(false);
  }

  patchLastAssistant({ streaming: false });
  if (finalData && mode !== "dry") {
    input.value = "";
    autoResizeComposer();
  }
  toast(mode === "dry" ? "Prompt 已生成" : "对话完成");
}

async function startCommand(key) {
  const data = await api("/api/jobs", { method: "POST", body: { command: key } });
  state.selectedJob = data.job.id;
  toast(`已启动：${data.job.label}`);
  await refreshJobs();
  setView("jobs");
}

async function refreshJobs() {
  const data = await api("/api/jobs");
  renderJobs(data.jobs);
  if (!state.selectedJob && data.jobs.length) state.selectedJob = data.jobs[0].id;
  if (state.selectedJob) loadJobLog(state.selectedJob);
}

function renderJobs(jobs) {
  $("#job-list").innerHTML =
    jobs
      .map((job) => {
        const cls = job.status === "success" ? "ok" : job.status === "failed" ? "bad" : "warn";
        return `
          <div class="job-item">
            <div class="job-row">
              <strong>${esc(job.label)}</strong>
              <span class="${cls}">${esc(job.status)}</span>
            </div>
            <p>${esc(job.id)}</p>
            <p>${esc(job.command.join(" "))}</p>
            <div class="button-row" style="padding:0">
              <button class="button secondary" data-job-log="${esc(job.id)}"><span class="icon">▤</span><span>日志</span></button>
              ${job.status === "running" ? `<button class="button secondary" data-job-stop="${esc(job.id)}"><span class="icon">×</span><span>停止</span></button>` : ""}
            </div>
          </div>`;
      })
      .join("") || `<div class="notice">暂无任务</div>`;
}

async function loadJobLog(jobId) {
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/log`);
  state.selectedJob = jobId;
  $("#job-log-title").textContent = jobId;
  $("#job-log").textContent = data.log || "";
  $("#job-log").scrollTop = $("#job-log").scrollHeight;
}

async function stopJob(jobId) {
  await api(`/api/jobs/${encodeURIComponent(jobId)}/stop`, { method: "POST" });
  toast("停止信号已发送");
  refreshJobs();
}

async function refreshAll() {
  const [status, commands] = await Promise.all([api("/api/status"), api("/api/commands")]);
  state.status = status;
  state.commands = commands.commands;
  $("#server-time").textContent = fmtDate(status.time);
  renderGpuList(status.gpus || []);
  renderDatasetCounters(status.data || {});
  renderArtifactCounters(status.experiments || [], status.reports || []);
  renderMetricBars(status.experiments || []);
  renderExperimentTable($("#experiment-table"), status.experiments || []);
  renderExperimentTable($("#metrics-table"), status.experiments || []);
  $("#experiment-count").textContent = `${(status.experiments || []).length} 项`;
  renderAdapterSelect(status.experiments || []);
  renderCommands(commands.commands || []);
  renderReports(status.reports || []);
  renderDataFiles(status.data || {});
  renderMemory(status.latest_memory || []);
  renderChatThread();
  autoResizeComposer();
  drawPipeline();
}

function bindEvents() {
  document.addEventListener("click", async (event) => {
    const nav = event.target.closest("[data-view]");
    if (nav) setView(nav.dataset.view);

    const jump = event.target.closest("[data-view-jump]");
    if (jump) setView(jump.dataset.viewJump);

    const job = event.target.closest("[data-job]");
    if (job) {
      await startCommand(job.dataset.job);
    }

    const report = event.target.closest("[data-report]");
    if (report) {
      await loadReport(report.dataset.report);
    }

    const chat = event.target.closest("[data-chat-mode]");
    if (chat) {
      try {
        await handleChat(chat.dataset.chatMode);
      } catch (err) {
        setLastAssistantContent(err.message, { emotion: "错误", mode: "error" });
        toast("对话失败");
      }
    }

    const jobLog = event.target.closest("[data-job-log]");
    if (jobLog) loadJobLog(jobLog.dataset.jobLog);

    const jobStop = event.target.closest("[data-job-stop]");
    if (jobStop) stopJob(jobStop.dataset.jobStop);
  });

  $("#refresh-btn").addEventListener("click", () => refreshAll().then(() => toast("已刷新")));
  $("#refresh-jobs").addEventListener("click", () => refreshJobs());
  $("#config-select").addEventListener("change", (event) => loadConfig(event.target.value));
  $("#sample-select").addEventListener("change", (event) => loadSample(event.target.value));
  $("#chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await handleChat("generate");
    } catch (err) {
      setLastAssistantContent(err.message, { emotion: "错误", mode: "error" });
      toast("对话失败");
    }
  });
  $("#chat-user").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#chat-user").addEventListener("input", autoResizeComposer);
  $("#new-chat").addEventListener("click", () => {
    state.chatMessages = [
      {
        role: "assistant",
        content: "你好，我是灵犀。你可以像聊天一样直接说现在的状态，我会结合短期情绪记忆回应。",
        emotion: "平静",
        mode: "welcome",
      },
    ];
    renderChatThread();
    $("#chat-user").value = "";
    autoResizeComposer();
    $("#safety-box").innerHTML = "";
    $("#chat-prompt").textContent = "";
    $("#prompt-drawer").hidden = true;
    $("#manual-panel").hidden = true;
    toast("已开启新对话");
  });
  $("#manual-toggle").addEventListener("click", () => {
    $("#manual-panel").hidden = !$("#manual-panel").hidden;
  });
  $("#show-prompt").addEventListener("click", async () => {
    try {
      await previewPrompt();
    } catch (err) {
      toast(err.message);
    }
  });
  $("#close-prompt").addEventListener("click", () => {
    $("#prompt-drawer").hidden = true;
  });
  $("#reset-memory").addEventListener("click", async () => {
    await api("/api/memory", { method: "DELETE" });
    await refreshMemory();
    toast("记忆已清空");
  });
  window.addEventListener("resize", drawPipeline);
}

async function init() {
  bindEvents();
  await refreshAll();
  await loadConfigs();
  await loadSample($("#sample-select").value);
  const firstReport = state.reports.find((item) => item.exists);
  if (firstReport) await loadReport(firstReport.path);
  setInterval(() => {
    if ($("#jobs").classList.contains("active")) refreshJobs();
  }, 5000);
}

init().catch((err) => {
  console.error(err);
  toast(err.message);
});

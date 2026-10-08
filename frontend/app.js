// Filumart Product Knowledge Assistant - Frontend Controller
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const chatMessages = document.getElementById("chat-messages");
    const sendBtn = document.getElementById("send-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    const clearBtn = document.getElementById("clear-chat-btn");
    const errorBanner = document.getElementById("error-banner");
    const errorText = document.getElementById("error-text");
    const retryBtn = document.getElementById("retry-btn");

    const categorySelect = document.getElementById("filter-category");
    const brandSelect = document.getElementById("filter-brand");
    const supplierSelect = document.getElementById("filter-supplier");
    const countrySelect = document.getElementById("filter-country");

    let currentAbortController = null;
    let lastQuery = "";

    // 1. Load Filter Options from API
    async function loadFilters() {
        try {
            const res = await fetch("/filters");
            if (!res.ok) return;
            const data = await res.json();

            populateDropdown(categorySelect, data.categories);
            populateDropdown(brandSelect, data.brands);
            populateDropdown(supplierSelect, data.suppliers);
            populateDropdown(countrySelect, data.countries);
        } catch (err) {
            console.warn("Could not load filter dropdown options:", err);
        }
    }

    function populateDropdown(selectEl, items) {
        if (!selectEl || !items) return;
        const currentVal = selectEl.value;
        const defaultOption = selectEl.options[0].outerHTML;
        selectEl.innerHTML = defaultOption;
        items.forEach((item) => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = item;
            selectEl.appendChild(opt);
        });
        if (currentVal) selectEl.value = currentVal;
    }

    loadFilters();

    // 2. Clear / Reset Conversation
    clearBtn?.addEventListener("click", () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        hideError();
        chatMessages.innerHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    Conversation reset. Ask me technical specifications, compliance details, or product comparisons from our knowledge base.
                </div>
            </div>`;
        setGeneratingState(false);
    });

    // 3. Input Keydown Handling (Enter to submit, Shift+Enter for newline)
    queryInput?.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event("submit"));
        }
    });

    // 4. Cancel / Stop Streaming
    cancelBtn?.addEventListener("click", () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            setGeneratingState(false);
            appendStreamingIndicator(false);
        }
    });

    // 5. Retry Button
    retryBtn?.addEventListener("click", () => {
        if (lastQuery) {
            hideError();
            submitQuery(lastQuery);
        }
    });

    function setGeneratingState(isGenerating) {
        if (isGenerating) {
            sendBtn.classList.add("hidden");
            cancelBtn.classList.remove("hidden");
            queryInput.disabled = true;
        } else {
            sendBtn.classList.remove("hidden");
            cancelBtn.classList.add("hidden");
            queryInput.disabled = false;
            queryInput.focus();
        }
    }

    function showError(message) {
        if (errorBanner && errorText) {
            errorText.textContent = message;
            errorBanner.classList.remove("hidden");
        }
    }

    function hideError() {
        if (errorBanner) {
            errorBanner.classList.add("hidden");
        }
    }

    // 6. Form Submission
    form?.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        lastQuery = query;
        queryInput.value = "";
        hideError();
        submitQuery(query);
    });

    // 7. Core SSE Stream Runner
    async function submitQuery(query) {
        // Append user message bubble
        const userMsg = document.createElement("div");
        userMsg.className = "message user-message";
        userMsg.innerHTML = `<div class="message-content">${escapeHtml(query)}</div>`;
        chatMessages.appendChild(userMsg);
        scrollToBottom();

        // Create Assistant message placeholder
        const assistantMsg = document.createElement("div");
        assistantMsg.className = "message assistant-message";

        const contentEl = document.createElement("div");
        contentEl.className = "message-content markdown-body";
        contentEl.innerHTML = `<span class="typing-indicator">Retrieving grounded knowledge...</span>`;
        assistantMsg.appendChild(contentEl);

        const sourcesContainer = document.createElement("div");
        sourcesContainer.className = "sources-container hidden";
        assistantMsg.appendChild(sourcesContainer);

        chatMessages.appendChild(assistantMsg);
        scrollToBottom();

        setGeneratingState(true);
        currentAbortController = new AbortController();

        const filters = {
            category: categorySelect?.value || null,
            brand: brandSelect?.value || null,
            supplier: supplierSelect?.value || null,
            country: countrySelect?.value || null,
        };

        const payload = {
            query: query,
            filters: filters,
        };

        let rawAnswer = "";

        try {
            const response = await fetch("/ask/stream", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
                signal: currentAbortController.signal,
            });

            if (!response.ok) {
                let errorDetail = "Server error";
                try {
                    const errJson = await response.json();
                    errorDetail = errJson.error?.message || errJson.detail || response.statusText;
                } catch {
                    errorDetail = `HTTP ${response.status}: ${response.statusText}`;
                }
                throw new Error(errorDetail);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                // The last element is either empty or an incomplete event chunk
                buffer = lines.pop() || "";

                for (const frame of lines) {
                    const trimmed = frame.trim();
                    if (!trimmed || !trimmed.startsWith("data: ")) continue;

                    const jsonStr = trimmed.substring(6).trim();
                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.type === "token") {
                            rawAnswer += event.content;
                            contentEl.innerHTML = renderMarkdown(rawAnswer);
                            scrollToBottom();
                        } else if (event.type === "sources") {
                            renderSources(sourcesContainer, event.sources);
                            scrollToBottom();
                        } else if (event.type === "error") {
                            showError(event.message || "An error occurred during generation.");
                        } else if (event.type === "done") {
                            // Stream complete
                        }
                    } catch (parseErr) {
                        console.debug("Error parsing SSE chunk:", parseErr);
                    }
                }
            }

            if (!rawAnswer) {
                contentEl.innerHTML = renderMarkdown("The requested information is not available in the provided knowledge base.");
            }
        } catch (err) {
            if (err.name === "AbortError") {
                contentEl.innerHTML += `<div class="generation-aborted"><em>[Generation stopped by user]</em></div>`;
            } else {
                console.error("Stream error:", err);
                showError(err.message || "Failed to communicate with the assistant.");
                contentEl.innerHTML = `<span class="error-inline">Failed to retrieve response: ${escapeHtml(err.message)}</span>`;
            }
        } finally {
            setGeneratingState(false);
            currentAbortController = null;
            scrollToBottom();
        }
    }

    function renderSources(container, sources) {
        if (!container || !sources || sources.length === 0) {
            if (container) {
                container.classList.add("hidden");
                container.innerHTML = "";
            }
            return;
        }

        container.classList.remove("hidden");
        container.innerHTML = "";

        const collapsible = document.createElement("div");
        collapsible.className = "sources-collapsible";

        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "sources-toggle-btn";
        toggleBtn.setAttribute("aria-expanded", "false");
        toggleBtn.innerHTML = `
            <div class="sources-toggle-left">
                <span class="sources-toggle-icon">▶</span>
                <span class="sources-toggle-title">Documents Used (${sources.length})</span>
            </div>
            <span class="sources-toggle-hint">Click to expand</span>
        `;

        const contentDiv = document.createElement("div");
        contentDiv.className = "sources-content hidden";

        const cardsGrid = document.createElement("div");
        cardsGrid.className = "sources-grid";

        sources.forEach((s) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const pName = s.product_name || "General Specification";
            const pid = s.product_id ? `(${s.product_id})` : "";
            const isOcr = s.source_type === "ocr";
            const typeBadge = isOcr ? `<span class="badge badge-ocr">OCR</span>` : `<span class="badge badge-text">TEXT</span>`;
            const scoreBadge = s.score ? `<span class="source-score">${Math.round(s.score * 100)}% match</span>` : "";

            card.innerHTML = `
                <div class="source-header">
                    <strong>${escapeHtml(pName)} ${escapeHtml(pid)}</strong>
                    ${typeBadge}
                </div>
                <div class="source-meta">
                    <span>${escapeHtml(s.document)} &bull; Page ${s.page}</span>
                    <span class="chunk-id">${escapeHtml(s.chunk_id)}</span>
                    ${scoreBadge}
                </div>
            `;
            cardsGrid.appendChild(card);
        });

        contentDiv.appendChild(cardsGrid);

        // Click handler to toggle expansion
        toggleBtn.addEventListener("click", () => {
            const isCurrentlyHidden = contentDiv.classList.contains("hidden");
            if (isCurrentlyHidden) {
                contentDiv.classList.remove("hidden");
                toggleBtn.setAttribute("aria-expanded", "true");
                toggleBtn.classList.add("active");
                const icon = toggleBtn.querySelector(".sources-toggle-icon");
                const hint = toggleBtn.querySelector(".sources-toggle-hint");
                if (icon) icon.textContent = "▼";
                if (hint) hint.textContent = "Click to collapse";
                scrollToBottom();
            } else {
                contentDiv.classList.add("hidden");
                toggleBtn.setAttribute("aria-expanded", "false");
                toggleBtn.classList.remove("active");
                const icon = toggleBtn.querySelector(".sources-toggle-icon");
                const hint = toggleBtn.querySelector(".sources-toggle-hint");
                if (icon) icon.textContent = "▶";
                if (hint) hint.textContent = "Click to expand";
            }
        });

        collapsible.appendChild(toggleBtn);
        collapsible.appendChild(contentDiv);
        container.appendChild(collapsible);
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // 8. Lightweight, secure Client-side Markdown Renderer (incl. Tables & Citations)
    function renderMarkdown(md) {
        if (!md) return "";

        // First escape HTML to prevent XSS
        let html = escapeHtml(md);

        // Code blocks: ```...```
        html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
            return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Markdown Tables
        html = html.replace(/((?:\|[^\n]+\|\r?\n?){2,})/g, (tableText) => {
            const rows = tableText.trim().split("\n");
            if (rows.length < 2) return tableText;

            let tableHtml = '<div class="table-responsive"><table class="spec-table">';
            let isHeader = true;

            for (let i = 0; i < rows.length; i++) {
                const row = rows[i].trim();
                if (!row.startsWith("|") || !row.endsWith("|")) continue;

                // Check for divider row (|---|---|)
                if (row.match(/^\|[\s\-:|]+\|$/)) {
                    isHeader = false;
                    continue;
                }

                const cells = row
                    .split("|")
                    .slice(1, -1)
                    .map((c) => c.trim());

                if (isHeader) {
                    tableHtml += "<thead><tr>";
                    cells.forEach((c) => (tableHtml += `<th>${c}</th>`));
                    tableHtml += "</tr></thead><tbody>";
                    isHeader = false;
                } else {
                    tableHtml += "<tr>";
                    cells.forEach((c) => {
                        // Highlight 'Not documented'
                        if (c.toLowerCase().includes("not documented")) {
                            tableHtml += `<td class="not-documented"><em>${c}</em></td>`;
                        } else {
                            tableHtml += `<td>${c}</td>`;
                        }
                    });
                    tableHtml += "</tr>";
                }
            }
            tableHtml += "</tbody></table></div>";
            return tableHtml;
        });

        // Headers
        html = html.replace(/^### (.*$)/gim, "<h4>$1</h4>");
        html = html.replace(/^## (.*$)/gim, "<h3>$1</h3>");
        html = html.replace(/^# (.*$)/gim, "<h2>$1</h2>");

        // Bold and Italic
        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

        // Inline Chunk Citations: [P001_01] or [P001_SOLARMAX55_P2_01]
        html = html.replace(/\[([A-Z0-9_\-]+(?:_P\d+_\d{2})?)\]/g, (match, id) => {
            return `<span class="inline-citation" title="Source Chunk: ${id}">${match}</span>`;
        });

        // Bullet lists
        html = html.replace(/^\s*[-*]\s+(.*)$/gim, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>)/gim, "<ul>$1</ul>");
        // Clean up nested <ul> tags from regex
        html = html.replace(/<\/ul>\s*<ul>/g, "");

        // Convert double newlines to paragraph breaks
        html = html.replace(/\n\n+/g, "<br><br>");

        return html;
    }
});

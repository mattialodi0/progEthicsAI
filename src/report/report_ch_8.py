"""Renderer for chapter 8: caption and semantic associations."""

from __future__ import annotations

import json
from typing import Any


def render_chapter_eight(chapter_data: dict[str, Any]) -> str:
    """Render precomputed TF-IDF caption associations."""
    serialized = json.dumps(chapter_data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
    <div id="chapter-container">
  <div class="eyebrow">Chapter 8</div>
  <h2 id="chapter-8">Caption and semantic association analysis</h2>
  <p class="subtitle">Caption vocabulary and TF-IDF rankings reveal terms that
  distinguish prompt values and detected attributes. Rankings use the complete
  caption vocabulary and exclude groups below the configured minimum size.</p>
  <section id="chapter-8-prompt" aria-labelledby="chapter-8-prompt-heading">
    <h3 id="chapter-8-prompt-heading">8.1 Top TF-IDF words by prompt value</h3>
    <p>For each prompt variable, the five highest-scoring words are shown for
    every variable value. TF-IDF is calculated across the values of that
    variable, so a term is high when it is frequent in one value and uncommon
    in the others.</p>
    <label>Prompt variable <select id="chapter-8-prompt-select"></select></label>
    <div id="chapter-8-prompt-cards" class="plot-grid"></div>
  </section>
  <section id="chapter-8-attributes" aria-labelledby="chapter-8-attributes-heading">
    <h3 id="chapter-8-attributes-heading">8.2 Top TF-IDF words by detected attribute</h3>
    <p>Detected groups are compared within each prompt-variable value. Select a
    prompt variable to account for its composition before comparing detected
    gender, race, age, or emotion groups.</p>
    <label>Prompt variable <select id="chapter-8-conditioning-select"></select></label>
    <label>Detected attribute <select id="chapter-8-detected-select"></select></label>
    <div id="chapter-8-attribute-cards" class="plot-grid"></div>
  </section>
  <section id="chapter-8-vocabulary" aria-labelledby="chapter-8-vocabulary-heading">
    <h3 id="chapter-8-vocabulary-heading">Caption vocabulary</h3>
    <p><span id="chapter-8-vocabulary-size"></span> unique normalized words were
    extracted with NLTK. The vocabulary is retained in the analysis artifact
    for reproducibility.</p>
    <details><summary>Show vocabulary</summary>
      <p id="chapter-8-vocabulary-list"></p>
    </details>
  </section>
  <script>
    const chapterEightData = {serialized};
    const chapterEightSafe = (value) => String(value).replace(/[&<>"']/g,
      (character) => ({{"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}})[character]);
    const chapterEightTitle = (value) => String(value).replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
    const chapterEightCard = (title, observations, terms) =>
      `<article class="plot-card"><h4>${{chapterEightSafe(title)}}</h4>
       <p class="note">${{observations}} observations</p><ol>` +
      (terms || []).map((item) => `<li><strong>${{chapterEightSafe(item.term)}}</strong>
        <span class="note">TF-IDF ${{Number(item.score).toFixed(3)}} ·
        document frequency ${{item.document_frequency}}</span></li>`).join("") +
      `</ol></article>`;

    const promptData = chapterEightData.prompt_associations || {{}};
    const promptSelect = document.getElementById("chapter-8-prompt-select");
    promptSelect.innerHTML = Object.keys(promptData).map((key) =>
      `<option value="${{key}}">${{chapterEightTitle(key)}}</option>`).join("");
    const renderPrompt = () => {{
      const data = promptData[promptSelect.value];
      document.getElementById("chapter-8-prompt-cards").innerHTML = data ?
        Object.entries(data).map(([value, item]) =>
          chapterEightCard(value, item.observations, item.top_terms)).join("") :
        '<p class="note">No prompt associations are available.</p>';
    }};
    promptSelect.addEventListener("change", renderPrompt);
    renderPrompt();

    const attributeData = chapterEightData.detected_associations || {{}};
    const conditioningSelect = document.getElementById("chapter-8-conditioning-select");
    const detectedSelect = document.getElementById("chapter-8-detected-select");
    conditioningSelect.innerHTML = Object.keys(attributeData).map((key) =>
      `<option value="${{key}}">${{chapterEightTitle(key)}}</option>`).join("");
    const renderDetectedOptions = () => {{
      const values = attributeData[conditioningSelect.value] || {{}};
      detectedSelect.innerHTML = Object.keys(values).map((key) =>
        `<option value="${{key}}">${{chapterEightTitle(key)}}</option>`).join("");
      renderDetected();
    }};
    const renderDetected = () => {{
      const data = (attributeData[conditioningSelect.value] || {{}})
        [detectedSelect.value] || {{}};
      const cards = [];
      Object.entries(data).forEach(([promptValue, groups]) =>
        Object.entries(groups).forEach(([detectedValue, item]) =>
          cards.push(chapterEightCard(
            `${{promptValue}} · detected ${{detectedValue}}`,
            item.observations, item.top_terms))));
      document.getElementById("chapter-8-attribute-cards").innerHTML = cards.join("") ||
        '<p class="note">No conditioned detected-attribute associations are available.</p>';
    }};
    conditioningSelect.addEventListener("change", renderDetectedOptions);
    detectedSelect.addEventListener("change", renderDetected);
    renderDetectedOptions();

    document.getElementById("chapter-8-vocabulary-size").textContent =
      chapterEightData.vocabulary.length;
    document.getElementById("chapter-8-vocabulary-list").textContent =
      chapterEightData.vocabulary.join(", ");
  </script>
  </div>
"""

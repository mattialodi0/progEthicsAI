    const chapterEightSafe = (value) => String(value).replace(/[&<>"']/g,
      (character) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[character]);
    const chapterEightTitle = (value) => String(value).replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
    const chapterEightCategories = chapterEightData.word_categories ||
      {adjectives: "Adjectives", verbs: "Verbs / actions", nouns: "Names / objects"};
    const chapterEightColors = {adjectives: "#2a78d6", verbs: "#eb6834", nouns: "#1baf7a"};
    const chapterEightPlotConfig = {responsive: true, displaylogo: false};
    const chapterEightFormat = (value, digits = 3) =>
      value === null || value === undefined ? "–" : Number(value).toFixed(digits);
    const chapterEightPercent = (value) =>
      value === null || value === undefined ? "–" : `${(100 * value).toFixed(0)}%`;
    document.getElementById("chapter-8-min-group").textContent =
      chapterEightData.min_group_size ?? 2;

    // ---- headline metrics -------------------------------------------------
    const chapterEightStop = chapterEightData.stopwords || {};
    const chapterEightSizes = chapterEightData.vocabulary_size || {};
    const chapterEightOverall = chapterEightData.overall_sentiment || {};
    const chapterEightMetric = (label, value) => `<div class="metric">
      <span class="metric-label">${label}</span><span class="metric-value">${value}</span></div>`;
    document.getElementById("chapter-8-metrics").innerHTML = [
      chapterEightMetric("Captions", chapterEightData.n_captions ?? "–"),
      chapterEightMetric("Stopword tokens removed", chapterEightStop.total_tokens ?
        chapterEightPercent(chapterEightStop.removed_tokens / chapterEightStop.total_tokens) : "–"),
      ...Object.entries(chapterEightCategories).map(([key, label]) =>
        chapterEightMetric(`Unique ${label.toLowerCase()}`, chapterEightSizes[key] ?? "–")),
      chapterEightMetric("Mean caption sentiment", chapterEightFormat(chapterEightOverall.mean, 2)),
    ].join("");

    // ---- 8.1 / 8.2 TF-IDF cards -----------------------------------------
    // Words seen fewer than this many times score high on TF-IDF by accident
    // (one caption is enough), so they are hidden from the cards.
    const chapterEightMinCount = 3;
    const chapterEightCard = (title, observations, terms) => {
      const columns = Object.entries(chapterEightCategories).map(([key, label]) => {
        const items = ((terms || {})[key] || [])
          .filter((item) => Number(item.frequency) >= chapterEightMinCount);
        const top = Math.max(1, ...items.map((item) => Number(item.frequency)));
        return `<div><h5>${label}</h5>` + (items.length ? `<ol>` + items.map((item) =>
          `<li title="TF-IDF ${Number(item.score).toFixed(4)}, ${item.frequency} occurrences">
           <strong>${chapterEightSafe(item.term)}</strong>
           <span class="term-count"><span class="term-bar" style="width:${(60 * item.frequency / top).toFixed(0)}%"></span>
           <span class="note">${item.frequency}×</span></span></li>`
          ).join("") + `</ol>` : `<p class="note">No distinctive word used ${chapterEightMinCount}+ times.</p>`) + `</div>`;
      }).join("");
      return `<article class="plot-card"><h4>${chapterEightSafe(title)}</h4>
        <p class="note">${observations} observations · ranked by TF-IDF · bar = occurrences</p>
        <div class="ch8-terms">${columns}</div></article>`;
    };

    const promptData = chapterEightData.prompt_associations || {};
    const promptSelect = document.getElementById("chapter-8-prompt-select");
    promptSelect.innerHTML = Object.keys(promptData).map((key) =>
      `<option value="${key}">${chapterEightTitle(key)}</option>`).join("");
    const renderPrompt = () => {
      const data = promptData[promptSelect.value];
      document.getElementById("chapter-8-prompt-cards").innerHTML =
        data && Object.keys(data).length ?
        Object.entries(data).map(([value, item]) =>
          chapterEightCard(value, item.observations, item.top_terms)).join("") :
        '<p class="note">No prompt associations are available.</p>';
    };
    promptSelect.addEventListener("change", renderPrompt);
    renderPrompt();

    const attributeData = chapterEightData.detected_associations || {};
    const conditioningSelect = document.getElementById("chapter-8-conditioning-select");
    const detectedSelect = document.getElementById("chapter-8-detected-select");
    conditioningSelect.innerHTML = Object.keys(attributeData).map((key) =>
      `<option value="${key}">${chapterEightTitle(key)}</option>`).join("");
    const renderDetected = () => {
      const data = (attributeData[conditioningSelect.value] || {})
        [detectedSelect.value] || {};
      const cards = [];
      Object.entries(data).forEach(([promptValue, groups]) =>
        Object.entries(groups).forEach(([detectedValue, item]) =>
          cards.push(chapterEightCard(
            `${promptValue} · detected ${detectedValue}`,
            item.observations, item.top_terms))));
      document.getElementById("chapter-8-attribute-cards").innerHTML = cards.join("") ||
        '<p class="note">No conditioned detected-attribute associations are available.</p>';
    };
    const renderDetectedOptions = () => {
      const values = attributeData[conditioningSelect.value] || {};
      detectedSelect.innerHTML = Object.keys(values).map((key) =>
        `<option value="${key}">${chapterEightTitle(key)}</option>`).join("");
      renderDetected();
    };
    conditioningSelect.addEventListener("change", renderDetectedOptions);
    detectedSelect.addEventListener("change", renderDetected);
    renderDetectedOptions();

    // ---- shared attribute selector for 8.3 / 8.4 -------------------------
    const lexicalProfiles = chapterEightData.lexical_profiles || {prompt: {}, detected: {}};
    const chapterEightAttributeOptions = ["prompt", "detected"].map((source) =>
      `<optgroup label="${source === "prompt" ? "Prompt attribute" : "Detected attribute"}">` +
      Object.keys(lexicalProfiles[source] || {}).map((key) =>
        `<option value="${source}|${key}">${chapterEightTitle(key)}` +
        `${source === "detected" ? " (detected)" : ""}</option>`).join("") +
      `</optgroup>`).join("");
    const chapterEightProfile = (value) => {
      const [source, attribute] = String(value).split("|");
      return ((lexicalProfiles[source] || {})[attribute]) || {groups: {}, rarefaction_depth: {}};
    };
    const chapterEightAttributeLabel = (value) => {
      const [source, attribute] = String(value).split("|");
      return `${source === "detected" ? "detected " : "prompt "}${String(attribute).replaceAll("_", " ")}`;
    };

    // ---- 8.3 sentiment ---------------------------------------------------
    const sentimentAttribute = document.getElementById("chapter-8-sentiment-attribute");
    const sentimentCategory = document.getElementById("chapter-8-sentiment-category");
    sentimentAttribute.innerHTML = chapterEightAttributeOptions;
    sentimentAttribute.value = "prompt|gender";
    sentimentCategory.innerHTML = `<option value="all">All descriptive words</option>` +
      Object.entries(chapterEightCategories).map(([key, label]) =>
        `<option value="${key}">${label}</option>`).join("");
    const renderSentiment = () => {
      const profile = chapterEightProfile(sentimentAttribute.value);
      const category = sentimentCategory.value;
      const entries = Object.entries(profile.groups || {})
        .filter(([, item]) => item.sentiment[category].mean !== null);
      const plot = document.getElementById("chapter-8-sentiment-plot");
      if (!entries.length) {
        Plotly.purge(plot);
        plot.innerHTML = '<p class="note">No sentiment scores are available for this selection.</p>';
        document.getElementById("chapter-8-sentiment-table").innerHTML = "";
        return;
      }
      plot.innerHTML = "";
      const stats = entries.map(([, item]) => item.sentiment[category]);
      Plotly.newPlot(plot, [{
        type: "scatter", mode: "markers", name: "Mean sentiment",
        x: entries.map(([group]) => group),
        y: stats.map((s) => s.mean),
        error_y: {type: "data", symmetric: false, thickness: 2, width: 6, color: "#2a78d6",
          array: stats.map((s) => s.ci_high - s.mean),
          arrayminus: stats.map((s) => s.mean - s.ci_low)},
        marker: {size: 11, color: "#2a78d6", line: {color: "#fff", width: 2}},
        customdata: stats.map((s) => [s.n, s.ci_low, s.ci_high,
          100 * s.positive_share, 100 * s.negative_share]),
        hovertemplate: "<b>%{x}</b><br>Mean compound: %{y:.3f}" +
          "<br>95% CI: %{customdata[1]:.3f} to %{customdata[2]:.3f}" +
          "<br>Positive: %{customdata[3]:.0f}% · Negative: %{customdata[4]:.0f}%" +
          "<br>Images: %{customdata[0]}<extra></extra>"
      }], {
        title: `Sentiment of ${sentimentCategory.selectedOptions[0].text.toLowerCase()} ` +
          `by ${chapterEightAttributeLabel(sentimentAttribute.value)}`,
        height: 380, template: "plotly_white", showlegend: false,
        margin: {l: 60, r: 20, t: 60, b: 90},
        xaxis: {title: "", automargin: true},
        yaxis: {title: "Mean VADER compound score", zeroline: true,
          zerolinecolor: "#829ab1", zerolinewidth: 1.5}
      }, chapterEightPlotConfig);
      const words = (list) => (list || []).map((w) =>
        `${chapterEightSafe(w.term)} (${w.frequency})`).join(", ") || "–";
      document.getElementById("chapter-8-sentiment-table").innerHTML =
        `<table><thead><tr><th>Group</th><th>Images</th><th>Mean</th><th>95% CI</th>
         <th>Positive</th><th>Negative</th><th>Most frequent positive words</th>
         <th>Most frequent negative words</th></tr></thead><tbody>` +
        entries.map(([group, item]) => {
          const s = item.sentiment[category];
          return `<tr><td>${chapterEightSafe(group)}</td><td>${s.n}</td>
            <td>${chapterEightFormat(s.mean)}</td>
            <td>${chapterEightFormat(s.ci_low)} to ${chapterEightFormat(s.ci_high)}</td>
            <td>${chapterEightPercent(s.positive_share)}</td>
            <td>${chapterEightPercent(s.negative_share)}</td>
            <td class="ch8-words">${words(item.sentiment_words.positive)}</td>
            <td class="ch8-words">${words(item.sentiment_words.negative)}</td></tr>`;
        }).join("") + `</tbody></table>
        <p class="note">Sentiment words are counted over all descriptive words of the
        group, whichever word category is selected above.</p>`;
    };
    const neutralWords = chapterEightData.sentiment_neutral_words || [];
    document.getElementById("chapter-8-neutral-count").textContent = neutralWords.length;
    document.getElementById("chapter-8-neutral-list").textContent = neutralWords.join(", ");
    sentimentAttribute.addEventListener("change", renderSentiment);
    sentimentCategory.addEventListener("change", renderSentiment);
    renderSentiment();

    // ---- 8.4 diversity ---------------------------------------------------
    const diversityAttribute = document.getElementById("chapter-8-diversity-attribute");
    const diversityMetric = document.getElementById("chapter-8-diversity-metric");
    diversityAttribute.innerHTML = chapterEightAttributeOptions;
    diversityAttribute.value = "prompt|gender";
    const renderDiversity = () => {
      const profile = chapterEightProfile(diversityAttribute.value);
      const metric = diversityMetric.value;
      const entries = Object.entries(profile.groups || {});
      const plot = document.getElementById("chapter-8-diversity-plot");
      if (!entries.length) {
        Plotly.purge(plot);
        plot.innerHTML = '<p class="note">No diversity measurements are available for this selection.</p>';
        document.getElementById("chapter-8-diversity-table").innerHTML = "";
        return;
      }
      plot.innerHTML = "";
      const depth = profile.rarefaction_depth || {};
      const traces = Object.entries(chapterEightCategories).map(([key, label]) => ({
        type: "bar",
        name: metric === "rarefied_types" ? `${label} (sample of ${depth[key]} words)` : label,
        x: entries.map(([group]) => group),
        y: entries.map(([, item]) => item.diversity[key][metric]),
        marker: {color: chapterEightColors[key], line: {color: "#fff", width: 2}},
        customdata: entries.map(([, item]) => [item.diversity[key].tokens,
          item.diversity[key].types, item.observations]),
        hovertemplate: `<b>%{x}</b> · ${label}<br>${diversityMetric.selectedOptions[0].text}: ` +
          "%{y:.3f}<br>Words: %{customdata[0]} · Unique: %{customdata[1]}" +
          "<br>Images: %{customdata[2]}<extra></extra>"
      }));
      Plotly.newPlot(plot, traces, {
        title: `${diversityMetric.selectedOptions[0].text} by ` +
          `${chapterEightAttributeLabel(diversityAttribute.value)}`,
        barmode: "group", bargap: .25, bargroupgap: .05,
        height: 400, template: "plotly_white",
        margin: {l: 60, r: 20, t: 60, b: 90},
        legend: {orientation: "h", y: -0.25},
        xaxis: {title: "", automargin: true},
        yaxis: {title: diversityMetric.selectedOptions[0].text,
          range: metric === "rarefied_types" ? null : [0, 1]}
      }, chapterEightPlotConfig);
      const header = Object.values(chapterEightCategories).map((label) =>
        `<th colspan="4">${label}</th>`).join("");
      const subheader = Object.keys(chapterEightCategories).map(() =>
        "<th>Words</th><th>Unique</th><th>Rarefied</th><th>Simpson</th>").join("");
      document.getElementById("chapter-8-diversity-table").innerHTML =
        `<table><thead><tr><th rowspan="2">Group</th><th rowspan="2">Images</th>${header}</tr>
         <tr>${subheader}</tr></thead><tbody>` +
        entries.map(([group, item]) => `<tr><td>${chapterEightSafe(group)}</td>
          <td>${item.observations}</td>` +
          Object.keys(chapterEightCategories).map((key) => {
            const d = item.diversity[key];
            return `<td>${d.tokens}</td><td>${d.types}</td>
              <td>${chapterEightFormat(d.rarefied_types, 1)}</td>
              <td>${chapterEightFormat(d.simpson)}</td>`;
          }).join("") + `</tr>`).join("") + `</tbody></table>`;
    };
    diversityAttribute.addEventListener("change", renderDiversity);
    diversityMetric.addEventListener("change", renderDiversity);
    renderDiversity();

    // ---- 8.5 stopwords and vocabulary -------------------------------------
    const nltkStopwords = chapterEightStop.nltk || [];
    const domainStopwords = chapterEightStop.domain || [];
    const removedShare = chapterEightStop.total_tokens ?
      chapterEightStop.removed_tokens / chapterEightStop.total_tokens : null;
    document.getElementById("chapter-8-stopword-summary").innerHTML =
      `Of ${chapterEightStop.total_tokens ?? "–"} word tokens in the captions,
       <strong>${chapterEightStop.removed_tokens ?? "–"}</strong>
       (${chapterEightPercent(removedShare)}) were removed as stopwords:
       ${nltkStopwords.length} NLTK English stopwords plus ${domainStopwords.length}
       captioning stopwords. A further ${chapterEightStop.other_pos_tokens ?? "–"} tokens
       were neither adjectives, verbs nor nouns (mostly adverbs and numbers) and are
       not used. The chart shows the most frequently removed stopwords.`;
    const mostRemoved = (chapterEightStop.most_removed || []).slice(0, 20).reverse();
    const stopwordPlot = document.getElementById("chapter-8-stopword-plot");
    if (mostRemoved.length) {
      Plotly.newPlot(stopwordPlot, [{
        type: "bar", orientation: "h", name: "Occurrences",
        x: mostRemoved.map((item) => item.frequency),
        y: mostRemoved.map((item) => item.term),
        marker: {color: mostRemoved.map((item) =>
          domainStopwords.includes(item.term) ? "#eb6834" : "#2a78d6")},
        customdata: mostRemoved.map((item) =>
          domainStopwords.includes(item.term) ? "Captioning stopword" : "NLTK stopword"),
        hovertemplate: "<b>%{y}</b><br>%{customdata}<br>Occurrences: %{x}<extra></extra>"
      }], {
        title: "Most frequently removed stopwords (blue: NLTK, orange: captioning)",
        height: 520, template: "plotly_white", showlegend: false,
        margin: {l: 90, r: 20, t: 60, b: 50},
        xaxis: {title: "Occurrences in captions"}, yaxis: {automargin: true}
      }, chapterEightPlotConfig);
    } else {
      stopwordPlot.innerHTML = '<p class="note">No stopword counts are available.</p>';
    }
    document.getElementById("chapter-8-domain-count").textContent = domainStopwords.length;
    document.getElementById("chapter-8-domain-list").textContent = domainStopwords.join(", ");
    document.getElementById("chapter-8-nltk-count").textContent = nltkStopwords.length;
    document.getElementById("chapter-8-nltk-list").textContent = nltkStopwords.join(", ");
    const chapterEightVocabulary = chapterEightData.vocabulary || {};
    document.getElementById("chapter-8-vocabulary-lists").innerHTML =
      Object.entries(chapterEightCategories).map(([key, label]) => {
        const words = Array.isArray(chapterEightVocabulary) ? [] : (chapterEightVocabulary[key] || []);
        return `<details><summary>${label} vocabulary (${words.length} lemmas)</summary>
          <p class="ch8-wordlist">${chapterEightSafe(words.join(", "))}</p></details>`;
      }).join("");

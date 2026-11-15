  // Show / hide the reading-suggestion call-outs (How to read, What to look
  // for, Caveats). Visible by default; the choice is remembered per browser.
  (() => {
    const button = document.getElementById("suggestion-toggle");
    if (!button) return;
    const apply = (visible) => {
      document.body.classList.toggle("hide-suggestions", !visible);
      button.setAttribute("aria-pressed", String(visible));
      button.querySelector(".toggle-state").textContent = visible ? "shown" : "hidden";
    };
    let visible = true;
    try { visible = localStorage.getItem("report-suggestions") !== "hidden"; } catch (e) {}
    apply(visible);
    button.addEventListener("click", () => {
      visible = !visible;
      apply(visible);
      try { localStorage.setItem("report-suggestions", visible ? "shown" : "hidden"); } catch (e) {}
    });
  })();
  // Highlight the sidebar entry of the section currently in view. Only
  // elements that have a sidebar link are observed, so plot containers whose
  // ids also start with "chapter-" cannot clear the highlight.
  const sectionLinks = new Map(Array.from(
    document.querySelectorAll("[data-section-link]")
  ).map((link) => [link.dataset.sectionLink, link]));
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        sectionLinks.forEach((link) => link.classList.remove("active"));
        sectionLinks.get(entry.target.id)?.classList.add("active");
      }
    });
  }, { rootMargin: "-15% 0px -70% 0px" });
  sectionLinks.forEach((link, id) => {
    const section = document.getElementById(id);
    if (section) sectionObserver.observe(section);
  });
  // Charts drawn inside a closed <details> get a default width; resize them
  // to their container when the block is opened.
  document.addEventListener("toggle", (event) => {
    if (event.target.open) {
      event.target.querySelectorAll(".js-plotly-plot").forEach((plot) => Plotly.Plots.resize(plot));
    }
  }, true);

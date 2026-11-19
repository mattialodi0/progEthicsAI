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
  // Only the chapter in view shows its section links in the sidebar.
  const openGroup = (link) => {
    const group = link?.closest(".nav-group");
    document.querySelectorAll(".nav-group.open").forEach((item) => {
      if (item !== group) item.classList.remove("open");
    });
    group?.classList.add("open");
  };
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        sectionLinks.forEach((link) => link.classList.remove("active"));
        const link = sectionLinks.get(entry.target.id);
        link?.classList.add("active");
        if (link) {
          openGroup(link);
          // Keep the active link visible inside the sidebar's own scroll area
          // (scrollIntoView would also scroll the page on narrow screens).
          const nav = link.closest("nav");
          if (nav && nav.scrollHeight > nav.clientHeight) {
            const top = link.offsetTop - nav.offsetTop;
            if (top < nav.scrollTop || top > nav.scrollTop + nav.clientHeight - 40) {
              nav.scrollTop = Math.max(0, top - nav.clientHeight / 3);
            }
          }
        }
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
  // Reading progress bar and "back to top" link.
  (() => {
    const bar = document.getElementById("read-progress");
    const top = document.getElementById("to-top");
    let queued = false;
    const update = () => {
      queued = false;
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      const share = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      if (bar) bar.style.width = (100 * share).toFixed(2) + "%";
      if (top) top.classList.toggle("visible", window.scrollY > 900);
    };
    window.addEventListener("scroll", () => {
      if (!queued) { queued = true; requestAnimationFrame(update); }
    }, {passive: true});
    update();
  })();
  // Printing: open every collapsed block and draw every chart that has not
  // been drawn yet, then restore the collapsed blocks afterwards.
  (() => {
    let reopened = [];
    window.addEventListener("beforeprint", () => {
      reopened = Array.from(document.querySelectorAll("details:not([open])"));
      reopened.forEach((item) => { item.open = true; });
      if (window.reportFlushCharts) window.reportFlushCharts();
      document.querySelectorAll(".js-plotly-plot").forEach((plot) => {
        try { Plotly.Plots.resize(plot); } catch (e) {}
      });
    });
    window.addEventListener("afterprint", () => {
      reopened.forEach((item) => { item.open = false; });
      reopened = [];
    });
  })();

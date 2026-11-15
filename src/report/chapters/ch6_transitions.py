"""Chapter 6: interactive before/after view of detected values in matched pairs.

The widget is generated per pair source (control or double-variable prompts):
``prefix`` namespaces its element ids and JavaScript variables, and
``double`` switches between one fixed attribute and several.
"""

from __future__ import annotations

from typing import Any

from ..static_files import script_json


def transition_selector(
    data: dict[str, Any],
    *,
    prefix: str,
    description: str,
    double: bool = False,
) -> str:
    serialized = script_json(data)
    return f"""
            <p>{description}</p>
            <div class="checkbox-grid" aria-label="{prefix} selectors">
              <label>Changing attribute
                <select id="{prefix}-variable"></select>
              </label>
              <label>Before
                <select id="{prefix}-start"></select>
              </label>
              <label>After
                <select id="{prefix}-end"></select>
              </label>
              <div id="{prefix}-fixed-selectors"></div>
            </div>
            <div id="{prefix}-plots" aria-live="polite"></div>
            <script>
              const {prefix}Data = {serialized};
              const {prefix}Title = (value) => String(value).replaceAll("_", " ")
                .replace(/\\b\\w/g, (letter) => letter.toUpperCase());
              const {prefix}Variable = document.getElementById("{prefix}-variable");
              const {prefix}Start = document.getElementById("{prefix}-start");
              const {prefix}End = document.getElementById("{prefix}-end");
              const {prefix}Fixed = document.getElementById("{prefix}-fixed-selectors");
              const {prefix}Plots = document.getElementById("{prefix}-plots");
              const {prefix}Pairs = {prefix}Data.pairs || [];
              const {prefix}Attributes = {prefix}Data.control_attributes || [];
              const {prefix}Detected = Array.from(new Set(
                {prefix}Data.detected_attributes || []));
              const {prefix}Option = (value) =>
                '<option value="' + value + '">' + {prefix}Title(value) + "</option>";
              const {prefix}Missing = (value) => value === null || value === undefined
                ? "missing" : String(value);
              const {prefix}Render = () => {{
                const variable = {prefix}Variable.value;
                const available = {{}};
                {prefix}Pairs.filter((pair) => pair.variable_attribute === variable)
                  .forEach((pair) => {{
                    {(" (available[pair.fixed_attribute] ||= new Set()).add(pair.fixed_value);" if double else
                       "Object.entries(pair.fixed_values || {}).forEach(([attribute, value]) => " +
                       "(available[attribute] ||= new Set()).add(value));")}
                  }});
                const fixedAttributes = {(
                    "Object.keys(available).sort()"
                    if double else
                    f"{prefix}Attributes.filter((attribute) => attribute !== variable)"
                )};
                {(
                    f'''{prefix}Fixed.innerHTML =
                  '<label>Fixed attribute<select id="{prefix}-fixed-attribute">' +
                  fixedAttributes.map({prefix}Option).join("") +
                  '</select></label><label>Fixed value<select id="{prefix}-fixed-value"></select></label>';
                const fixedAttribute = document.getElementById("{prefix}-fixed-attribute");
                const fixedValue = document.getElementById("{prefix}-fixed-value");
                const updateFixedValues = () => {{
                  fixedValue.innerHTML = Array.from(available[fixedAttribute.value] || [])
                    .sort().map({prefix}Option).join("");
                }};
                fixedAttribute.addEventListener("change", () => {{
                  updateFixedValues();
                  draw();
                }});
                updateFixedValues();'''
                    if double else
                    f'''{prefix}Fixed.innerHTML = fixedAttributes.map((attribute) => {{
                  const values = Array.from(available[attribute] || []).sort();
                  return "<label>Fix " + {prefix}Title(attribute) +
                    '<select data-attribute="' + attribute + '">' +
                    values.map({prefix}Option).join("") + "</select></label>";
                }}).join("");'''
                )}
                const transitions = Array.from(new Set(
                  {prefix}Pairs.filter((pair) => pair.variable_attribute === variable)
                    .flatMap((pair) => pair.variable_values || [])
                )).sort();
                const previousStart = {prefix}Start.value;
                const previousEnd = {prefix}End.value;
                {prefix}Start.innerHTML = transitions.map({prefix}Option).join("");
                {prefix}End.innerHTML = transitions.map({prefix}Option).join("");
                {prefix}Start.value = transitions.includes(previousStart)
                  ? previousStart : (transitions[0] || "");
                {prefix}End.value = transitions.includes(previousEnd)
                  ? previousEnd : (transitions[transitions.length - 1] || "");
                const selected = () => {(
                    f'''({{
                  attribute: document.getElementById("{prefix}-fixed-attribute").value,
                  value: document.getElementById("{prefix}-fixed-value").value
                }})'''
                    if double else
                    f'''Object.fromEntries(
                  Array.from({prefix}Fixed.querySelectorAll("select")).map((select) =>
                    [select.dataset.attribute, select.value]))'''
                )};
                // Pairs are stored with their two values in sorted order; accept
                // either order and orient each pair as before -> after.
                const matching = () => {prefix}Pairs.flatMap((pair) => {{
                  if (pair.variable_attribute !== variable) return [];
                  const chosen = selected();
                  const fixedMatches = {(
                      "pair.fixed_attribute === chosen.attribute && pair.fixed_value === chosen.value"
                      if double else
                      "Object.entries(chosen).every(([attribute, value]) => " +
                      "pair.fixed_values && pair.fixed_values[attribute] === value)"
                  )};
                  if (!fixedMatches) return [];
                  const [left, right] = pair.variable_values || [];
                  if (left === {prefix}Start.value && right === {prefix}End.value) return [pair];
                  if (left === {prefix}End.value && right === {prefix}Start.value) {{
                    return [{{...pair, detected: {{
                      first: pair.detected.second, second: pair.detected.first
                    }}}}];
                  }}
                  return [];
                }});
                const draw = () => {{
                  const pairs = matching();
                  if ({prefix}Start.value === {prefix}End.value) {{
                    {prefix}Plots.innerHTML =
                      '<p class="note">Choose two different values for Before and After.</p>';
                    return;
                  }}
                  if (!pairs.length) {{
                    {prefix}Plots.innerHTML =
                      '<p class="note">No matched pairs fit these selections.</p>';
                    return;
                  }}
                  const attributes = {prefix}Detected.filter(
                    (attribute) => attribute !== variable);
                  // Two-level (multicategory) x axis: each attribute groups its
                  // "before" and "after" bars side by side under one label.
                  // Blank spacer slots (unique whitespace names) separate the groups.
                  const slots = attributes.flatMap((attribute, i) => [
                    ...(i ? [{{spacer: true, group: " ".repeat(i), side: " "}}] : []),
                    {{attribute, side: "first", group: {prefix}Title(attribute), label: "before"}},
                    {{attribute, side: "second", group: {prefix}Title(attribute), label: "after"}}
                  ]);
                  const groups = slots.map((slot) => slot.group);
                  const sides = slots.map((slot) => slot.spacer ? " " : slot.label);
                  const categories = Array.from(new Set(attributes.flatMap((attribute) =>
                    pairs.flatMap((pair) => [
                      {prefix}Missing(pair.detected.first[attribute]),
                      {prefix}Missing(pair.detected.second[attribute])
                    ])))).sort();
                  const counts = (attribute, side, category) => pairs.filter((pair) =>
                    {prefix}Missing(pair.detected[side][attribute]) === category).length;
                  {prefix}Plots.innerHTML = '<div id="{prefix}-plot"></div>';
                  const traces = categories.map((category) => ({{
                    type: "bar",
                    name: {prefix}Title(category),
                    x: [groups, sides],
                    customdata: groups.map((group, i) => group.trim() + " (" + sides[i] + ")"),
                    y: slots.map((slot) => slot.spacer ? null
                      : counts(slot.attribute, slot.side, category)),
                    text: slots.map((slot) => !slot.spacer
                      && counts(slot.attribute, slot.side, category)
                      ? {prefix}Title(category) : ""),
                    textposition: "inside",
                    insidetextanchor: "middle",
                    hovertemplate: "%{{customdata}}<br>Detected: %{{text}}<br>Pairs: %{{y}}<extra></extra>"
                  }}));
                  Plotly.react(document.getElementById("{prefix}-plot"), traces, {{
                    barmode: "stack",
                    bargap: 0.05,
                    height: Math.max(420, attributes.length * 80),
                    margin: {{l: 65, r: 25, t: 20, b: 100}},
                    yaxis: {{title: "Matched pairs"}},
                    template: "plotly_white"
                  }}, {{responsive: true, displaylogo: false}});
                }};
                {(
                    f'''document.getElementById("{prefix}-fixed-value")
                  .addEventListener("change", draw);'''
                    if double else
                    f'''{prefix}Fixed.querySelectorAll("select").forEach((select) =>
                  select.addEventListener("change", draw));'''
                )}
                // Assigned, not added: Render runs on every attribute change and
                // addEventListener would stack handlers bound to stale closures.
                {prefix}Start.onchange = draw;
                {prefix}End.onchange = draw;
                draw();
              }};
              {prefix}Variable.innerHTML = {prefix}Attributes.map({prefix}Option).join("");
              {prefix}Variable.addEventListener("change", {prefix}Render);
              if ({prefix}Attributes.length) {prefix}Render();
              else {prefix}Plots.innerHTML =
                '<p class="note">No matching analysis is available.</p>';
            </script>
"""

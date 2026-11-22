Create today's plain-English market breadth explainer page.

Use the `eli5` skill for the approach (big pictures, very few words, one idea per panel).
The `market-breadth-analyzer` skill is available if you need its methodology, but the
analysis has already been run: do not re-run it and do not try to fetch anything.

Inputs in this directory:
- `analysis.md` and `analysis.json`: today's Market Breadth Analyzer report. Every number,
  date and claim on the page must come from these two files. Do not state anything (time
  spans, history start dates, causes) that the files do not support.
- `facts.json`: the key facts the publish gate checks.
- `example.html`: an approved page from an earlier day. Match its structure, tone, visual
  style, length and CSS, but rebuild every number, label, picture and sentence from today's
  analysis. Never copy a number from the example. If today's story differs (for example
  breadth is rising, or a different zone), change the headline, pictures and "So what?" to fit.

Write exactly one file: `explainer.html` in this directory. A script checks it; a page that
fails any rule is discarded:
- English. One self-contained HTML file: inline CSS and inline SVG only.
- The file must start with `<!DOCTYPE html>`, then `<html lang="en">`, then `<head>`, with
  nothing in between. Put all CSS in a single `<style>` inside `<head>` (no `<style>` in the
  body or inside SVG). No HTML comments (`<!-- -->`) anywhere.
- No JavaScript at all (no `<script>`, no `on...=` attributes). Draw repeated shapes, such as
  the dot grid, as plain SVG elements.
- No external or embedded resources: no images, `<use>`, iframes, forms, `<link>`, `<base>`.
  In CSS: no `url(...)`, `@import`, `image-set`, backslash escapes, `display:none`,
  `visibility:hidden` or `opacity:0`. In SVG, `url(...)` may only point to an id in the page,
  such as `fill="url(#heat)"`.
- Do not use `hidden` or `aria-hidden` (except `aria-hidden` on an `<svg>`).
- The only links allowed are `href="index.html"` (required, back to the dashboard), `#...`
  anchors, and URLs starting with `https://tradermonty.github.io/`.
- In `<head>`, exactly these four tags, each once:
  `<meta name="breadth-data-date" content="{{DATA_DATE}}">`
  `<meta name="breadth-score" content="{{SCORE}}">`
  `<meta name="breadth-zone" content="{{ZONE}}">`
  `<meta name="breadth-pct" content="{{BREADTH_PCT}}">`
  Do not add other `<meta>` tags except charset, viewport and description.
- Label the key facts where they are shown, exactly like `example.html` does:
  `<time data-fact="date" datetime="{{DATA_DATE}}">...</time>`,
  `<span data-fact="score">{{SCORE}}</span>`, `<span data-fact="zone">{{ZONE}}</span>`,
  `<b data-fact="breadth">{{BREADTH_PCT}}</b>` (the share of S&P 500 stocks above trend,
  out of 100). Every element carrying `data-fact` must contain exactly that value, be a plain
  `<span>`, `<b>`, `<strong>` or `<time>` in the normal page text (not inside SVG), and have no
  `class`, `style` or `id`. The `<time>` text must show the same day and year.
- Inside `<head>` use only `<meta>`, `<title>` and one `<style>`.
- The visible words "Not investment advice".
- Mobile-first, readable at 390px wide, light and dark mode. Under 300 KB.

Do not write any other files.

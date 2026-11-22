/* market-breadth-mobile: phone layout for the market breadth Plotly chart.
 *
 * Embedded into market_breadth.html via Plotly's post_script. At widths <= 768px it:
 *  - opens on the last 2 years (override with ?initial=1y .. ?initial=5y),
 *  - keeps the latest date just inside the right edge for every range button,
 *  - replaces the long legend with a colour key + latest values in each panel title,
 *  - marks the latest point of every line,
 *  - turns off drag-zoom, double-tap reset and the mode bar so scrolling never changes the chart.
 * Desktop keeps the original look and interactions.
 */
(function () {
  'use strict';
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd || !window.Plotly) return;

  var MOBILE_MAX = 768;
  var MOBILE_MIN_HEIGHT = 840;
  var RIGHT_PAD_RATIO = 0.05;
  var MIN_PAD_DAYS = 3;
  var DAY_MS = 864e5;
  var initialMatch = /[?&]initial=([1-5])y\b/.exec(location.search);
  var INITIAL_YEARS = initialMatch ? Number(initialMatch[1]) : 2;

  var AXES = ['yaxis', 'yaxis2', 'yaxis3'].filter(function (k) { return gd.layout[k]; });
  var AXIS_ID = { yaxis: 'y', yaxis2: 'y2', yaxis3: 'y3' };
  var LAST_X = AXES.length > 1 ? 'xaxis' + AXES.length : 'xaxis';
  var ORIGINAL_COUNT = gd.data.length;

  // Snapshot of the desktop look, so switching back restores it exactly.
  var desk = {
    height: gd.layout.height,
    margin: clone(gd.layout.margin || {}),
    annotations: clone(gd.layout.annotations || []),
    titles: AXES.map(function (k) { return (gd.layout[k].title || {}).text || ''; }),
    markerSizes: gd.data.map(function (t) { return t.marker && t.marker.size; }),
    config: {
      displayModeBar: gd._context.displayModeBar,
      modeBarButtonsToRemove: gd._context.modeBarButtonsToRemove,
      displaylogo: gd._context.displaylogo,
      responsive: true,
    },
  };
  var mode = null;

  function clone(v) { return JSON.parse(JSON.stringify(v)); }
  function day(s) { return String(s).slice(0, 10); }
  function ms(isoDay) { return Date.parse(isoDay + 'T00:00:00Z'); }
  function iso(t) { return new Date(t).toISOString().slice(0, 10); }
  function lastDay() { var x = gd._fullData[0].x; return day(x[x.length - 1]); }
  function firstDay() { return day(gd._fullData[0].x[0]); }
  function minusYears(isoDay, n) {
    var d = new Date(ms(isoDay));
    d.setUTCFullYear(d.getUTCFullYear() - n);
    return iso(d.getTime());
  }
  function padMs(widthMs) { return Math.max(widthMs * RIGHT_PAD_RATIO, MIN_PAD_DAYS * DAY_MS); }
  function shortDate(isoDay) { var p = isoDay.split('-'); return Number(p[1]) + '/' + Number(p[2]); }

  function lastPoint(i) {
    var t = gd._fullData[i];
    var n = t.x.length;
    return { x: day(t.x[n - 1]), y: t.y[n - 1] };
  }
  function lineTraceIndexes(axisId) {
    var out = [];
    for (var i = 0; i < ORIGINAL_COUNT; i++) {
      var t = gd.data[i];
      if (t.yaxis === axisId && t.mode !== 'markers' && t.line) out.push(i);
    }
    return out;
  }

  function shortLabel(name) {
    var m = /\((\d+)-Day MA\)/.exec(name) || /(\d+)-Day MA/.exec(name);
    return m ? m[1] + 'MA' : name;
  }
  function swatch(color, label) { return '<span style="color:' + color + '">━</span> ' + label; }

  // Panel title with a colour key and the latest values, replacing the long legend on phones.
  function mobileTitle(axisKey) {
    var idx = lineTraceIndexes(AXIS_ID[axisKey]);
    if (!idx.length) return '';
    if (axisKey === 'yaxis') {
      var p = lastPoint(idx[0]);
      return '<b>S&P 500</b> ' + p.y.toFixed(2) + ' <span style="color:#888">(' + shortDate(p.x) + ', log)</span>';
    }
    var period = idx.some(function (i) { return /50-Day/.test(gd.data[i].name); }) ? '50D' : '200D';
    var keys = idx.map(function (i) {
      return swatch(gd.data[i].line.color, shortLabel(gd.data[i].name) + ' <b>' + lastPoint(i).y.toFixed(2) + '</b>');
    });
    return '<b>Breadth ' + period + '</b>  ' + keys.join('  ');
  }

  function subplotTitleIndexes() {
    // make_subplots titles are paper-referenced annotations, ordered here top to bottom.
    return desk.annotations
      .map(function (a, i) { return { a: a, i: i }; })
      .filter(function (o) { return o.a.xref === 'paper' && o.a.yref === 'paper'; })
      .sort(function (p, q) { return q.a.y - p.a.y; })
      .map(function (o) { return o.i; });
  }

  function mobileAnnotations() {
    var ann = clone(desk.annotations);
    subplotTitleIndexes().forEach(function (idx, n) {
      var a = ann[idx];
      a.text = mobileTitle(AXES[n]);
      a.x = 0;
      a.xanchor = 'left';
      a.font = { size: 12, color: '#222' };
    });
    ann.forEach(function (a) { if (a.xref !== 'paper') a.font = { size: 9, color: (a.font || {}).color }; });
    ann.push({
      xref: 'paper', yref: 'paper', x: 0, y: 0, yshift: -22, xanchor: 'left', yanchor: 'top', showarrow: false,
      font: { size: 10, color: '#555' },
      text: '<span style="color:#FF0000">▲</span> Peak  <span style="color:#0000FF">▼</span> Trough  ' +
        '<span style="color:#800080">▼</span> Deep trough  <span style="color:#e89cc8">■</span> Downtrend',
    });
    return ann;
  }

  // One dot on the latest point of every line; visible on phones only.
  function addLatestDots() {
    var dots = [];
    AXES.forEach(function (axisKey) {
      lineTraceIndexes(AXIS_ID[axisKey]).forEach(function (i) {
        var p = lastPoint(i);
        dots.push({
          x: [p.x], y: [p.y], xaxis: gd.data[i].xaxis, yaxis: gd.data[i].yaxis, mode: 'markers',
          name: 'latest', showlegend: false, hoverinfo: 'skip', visible: false,
          marker: { size: 8, color: gd.data[i].line.color, line: { color: 'white', width: 1.5 } },
        });
      });
    });
    return Plotly.addTraces(gd, dots);
  }

  function apply(next) {
    if (next === mode) return Promise.resolve();
    mode = next;
    var mobile = mode === 'mobile';
    var config = Object.assign({}, desk.config, {
      displayModeBar: mobile ? false : desk.config.displayModeBar,
      doubleClick: mobile ? false : 'reset+autosize',
    });
    var lay = {
      autosize: true,
      showlegend: !mobile,
      dragmode: mobile ? false : 'zoom',
      annotations: mobile ? mobileAnnotations() : desk.annotations,
      margin: mobile ? { l: 34, r: 8, t: 44, b: 44 } : desk.margin,
      height: mobile ? Math.max(window.innerHeight, MOBILE_MIN_HEIGHT) : desk.height,
      'font.size': mobile ? 10 : 12,
      'xaxis.rangeselector.font.size': mobile ? 13 : 12,
      'xaxis.rangeselector.x': mobile ? 1 : 0,
      'xaxis.rangeselector.xanchor': mobile ? 'right' : 'left',
    };
    AXES.forEach(function (k, n) { lay[k + '.title.text'] = mobile ? '' : desk.titles[n]; });
    if (mobile) {
      var end = lastDay();
      var start = minusYears(end, INITIAL_YEARS);
      lay[LAST_X + '.range'] = [start, iso(ms(end) + padMs(ms(end) - ms(start)))];
    } else {
      // Back to the desktop default: full history and an autoscaled price axis.
      lay[LAST_X + '.autorange'] = true;
      lay['yaxis.autorange'] = true;
    }
    var orig = [];
    var sizes = [];
    for (var i = 0; i < ORIGINAL_COUNT; i++) {
      var s = desk.markerSizes[i];
      orig.push(i);
      sizes.push(gd.data[i].mode === 'markers' && typeof s === 'number' && mobile ? Math.round(s * 0.7) : s);
    }
    var dots = [];
    for (var j = ORIGINAL_COUNT; j < gd.data.length; j++) dots.push(j);

    // write_html pins the plot div (and, in Plotly 7, its wrapper div) to the desktop height
    // inline; keep both equal to the drawn chart so phones get no blank scroll area below it.
    gd.style.height = lay.height + 'px';
    var wrapper = gd.parentElement;
    if (wrapper && wrapper !== document.body) wrapper.style.height = lay.height + 'px';
    // Drop the width Plotly auto-filled for the previous size so the chart refits after a breakpoint change.
    var base = Object.assign({}, gd.layout);
    delete base.width;
    return Plotly.react(gd, gd.data, base, config)
      .then(function () { return Plotly.restyle(gd, { 'marker.size': sizes }, orig); })
      .then(function () { return dots.length ? Plotly.restyle(gd, { visible: mobile }, dots) : null; })
      .then(function () { return Plotly.relayout(gd, lay); })
      .then(function () { return mobile ? fitPriceAxis() : null; })
      .then(function () { document.documentElement.dataset.mbMode = mode; });
  }

  // Plotly's range buttons count back from the *current* right edge, and ALL's autorange adds
  // months of blank space. Whenever the window reaches the latest bar, re-anchor it so the latest
  // date sits just inside the right edge, keeping the window width that was picked.
  function anchorRight(ev) {
    var upd = {};
    var last = lastDay();
    var autorange = Object.keys(ev).some(function (k) { return /autorange/.test(k) && ev[k]; });
    if (autorange) {
      upd[LAST_X + '.range'] = [firstDay(), iso(ms(last) + padMs(ms(last) - ms(firstDay())))];
      Plotly.relayout(gd, upd);
      return true;
    }
    var r = gd._fullLayout[LAST_X].range;
    if (!r) return false;
    var lo = ms(day(r[0]));
    var hi = ms(day(r[1]));
    if (hi < ms(last) - DAY_MS) return false; // looking at the past: leave it alone
    var end = ms(last) + padMs(hi - lo);
    if (Math.abs(hi - end) <= DAY_MS) return false; // already anchored
    upd[LAST_X + '.range'] = [iso(end - (hi - lo)), iso(end)];
    Plotly.relayout(gd, upd);
    return true;
  }

  // Plotly autoranges the log price axis over all data, which flattens short windows.
  // Fit it to the prices inside the visible window instead.
  function fitPriceAxis() {
    var r = gd._fullLayout[LAST_X].range;
    if (!r) return null;
    var lo = day(r[0]);
    var hi = day(r[1]);
    var min = Infinity;
    var max = -Infinity;
    gd._fullData.forEach(function (t) {
      // Hidden traces (the desktop-hidden latest dots) carry no x/y in _fullData.
      if (t.yaxis !== 'y' || t.visible !== true || !t.x) return;
      for (var i = 0; i < t.x.length; i++) {
        var d = day(t.x[i]);
        if (d >= lo && d <= hi) {
          if (t.y[i] < min) min = t.y[i];
          if (t.y[i] > max) max = t.y[i];
        }
      }
    });
    if (!isFinite(min) || min <= 0) return null;
    return Plotly.relayout(gd, { 'yaxis.range': [Math.log10(min * 0.97), Math.log10(max * 1.03)] });
  }

  // Phone-only: on desktop, zooms/pans and the price autorange behave exactly as Plotly's defaults.
  gd.on('plotly_relayout', function (ev) {
    if (mode !== 'mobile') return;
    var keys = Object.keys(ev || {});
    if (keys.some(function (k) { return /^xaxis\d*\.(range|autorange)/.test(k); })) {
      if (!anchorRight(ev)) fitPriceAxis();
    }
  });

  function currentMode() { return window.innerWidth <= MOBILE_MAX ? 'mobile' : 'desktop'; }
  var timer = null;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { apply(currentMode()); }, 150);
  });
  addLatestDots().then(function () { return apply(currentMode()); });
})();

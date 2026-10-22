import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/zipf-heaps-vocabulary-statistics/zipf_heaps_vocabulary_statistics.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/vocabulary-laws-lab.json';

/**
 * Vocabulary Laws Laboratory — two linked panels for the Zipf/Heaps topic.
 *   A. "Fit the line": the finance corpus's rank–frequency curve. Pick a window of ranks and
 *      the dense log–log OLS slope swings from −0.554 to −1.807; the discrete power-law MLE
 *      (Clauset et al.) is a fixed line through (r(x_min), x_min) by Proposition 1.
 *   B. "Turn the discount": one Pitman–Yor(d, θ = 20) restaurant per discount. Its type count
 *      K_n against the exact E[K_n] of Theorem 3, the naive and offset-corrected Heaps slopes,
 *      and an inset rank–frequency curve with reference slope −1/d.
 * A shared strip ties the two: β = d ⇔ s = 1/d, against the corpus's fitted slopes.
 */

type Window = { slope: number; intercept: number };
type Growth = {
  n: number[]; K: number[]; EK: number[]; heaps_naive: number; heaps_shifted: number;
  rank: number[]; freq: number[];
};
const CORPUS = payload.corpus;
const WINDOWS = payload.windows as Record<string, Window>;
const MLE = payload.mle;
const THETA = payload.theta;
const GROWTH = payload.growth as Record<string, Growth>;
const DISCOUNTS = Object.keys(GROWTH).sort(); // "0.3" … "0.8"
const WINDOW_KEYS = ['1-10', '10-100', '100-1000', '1000-3425', '1-1000', '1-3425'];

const W = 560, H = 320;
const M = { t: 16, r: 16, b: 40, l: 52 };
const sans = 'var(--font-sans)';

function logAxes(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  x: d3.ScaleLogarithmic<number, number>, y: d3.ScaleLogarithmic<number, number>,
  w: number, h: number, xLabel: string, yLabel: string,
) {
  const fmt = (v: d3.NumberValue) => {
    const e = Math.log10(+v);
    return Number.isInteger(e) ? d3.format(',')(+v) : '';
  };
  svg.append('g').attr('transform', `translate(0,${h - M.b})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(fmt))
    .call((g) => g.selectAll('line,path').style('stroke', 'var(--color-muted-border)'));
  svg.append('g').attr('transform', `translate(${M.l},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmt))
    .call((g) => g.selectAll('line,path').style('stroke', 'var(--color-muted-border)'));
  svg.append('text').attr('x', (M.l + w - M.r) / 2).attr('y', h - 6).attr('text-anchor', 'middle')
    .style('font-family', sans).style('font-size', '12px').text(xLabel);
  svg.append('text').attr('transform', `translate(14,${(M.t + h - M.b) / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').style('font-family', sans).style('font-size', '12px').text(yLabel);
}

function Readout({ label, value, id, emphasis }: { label: string; value: string; id: string; emphasis?: boolean }) {
  return (
    <div style={{ fontFamily: sans, fontSize: '0.8rem' }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)' }}>
        {value}
      </div>
    </div>
  );
}

export default function VocabularyLawsLab() {
  const [windowKey, setWindowKey] = useState('100-1000');
  const [dIndex, setDIndex] = useState(DISCOUNTS.indexOf('0.6'));
  const zipfRef = useRef<SVGSVGElement>(null);
  const heapsRef = useRef<SVGSVGElement>(null);
  const insetRef = useRef<SVGSVGElement>(null);

  const dKey = DISCOUNTS[dIndex];
  const d = parseFloat(dKey);
  const g = GROWTH[dKey];
  const win = WINDOWS[windowKey];
  const [lo, hi] = windowKey.split('-').map(Number);

  // Panel A — corpus rank–frequency, OLS over the chosen window, fixed MLE line.
  useEffect(() => {
    if (!zipfRef.current) return;
    const svg = d3.select(zipfRef.current);
    svg.selectAll('*').remove();
    const x = d3.scaleLog().domain([1, CORPUS.V]).range([M.l, W - M.r]);
    const y = d3.scaleLog().domain([1, 4000]).range([H - M.b, M.t]);
    logAxes(svg, x, y, W, H, 'rank r', 'count f(r)');

    svg.append('rect').attr('x', x(lo)).attr('width', x(hi) - x(lo)).attr('y', M.t).attr('height', H - M.b - M.t)
      .style('fill', 'var(--color-muted-bg)');
    svg.append('g').selectAll('circle').data(CORPUS.rank.map((r, i) => [r, CORPUS.freq[i]]))
      .join('circle').attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 3.2)
      .style('fill', (p) => (p[0] >= lo && p[0] <= hi ? 'var(--color-accent)' : 'var(--color-text-secondary)'))
      .style('opacity', (p) => (p[0] >= lo && p[0] <= hi ? 1 : 0.55));

    // OLS line over the window: log f = intercept + slope · log r (natural logs).
    const ols = (r: number) => Math.exp(win.intercept + win.slope * Math.log(r));
    svg.append('line').attr('x1', x(lo)).attr('y1', y(ols(lo))).attr('x2', x(hi)).attr('y2', y(ols(hi)))
      .style('stroke', 'var(--color-accent)').style('stroke-width', 2.5);

    // MLE line on the rank plot: f(r) = x_min · (r / r(x_min))^(−ŝ), through (r(x_min), x_min).
    const mle = (r: number) => MLE.xmin * Math.pow(r / MLE.r_xmin, -MLE.s);
    const r0 = Math.pow(4000 / MLE.xmin, -1 / MLE.s) * MLE.r_xmin; // where the line leaves the top
    svg.append('line').attr('x1', x(Math.max(1, r0))).attr('y1', y(mle(Math.max(1, r0))))
      .attr('x2', x(MLE.r_xmin)).attr('y2', y(MLE.xmin))
      .style('stroke', 'var(--color-accent-secondary)').style('stroke-width', 2).style('stroke-dasharray', '6 4');
    const legend = svg.append('g').attr('transform', `translate(${W - M.r - 4},${M.t + 6})`)
      .style('font-family', sans).style('font-size', '11px').attr('text-anchor', 'end');
    legend.append('text').attr('y', 6).style('fill', 'var(--color-accent)')
      .text(`━ OLS over ranks ${lo}–${hi}: slope ${win.slope.toFixed(3)}`);
    legend.append('text').attr('y', 22).style('fill', 'var(--color-accent-secondary)')
      .text(`╍ MLE: ŝ = ${MLE.s.toFixed(3)} (x_min = ${MLE.xmin})`);
  }, [windowKey]);

  // Panel B — Pitman–Yor type growth: one path vs. the exact mean (Theorem 3).
  useEffect(() => {
    if (!heapsRef.current) return;
    const svg = d3.select(heapsRef.current);
    svg.selectAll('*').remove();
    const x = d3.scaleLog().domain([10, 20000]).range([M.l, W - M.r]);
    const y = d3.scaleLog().domain([5, 8000]).range([H - M.b, M.t]);
    logAxes(svg, x, y, W, H, 'tokens n', 'types K_n');
    const line = d3.line<number>().x((_, i) => x(g.n[i])).y((v) => y(v));
    svg.append('path').datum(g.EK).attr('d', line).style('fill', 'none')
      .style('stroke', 'var(--color-accent-secondary)').style('stroke-width', 2.5);
    svg.append('g').selectAll('circle').data(g.K).join('circle')
      .attr('cx', (_, i) => x(g.n[i])).attr('cy', (v) => y(v)).attr('r', 3.2).style('fill', 'var(--color-accent)');
    const last = g.n.length - 1;
    const legend = svg.append('g').attr('transform', `translate(${M.l + 10},${M.t + 6})`)
      .style('font-family', sans).style('font-size', '11px');
    legend.append('text').attr('y', 6).style('fill', 'var(--color-accent-secondary)')
      .text(`━ exact E[K_n] (Theorem 3): ${g.EK[last].toFixed(1)} at n = 20,000`);
    legend.append('text').attr('y', 22).style('fill', 'var(--color-accent)')
      .text(`● this restaurant: K = ${g.K[last]} at n = 20,000`);

    // Inset: the same restaurant's rank–frequency curve with reference slope −1/d.
    if (!insetRef.current) return;
    const ins = d3.select(insetRef.current);
    ins.selectAll('*').remove();
    const iw = 220, ih = 150, im = { t: 8, r: 8, b: 26, l: 36 };
    const ix = d3.scaleLog().domain([1, 10000]).range([im.l, iw - im.r]);
    const iy = d3.scaleLog().domain([1, 5000]).range([ih - im.b, im.t]);
    ins.append('g').attr('transform', `translate(0,${ih - im.b})`).call(d3.axisBottom(ix).ticks(3, '~s'))
      .call((s) => s.selectAll('line,path').style('stroke', 'var(--color-muted-border)'));
    ins.append('g').attr('transform', `translate(${im.l},0)`).call(d3.axisLeft(iy).ticks(3, '~s'))
      .call((s) => s.selectAll('line,path').style('stroke', 'var(--color-muted-border)'));
    ins.append('g').selectAll('circle').data(g.rank).join('circle')
      .attr('cx', (r) => ix(r)).attr('cy', (_, i) => iy(g.freq[i])).attr('r', 2.2).style('fill', 'var(--color-accent)');
    // Reference slope −1/d through the median displayed point (placement only; the slope is the claim).
    const mid = Math.floor(g.rank.length / 2);
    const ref = (r: number) => g.freq[mid] * Math.pow(r / g.rank[mid], -1 / d);
    const rA = Math.max(1, g.rank[mid] / 30), rB = Math.min(10000, g.rank[mid] * 30);
    ins.append('line').attr('x1', ix(rA)).attr('y1', iy(Math.min(5000, ref(rA)))).attr('x2', ix(rB))
      .attr('y2', iy(Math.max(1, ref(rB))))
      .style('stroke', 'var(--color-accent-secondary)').style('stroke-dasharray', '4 3').style('stroke-width', 1.5);
    ins.append('text').attr('x', iw - im.r).attr('y', im.t + 10).attr('text-anchor', 'end')
      .style('font-family', sans).style('font-size', '10px').text(`slope −1/d = ${(-1 / d).toFixed(3)}`);
  }, [dIndex]);

  const button = (k: string) => (
    <button key={k} type="button" onClick={() => setWindowKey(k)} aria-pressed={k === windowKey} data-window={k}
      style={{
        fontFamily: sans, fontSize: '0.75rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
        border: '1px solid var(--color-border)',
        background: k === windowKey ? 'var(--color-accent)' : 'var(--color-surface)',
        color: k === windowKey ? 'var(--color-bg)' : 'var(--color-text)',
      }}>
      ranks {k.replace('-', '–')}
    </button>
  );

  return (
    <div data-lab="vocabulary-laws" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Fit the line — {CORPUS.N.toLocaleString('en-US')} tokens, {CORPUS.V.toLocaleString('en-US')} types of the finance corpus
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>{WINDOW_KEYS.map(button)}</div>
        <svg ref={zipfRef} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
          aria-label="Log–log rank–frequency plot of the finance corpus with an OLS line over the chosen rank window and the fixed MLE line" />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="ols-slope" label={`OLS slope, ranks ${lo}–${hi}`} value={win.slope.toFixed(3)} emphasis />
          <Readout id="ols-s" label="implied Zipf s" value={(-win.slope).toFixed(3)} />
          <Readout id="mle-s" label="MLE ŝ = 1/(α̂ − 1)" value={MLE.s.toFixed(3)} />
        </div>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Turn the discount — one Pitman–Yor restaurant, θ = {THETA}, n = 20,000
        </figcaption>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.875rem' }}>
          <span style={{ minWidth: '6rem' }}>discount d = <strong data-readout="d">{d.toFixed(1)}</strong></span>
          <input type="range" min={0} max={DISCOUNTS.length - 1} step={1} value={dIndex} aria-label="Pitman–Yor discount d"
            onChange={(e) => setDIndex(parseInt(e.target.value, 10))} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', gap: '0.75rem' }}>
          <svg ref={heapsRef} viewBox={`0 0 ${W} ${H}`} style={{ flex: '1 1 340px', minWidth: 0, height: 'auto' }} role="img"
            aria-label="Log–log plot of distinct types against tokens for one Pitman–Yor restaurant, with its exact expected type count" />
          <svg ref={insetRef} viewBox="0 0 220 150" style={{ flex: '0 1 220px', minWidth: 0, height: 'auto' }} role="img"
            aria-label="Rank–frequency plot of the same restaurant with a reference line of slope −1/d" />
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="heaps-naive" label="naive Heaps slope" value={g.heaps_naive.toFixed(3)} />
          <Readout id="heaps-shifted" label="offset-corrected slope" value={g.heaps_shifted.toFixed(3)} emphasis />
          <Readout id="heaps-d" label="planted d" value={d.toFixed(3)} />
        </div>
      </figure>

      <div data-readout="strip" style={{
        marginTop: '1rem', padding: '0.6rem 0.8rem', borderRadius: '0.5rem', background: 'var(--color-definition-bg)',
        fontFamily: sans, fontSize: '0.85rem',
      }}>
        Pitman–Yor ties the laws together: <strong>β = d = {d.toFixed(1)}</strong> ⇔ <strong>s = 1/d = {(1 / d).toFixed(3)}</strong>.
        {' '}The corpus window above reads s = {(-win.slope).toFixed(3)}; the corpus's content words were planted at d = 0.6, s = 1.667.
      </div>
      <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
        One restaurant's level is random — K_n / n^d converges to a random limit — so the points can sit off the
        exact mean while their slope still tracks d. The naive slope ignores the θ/d offset of Theorem 3 and overshoots.
      </p>
    </div>
  );
}

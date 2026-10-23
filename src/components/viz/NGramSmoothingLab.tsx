import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/n-gram-models-smoothing/n_gram_models_smoothing.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/n-gram-smoothing-lab.json';

/**
 * n-Gram Smoothing Laboratory — two linked panels for the n-gram smoothing topic.
 *   A. "Smooth a context": the next-word distribution after one of five trigram contexts under seven
 *      smoothers — the top eight words (filled if seen after the context, hollow if not), then the rest
 *      of the seen and never-seen mass — with the whole model's held-out log-loss.
 *   B. "Discount versus count": the discount each method takes from a trigram seen r times — Turing's
 *      raw r − r*, Simple Good–Turing, interpolated KN's constant D, modified KN's three steps, and the
 *      sampled hierarchical Pitman–Yor model's d·E[t | r]; a toggle shows the trigram spectrum N_r.
 * One D (applied at every order) drives absolute discounting and interpolated KN in A and the KN line in
 * B; hovering a count r in B highlights the words in A that followed the context exactly r times.
 */

type Summary = {
  top: [string, number, number][]; // word, probability, training count after the context (0 = unseen)
  rest_seen: number;
  rest_unseen: number;
  unseen_mass: number;
  watch: [number, number]; // probability and rank of the watch word
};
type Context = {
  context: string[];
  count: number;
  types: number;
  fixed: Record<'mle' | 'laplace' | 'dirichlet' | 'katz' | 'mkn', Summary | null>;
  by_d: Record<'absdisc' | 'ikn', Summary[]>;
};
type Smoother = 'mle' | 'laplace' | 'dirichlet' | 'katz' | 'absdisc' | 'ikn' | 'mkn';

const D_GRID = payload.D as number[];
const CONTEXTS = payload.contexts as unknown as Context[];
const LOGLOSS = payload.logloss as unknown as Record<string, number | null> & {
  katz_finite: number; katz_zero_events: number; hpylm: number; by_d: Record<'absdisc' | 'ikn', number[]>;
};
const DISCOUNT = payload.discount as { r: number[]; gt: number[]; sgt: number[]; ney: number; mkn: number[]; hpylm: number[] };
const SPECTRUM = payload.spectrum as { r: number[]; n: number[] };
const WATCH = payload.watch as string;

const SMOOTHERS: { key: Smoother; label: string }[] = [
  { key: 'mle', label: 'MLE' },
  { key: 'laplace', label: 'Laplace' },
  { key: 'dirichlet', label: 'Dirichlet (Type-II α̂)' },
  { key: 'katz', label: 'Katz' },
  { key: 'absdisc', label: 'Absolute discounting' },
  { key: 'ikn', label: 'Interpolated KN' },
  { key: 'mkn', label: 'Modified KN' },
];
const USES_D = new Set<Smoother>(['absdisc', 'ikn']);

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const show = (w: string) => (w === '<unk>' ? '⟨unk⟩' : w);
const label = (c: Context) => `${c.context.map(show).join(' ')} ___`;

function summaryOf(ctx: Context, s: Smoother, d: number): Summary | null {
  return USES_D.has(s) ? ctx.by_d[s as 'absdisc' | 'ikn'][d] : ctx.fixed[s as keyof Context['fixed']];
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

function Pill({ active, onClick, children, data }: { active: boolean; onClick: () => void; children: string; data: Record<string, string> }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} {...data}
      style={{
        fontFamily: sans, fontSize: '0.75rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
        border: '1px solid var(--color-border)',
        background: active ? 'var(--color-accent)' : 'var(--color-surface)',
        color: active ? 'var(--color-bg)' : 'var(--color-text)',
      }}>
      {children}
    </button>
  );
}

// Panel A geometry: a word column, a bar track, and a value column.
const AW = 560, LABEL = 140, VALUE = 70, ROW = 21;
const AH = (CONTEXTS[0].fixed.laplace!.top.length + 2) * ROW + 8;

// Panel B geometry.
const BW = 560, BH = 270, BM = { top: 14, right: 16, bottom: 34, left: 44 };

export default function NGramSmoothingLab() {
  const [ctxIdx, setCtxIdx] = useState(0);
  const [smoother, setSmoother] = useState<Smoother>('ikn');
  const [dIdx, setDIdx] = useState(D_GRID.indexOf(0.7));
  const [view, setView] = useState<'discount' | 'spectrum'>('discount');
  const [hoverR, setHoverR] = useState<number | null>(null);
  const barsRef = useRef<SVGSVGElement>(null);
  const curveRef = useRef<SVGSVGElement>(null);

  const ctx = CONTEXTS[ctxIdx];
  const D = D_GRID[dIdx];
  const sum = summaryOf(ctx, smoother, dIdx);

  // One probability scale per context, shared by every smoother, so switching smoothers compares bars.
  const ctxMax = useMemo(() => {
    const all = [...Object.values(ctx.fixed), ...ctx.by_d.absdisc, ...ctx.by_d.ikn].filter(Boolean) as Summary[];
    return d3.max(all.flatMap((s) => [...s.top.map((t) => t[1]), s.rest_seen, s.rest_unseen])) ?? 1;
  }, [ctxIdx]);

  // Panel A — the next-word distribution.
  useEffect(() => {
    if (!barsRef.current) return;
    const svg = d3.select(barsRef.current);
    svg.selectAll('*').remove();
    if (!sum) {
      svg.append('text').attr('x', AW / 2).attr('y', AH / 2).attr('text-anchor', 'middle').attr('data-undefined', 'true')
        .style('font-family', sans).style('font-size', '13px').style('fill', 'var(--color-text-secondary)')
        .text('Undefined: the MLE divides by c(u·) = 0 — this context never occurred in training.');
      return;
    }
    const x = d3.scaleLinear().domain([0, ctxMax]).range([0, AW - LABEL - VALUE]);
    const rows: { name: string; p: number; count: number; kind: 'word' | 'rest-seen' | 'rest-unseen' }[] = [
      ...sum.top.map(([w, p, c]) => ({ name: show(w), p, count: c, kind: 'word' as const })),
      { name: 'other seen here', p: sum.rest_seen, count: -1, kind: 'rest-seen' },
      { name: 'never seen here', p: sum.rest_unseen, count: -1, kind: 'rest-unseen' },
    ];
    rows.forEach((r, i) => {
      const y = 4 + i * ROW;
      const seen = r.kind === 'rest-seen' || (r.kind === 'word' && r.count > 0);
      const lit = hoverR !== null && r.kind === 'word' && r.count === hoverR;
      const g = svg.append('g').attr('data-bar', r.name).attr('data-count', String(r.count)).attr('data-lit', String(lit));
      g.append('text').attr('x', LABEL - 8).attr('y', y + ROW / 2 + 3).attr('text-anchor', 'end')
        .style('font-family', r.kind === 'word' ? mono : sans).style('font-size', r.kind === 'word' ? '12px' : '11px')
        .style('font-style', r.kind === 'word' ? 'normal' : 'italic')
        .style('fill', r.kind === 'word' && r.name === WATCH ? 'var(--color-accent-secondary)' : 'var(--color-text)')
        .text(r.kind === 'word' && r.count > 0 ? `${r.name} ×${r.count}` : r.name);
      g.append('rect').attr('x', LABEL).attr('y', y + 2).attr('width', Math.max(1, x(r.p))).attr('height', ROW - 6).attr('rx', 3)
        .style('fill', seen ? 'var(--color-accent)' : 'var(--color-surface)')
        .style('fill-opacity', r.kind === 'word' ? 1 : 0.55)
        .style('stroke', lit ? 'var(--color-accent-secondary)' : 'var(--color-accent)')
        .style('stroke-width', lit ? 3 : 1.25);
      g.append('text').attr('x', LABEL + x(r.p) + 6).attr('y', y + ROW / 2 + 3)
        .style('font-family', mono).style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text(r.p < 1e-4 && r.p > 0 ? r.p.toExponential(1) : r.p.toFixed(4));
    });
  }, [ctxIdx, smoother, dIdx, hoverR]);

  // Panel B — discount versus count, or the trigram spectrum.
  useEffect(() => {
    if (!curveRef.current) return;
    const svg = d3.select(curveRef.current);
    svg.selectAll('*').remove();
    const iw = BW - BM.left - BM.right, ih = BH - BM.top - BM.bottom;
    const g = svg.append('g').attr('transform', `translate(${BM.left},${BM.top})`);
    const axisStyle = (sel: d3.Selection<SVGGElement, unknown, null, undefined>) => {
      sel.selectAll('text').style('font-family', sans).style('font-size', '10px').style('fill', 'var(--color-text-secondary)');
      sel.selectAll('path,line').style('stroke', 'var(--color-muted-border)');
    };

    if (view === 'spectrum') {
      const x = d3.scaleLog().domain([1, 30]).range([0, iw]);
      const y = d3.scaleLog().domain([d3.min(SPECTRUM.n.filter((n) => n > 0))!, d3.max(SPECTRUM.n)!]).nice().range([ih, 0]);
      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5, '~d')).call(axisStyle);
      g.append('g').call(d3.axisLeft(y).ticks(5, '~s')).call(axisStyle);
      g.selectAll('circle').data(SPECTRUM.r.map((r, i) => [r, SPECTRUM.n[i]] as const).filter(([, n]) => n > 0)).join('circle')
        .attr('cx', ([r]) => x(r)).attr('cy', ([, n]) => y(n)).attr('r', 3.5).attr('data-series', 'spectrum')
        .style('fill', 'var(--color-accent)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('font-family', sans)
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('count r (log scale)');
      g.append('text').attr('x', 6).attr('y', 4).style('font-family', sans).style('font-size', '11px')
        .style('fill', 'var(--color-text-secondary)').text('N_r: trigram types seen r times');
      return;
    }

    const x = d3.scaleLinear().domain([0.6, DISCOUNT.r.length + 0.4]).range([0, iw]);
    const y = d3.scaleLinear().domain([-1, 3]).range([ih, 0]);
    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(DISCOUNT.r.length).tickFormat(d3.format('d'))).call(axisStyle);
    g.append('g').call(d3.axisLeft(y).ticks(8)).call(axisStyle);
    g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--color-muted-border)');
    if (hoverR !== null) {
      g.append('rect').attr('x', x(hoverR - 0.4)).attr('width', x(hoverR + 0.4) - x(hoverR - 0.4)).attr('y', 0).attr('height', ih)
        .style('fill', 'var(--color-muted-bg)');
    }
    const line = d3.line<[number, number]>().x(([r]) => x(r)).y(([, v]) => y(v));
    const pairs = (vs: number[]) => DISCOUNT.r.map((r, i) => [r, vs[i]] as [number, number]);
    // Modified KN's three discounts as a step function: D₁ at r = 1, D₂ at r = 2, D₃₊ beyond.
    const mknAt = (r: number) => DISCOUNT.mkn[Math.min(r, 3) - 1];
    g.append('path').datum(DISCOUNT.r.map((r) => [r, mknAt(r)] as [number, number]))
      .attr('d', d3.line<[number, number]>().x(([r]) => x(r)).y(([, v]) => y(v)).curve(d3.curveStepAfter))
      .attr('data-series', 'mkn').style('fill', 'none').style('stroke', 'var(--color-badge-blue-text)').style('stroke-width', 2);
    g.append('path').datum(pairs(DISCOUNT.sgt)).attr('d', line).attr('data-series', 'sgt')
      .style('fill', 'none').style('stroke', 'var(--color-badge-green-text)').style('stroke-width', 2);
    g.append('path').datum(pairs(DISCOUNT.hpylm)).attr('d', line).attr('data-series', 'hpylm')
      .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.5);
    g.selectAll('circle.gt').data(pairs(DISCOUNT.gt)).join('circle').attr('class', 'gt').attr('data-series', 'gt')
      .attr('cx', ([r]) => x(r)).attr('cy', ([, v]) => y(v)).attr('r', 3.5)
      .style('fill', 'var(--color-surface)').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.5);
    // The shared slider's D: interpolated KN subtracts the same D from every count.
    g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(D)).attr('y2', y(D)).attr('data-series', 'kn-d')
      .style('stroke', 'var(--color-accent-secondary)').style('stroke-width', 2).style('stroke-dasharray', '6 4');
    g.append('text').attr('x', iw - 4).attr('y', y(D) - 5).attr('text-anchor', 'end').style('font-family', sans)
      .style('font-size', '10px').style('fill', 'var(--color-accent-secondary)').text(`interpolated KN, D = ${D.toFixed(1)}`);
    g.append('line').attr('x1', -6).attr('x2', 0).attr('y1', y(DISCOUNT.ney)).attr('y2', y(DISCOUNT.ney))
      .style('stroke', 'var(--color-accent-secondary)').style('stroke-width', 2);
    // Hover targets, one per count r.
    g.selectAll('rect.hit').data(DISCOUNT.r).join('rect').attr('class', 'hit').attr('data-r', (r) => String(r))
      .attr('x', (r) => x(r - 0.5)).attr('width', (r) => x(r + 0.5) - x(r - 0.5)).attr('y', 0).attr('height', ih)
      .style('fill', 'transparent').style('cursor', 'pointer')
      .on('mouseenter', (_, r) => setHoverR(r)).on('mouseleave', () => setHoverR(null))
      .on('click', (_, r) => setHoverR((cur) => (cur === r ? null : r)));
    g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('font-family', sans)
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('training count r of a trigram');
    g.append('text').attr('transform', `translate(${-32},${ih / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .style('font-family', sans).style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('discount taken from r');
  }, [view, dIdx, hoverR]);

  const loss = USES_D.has(smoother) ? LOGLOSS.by_d[smoother as 'absdisc' | 'ikn'][dIdx] : LOGLOSS[smoother];
  const lossText = smoother === 'katz'
    ? `∞ — ${LOGLOSS.katz_zero_events} events get 0; ${LOGLOSS.katz_finite.toFixed(3)} on the rest`
    : loss === null ? '∞' : `${(loss as number).toFixed(3)} nats/token`;
  const r = hoverR === null ? null : hoverR - 1;

  return (
    <div data-lab="n-gram-smoothing" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.875rem', marginBottom: '0.75rem' }}>
        <span style={{ minWidth: '9rem' }}>discount D = <strong data-readout="d">{D.toFixed(1)}</strong></span>
        <input type="range" min={0} max={D_GRID.length - 1} step={1} value={dIdx} aria-label="Discount D at every order (both panels)"
          onChange={(e) => setDIdx(parseInt(e.target.value, 10))} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      </label>

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Smooth a context — what follows <span style={{ fontFamily: mono }}>{label(ctx)}</span>?
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.4rem' }}>
          {CONTEXTS.map((c, i) => (
            <Pill key={i} active={i === ctxIdx} onClick={() => setCtxIdx(i)} data={{ 'data-context': c.context.join(' ') }}>{label(c)}</Pill>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          {SMOOTHERS.map((s) => (
            <Pill key={s.key} active={s.key === smoother} onClick={() => setSmoother(s.key)} data={{ 'data-smoother': s.key }}>
              {USES_D.has(s.key) ? `${s.label} (D)` : s.label}
            </Pill>
          ))}
        </div>
        <svg ref={barsRef} viewBox={`0 0 ${AW} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img"
          aria-label="Next-word probabilities after the chosen context under the chosen smoother: the top eight words, then the remaining mass on words seen and never seen after the context" />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="context-count" label="context seen in training" value={`${ctx.count}× · ${ctx.types} distinct next words`} />
          <Readout id="unseen-mass" label="mass on words never seen here" value={sum ? sum.unseen_mass.toFixed(4) : '—'} emphasis />
          <Readout id="watch" label={`“${WATCH}”: probability · rank`} value={sum ? `${sum.watch[0].toExponential(2)} · #${sum.watch[1]}` : '—'} />
          <Readout id="logloss" label="whole model, held-out log-loss" value={lossText} />
        </div>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
          Filled bars: words seen after this context (×count). Hollow: never seen here. For reference, the sampled
          hierarchical Pitman–Yor model reaches {LOGLOSS.hpylm.toFixed(3)} nats/token.
        </p>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · {view === 'discount' ? 'Discount versus count — how much each method takes from a trigram seen r times' : 'The trigram spectrum N_r'}
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          <Pill active={view === 'discount'} onClick={() => setView('discount')} data={{ 'data-view': 'discount' }}>discount vs. count</Pill>
          <Pill active={view === 'spectrum'} onClick={() => setView('spectrum')} data={{ 'data-view': 'spectrum' }}>spectrum N_r</Pill>
        </div>
        <svg ref={curveRef} viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img"
          aria-label="Discount taken from a trigram seen r times, r from 1 to 8, by Good–Turing, Simple Good–Turing, interpolated and modified Kneser–Ney, and the sampled hierarchical Pitman–Yor model" />
        {view === 'discount' && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.9rem', fontFamily: sans, fontSize: '0.75rem', marginTop: '0.25rem' }}>
            <span><span style={{ color: 'var(--color-text-secondary)' }}>○</span> Turing r − r* (raw)</span>
            <span><span style={{ color: 'var(--color-badge-green-text)' }}>━</span> Simple Good–Turing</span>
            <span><span style={{ color: 'var(--color-accent-secondary)' }}>┅</span> interpolated KN (D; Ney's {DISCOUNT.ney.toFixed(3)} ticked)</span>
            <span><span style={{ color: 'var(--color-badge-blue-text)' }}>━</span> modified KN</span>
            <span><span style={{ color: 'var(--color-accent)' }}>━</span> HPYLM d·E[t | r]</span>
          </div>
        )}
        {view === 'discount' && (
          <div data-readout="at-r" style={{
            marginTop: '0.6rem', padding: '0.6rem 0.8rem', borderRadius: '0.5rem', background: 'var(--color-definition-bg)',
            fontFamily: sans, fontSize: '0.85rem',
          }}>
            {r === null ? (
              <>Hover a count r to read the discounts and light up, in panel A, the words seen exactly r times after the context.</>
            ) : (
              <>
                At r = {hoverR}: Turing <strong>{DISCOUNT.gt[r].toFixed(3)}</strong>, Simple Good–Turing <strong>{DISCOUNT.sgt[r].toFixed(3)}</strong>,
                {' '}modified KN <strong>{DISCOUNT.mkn[Math.min(hoverR!, 3) - 1].toFixed(3)}</strong>, HPYLM <strong>{DISCOUNT.hpylm[r].toFixed(3)}</strong>,
                {' '}interpolated KN <strong>{D.toFixed(1)}</strong>.
              </>
            )}
          </div>
        )}
      </figure>
      <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
        The slider applies one D at every order (the page's tables use Ney's per-order discounts). Turing's raw
        estimate is erratic once N_(r+1) is small; the HPYLM curve is an average over 20 Gibbs sweeps.
      </p>
    </div>
  );
}

import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/lexicon-sentiment-loughran-mcdonald/lexicon_sentiment_loughran_mcdonald.py; test_viz_mirror_matches_payload
// asserts this file equals it exactly, and the harness's tests assert the claims the views make
// (test_net_tone_discards_a_coordinate, test_thm1_bound_holds, test_length_on_the_corpus,
// test_general_lexicon_misreads_finance, test_constructions_a_lexicon_cannot_read, test_negation_scope,
// test_thm2_prior_limit, test_dictionary_versus_labels, test_induction_on_the_corpus). Change the .py, regenerate,
// never the reverse. No number here comes from the real Loughran–McDonald dictionary.
import payload from './data/lexicon-sentiment-lab.json';

/**
 * Lexicon Sentiment Laboratory — six figures for the lexicon-sentiment-loughran-mcdonald topic.
 *   A. "Simplex": 600 test filings at their positive and negative hit rates, with net tone's two cutpoints or the
 *      category-world Bayes regions (Proposition 1).
 *   B. "Length": the raw score's exact sign error and Theorem 1's bound against document length, for three hit-rate
 *      presets, with the corpus's measured error.
 *   C. "Contamination": net-tone histograms by tone for the finance stand-in and the general list (Proposition 2).
 *   D. "Constructions": sentence-level lexicon accuracy by template, with and without a one-token negation scope, and
 *      document accuracy against the scope k (Definition 2).
 *   E. "Prior": learning curves of the dictionary, word power, logistic regression, and the two dictionary-centered
 *      priors, and the λ path (Theorem 2).
 *   F. "Induction": each lexicon word's propagated score from four seeds (Theorem 3).
 */

type Tone = -1 | 0 | 1;
const SIM = payload.simplex as unknown as {
  points: [number, number, Tone][]; logQ: number[][]; net: { lo: number; hi: number; acc: number };
  bayes_acc: number; lr_acc: number; slopes: Record<string, [number, number]>;
};
type Preset = { id: string; label: string; a: number; b: number; rho: number; exact: number[]; bound: number[] | null };
const LEN = payload.length as unknown as {
  L: number[]; presets: Preset[]; doc: { L: number; err: number; iid: number; bound: number };
};
type Contam = {
  hist: Record<string, number[]>; lo: number; hi: number; acc: number; sign_acc: number; snr: number;
  mean: Record<string, number>;
};
const CON = payload.contamination as unknown as { edges: number[]; finance: Contam; general: Contam; c: Record<string, number> };
type CRow = { c: string; n: number; acc: number; abstain: number; acc_k1: number };
const CST = payload.constructions as unknown as {
  rows: CRow[]; all: { acc: number; acc_k1: number };
  negation: { k: number[]; acc: number[]; far: number[]; flips: number[] };
};
type Kind = 'net' | 'categories';
const PRI = payload.prior as unknown as {
  n: number[]; net: number[]; categories: number[]; support: number[]; lr: number[]; prior_net: number[];
  prior_categories: number[]; crossover: Record<Kind, number | null>; ceiling: number;
  path: Record<Kind, { lam: string[]; acc: number[] }>;
  limit: Record<Kind, { G: number; rows: { lam: string; lam_norm: number; gap: number }[] }>;
};
type IWord = { w: string; cat: string; z: number | null; seed: boolean };
const IND = payload.induction as unknown as {
  words: IWord[]; sign_acc: number; n_polar: number; V: number; mean_abs: Record<string, number>; top: string[];
  bottom: string[]; prec_pos: number; prec_neg: number; robust: { median: number; p10: number; min: number };
  seeds: string[][]; min_df: number;
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const TONE_COLOR: Record<string, string> = {
  '-1': 'var(--color-badge-red-text)', '0': 'var(--color-badge-blue-text)', '1': 'var(--color-badge-green-text)',
};
const TONE_FILL: Record<string, string> = {
  '-1': 'var(--color-badge-red-bg)', '0': 'var(--color-badge-blue-bg)', '1': 'var(--color-badge-green-bg)',
};
const TONE_LABEL: Record<string, string> = { '-1': 'negative', '0': 'neutral', '1': 'positive' };

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

function Readout({ label, value, id, emphasis }: { label: string; value: string; id: string; emphasis?: boolean }) {
  return (
    <div style={{ fontFamily: sans, fontSize: '0.8rem', minWidth: 0 }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)', overflowWrap: 'anywhere' }}>
        {value}
      </div>
    </div>
  );
}

function Row({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', ...style }}>{children}</div>;
}

function Readouts({ children }: { children: ReactNode }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>{children}</div>;
}

function Note({ children }: { children: ReactNode }) {
  return <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>{children}</p>;
}

function Caption({ children }: { children: ReactNode }) {
  return <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>{children}</figcaption>;
}

function Legend({ items }: { items: { label: string; color: string; dash?: string; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="10" aria-hidden="true">
            {s.dot
              ? <circle cx="11" cy="5" r="3.5" style={{ fill: s.color }} />
              : <line x1="0" y1="5" x2="22" y2="5" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Slider({ id, label, value, max, onChange, text }: {
  id: string; label: string; value: number; max: number; onChange: (v: number) => void; text: string;
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
      <label htmlFor={`lex-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`lex-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

const W = 600, M = { top: 24, right: 14, bottom: 38, left: 46 };

function YGrid({ y, ticks, iw, fmt }: { y: d3.ScaleLinear<number, number>; ticks: number[]; iw: number; fmt: (t: number) => string }) {
  return (
    <>
      {ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={0} x2={iw} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.6 }} />
          <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{fmt(t)}</text>
        </g>
      ))}
    </>
  );
}

// ----- A. Simplex -----

const SH = 380, SIW = W - M.left - M.right, SIH = SH - M.top - M.bottom;
const SMAX = 3.2; // hits per 100 tokens shown on each axis

function bayesTone(pp: number, pn: number): Tone {
  // Proposition 1(a): each tone's score is affine in the two rates (rates as fractions; equal priors).
  let best = 0, arg: Tone = 0;
  SIM.logQ.forEach((lq, k) => {
    const s = pp * lq[0] + pn * lq[1] + (1 - pp - pn) * lq[2];
    if (k === 0 || s > best) { best = s; arg = (k - 1) as Tone; }
  });
  return arg;
}

function SimplexView() {
  const [overlay, setOverlay] = useState<'net' | 'bayes'>('net');
  const x = d3.scaleLinear().domain([0, SMAX]).range([0, SIW]);
  const y = d3.scaleLinear().domain([0, SMAX]).range([SIH, 0]);
  const cells: { i: number; j: number; t: Tone }[] = [];
  const G = 64;
  if (overlay === 'bayes') {
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const pp = ((i + 0.5) / G) * SMAX, pn = ((j + 0.5) / G) * SMAX;
      cells.push({ i, j, t: bayesTone(pp / 100, pn / 100) });
    }
  }
  const cw = SIW / G, ch = SIH / G;
  // Net tone's cutpoints are lines p̂₋ = p̂₊ − c (per 100 tokens), clipped to the square.
  const cut = (c: number) => {
    const x0 = Math.max(0, c), x1 = Math.min(SMAX, SMAX + c);
    return { x1: x(x0), y1: y(x0 - c), x2: x(x1), y2: y(x1 - c) };
  };
  const acc = overlay === 'net' ? SIM.net.acc : SIM.bayes_acc;
  return (
    <figure style={{ margin: 0 }} data-view="simplex">
      <Caption>Each filing is a point of the category simplex: net tone keeps one direction of it</Caption>
      <Row>
        <Pill active={overlay === 'net'} onClick={() => setOverlay('net')} data={{ 'data-overlay': 'net' }}>net-tone cutpoints</Pill>
        <Pill active={overlay === 'bayes'} onClick={() => setOverlay('bayes')} data={{ 'data-overlay': 'bayes' }}>category-world Bayes regions</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} ${SH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="simplex"
        aria-label={`600 test filings by positive and negative hits per 100 tokens; ${overlay === 'net' ? 'net-tone cutpoints' : 'category-world Bayes regions'}, test accuracy ${f3(acc)}`}>
        <text x={8} y={14} style={titleText}>600 test filings: negative hits against positive hits, per 100 tokens</text>
        <g transform={`translate(${M.left},${M.top})`}>
          {cells.map((c) => (
            <rect key={`${c.i}-${c.j}`} x={c.i * cw} y={SIH - (c.j + 1) * ch} width={cw + 0.3} height={ch + 0.3}
              style={{ fill: TONE_FILL[String(c.t)], opacity: 0.85 }} />
          ))}
          {[0, 1, 2, 3].map((t) => (
            <g key={`g${t}`}>
              <line x1={0} x2={SIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.6 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
              <line x1={x(t)} x2={x(t)} y1={0} y2={SIH} style={{ stroke: 'var(--color-border)', strokeWidth: 0.6 }} />
              <text x={x(t)} y={SIH + 14} textAnchor="middle" style={axisText}>{t}</text>
            </g>
          ))}
          {SIM.points.map(([pp, pn, t], i) => (
            pp <= SMAX && pn <= SMAX
              ? <circle key={i} cx={x(pp)} cy={y(pn)} r={2.1} style={{ fill: TONE_COLOR[String(t)], opacity: 0.75 }} />
              : null
          ))}
          {overlay === 'net' && [SIM.net.lo, SIM.net.hi].map((c) => {
            const l = cut(c);
            return <line key={c} {...l} data-cut={c} style={{ stroke: 'var(--color-accent)', strokeWidth: 1.8, strokeDasharray: '5 3' }} />;
          })}
          <text x={SIW / 2} y={SIH + 30} textAnchor="middle" style={axisText}>positive hits per 100 tokens (p̂₊)</text>
          <text transform={`translate(${-34},${SIH / 2}) rotate(-90)`} textAnchor="middle" style={axisText}>negative hits per 100 tokens (p̂₋)</text>
        </g>
      </svg>
      <Legend items={[-1, 0, 1].map((t) => ({ label: `${TONE_LABEL[String(t)]} filing`, color: TONE_COLOR[String(t)], dot: true }))} />
      <Readouts>
        <Readout label="net tone, tuned cutpoints" value={f3(SIM.net.acc)} id="simplex-net" emphasis={overlay === 'net'} />
        <Readout label="category-world Bayes" value={f3(SIM.bayes_acc)} id="simplex-bayes" emphasis={overlay === 'bayes'} />
        <Readout label="unigram logistic regression" value={f3(SIM.lr_acc)} id="simplex-lr" />
        <Readout label="slopes (u₊, u₋), positive vs neutral" value={`${f3(SIM.slopes['+1/+0'][0])}, ${f3(SIM.slopes['+1/+0'][1])}`} id="simplex-slopes" />
      </Readouts>
      <Note>
        Net tone p̂₊ − p̂₋ can only cut along the dashed diagonals, at the cutpoints {SIM.net.lo} and {SIM.net.hi} per
        100 tokens. Neutral filings sit near the origin, with few polar words of either kind, and a diagonal band cannot
        separate them from balanced filings with many. The Bayes rule's boundaries (Proposition 1) are lines too, but
        they are not parallel to the diagonal. A few filings beyond 3.2 hits per 100 tokens are off the chart.
      </Note>
    </figure>
  );
}

// ----- B. Length -----

const LH = 330, LIW = W - M.left - M.right, LIH = LH - M.top - M.bottom;

function LengthView() {
  const [pid, setPid] = useState('finance');
  const [li, setLi] = useState(LEN.L.findIndex((v) => v >= 200));
  const p = LEN.presets.find((q) => q.id === pid)!;
  const x = d3.scaleLog().domain([1, LEN.L[LEN.L.length - 1]]).range([0, LIW]);
  const y = d3.scaleLinear().domain([0, 1]).range([LIH, 0]);
  const line = d3.line<number>().x((_, i) => x(LEN.L[i])).y((v) => y(v));
  const L = LEN.L[li];
  return (
    <figure style={{ margin: 0 }} data-view="length">
      <Caption>The raw score's sign error against length: exact law and Theorem 1's bound</Caption>
      <Row>
        {LEN.presets.map((q) => (
          <Pill key={q.id} active={pid === q.id} onClick={() => setPid(q.id)} data={{ 'data-preset': q.id }}>{q.label}</Pill>
        ))}
      </Row>
      <Slider id="length" label="document length L" value={li} max={LEN.L.length - 1} onChange={setLi} text={`${L} tokens`} />
      <svg viewBox={`0 0 ${W} ${LH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="length"
        aria-label={`${p.label}: at L = ${L}, exact sign error ${f3(p.exact[li])}${p.bound ? `, bound ${f3(p.bound[li])}` : ', no bound since b > a'}`}>
        <text x={8} y={14} style={titleText}>P(raw score ≤ 0) for a positive document of L tokens (log axis)</text>
        <g transform={`translate(${M.left},${M.top})`}>
          <YGrid y={y} ticks={[0, 0.25, 0.5, 0.75, 1]} iw={LIW} fmt={(t) => t.toFixed(2)} />
          {[1, 10, 100, 1000].map((t) => (
            <text key={t} x={x(t)} y={LIH + 14} textAnchor="middle" style={axisText}>{t}</text>
          ))}
          {p.bound && <path d={line(p.bound) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 2, strokeDasharray: '5 3' }} />}
          <path d={line(p.exact) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
          <line x1={x(L)} x2={x(L)} y1={0} y2={LIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.8 }} />
          {pid === 'finance' && (
            <g data-marker="corpus">
              <circle cx={x(LEN.doc.L)} cy={y(LEN.doc.err)} r={4.5} style={{ fill: 'var(--color-text)' }} />
              <text x={x(LEN.doc.L) + 7} y={y(LEN.doc.err) - 6} style={axisText}>corpus, measured</text>
            </g>
          )}
          <text x={LIW / 2} y={LIH + 30} textAnchor="middle" style={axisText}>document length L (tokens)</text>
        </g>
      </svg>
      <Legend items={[
        { label: 'exact (i.i.d. tokens)', color: 'var(--color-accent)' },
        ...(p.bound ? [{ label: 'bound ρ^L', color: 'var(--color-accent-secondary)', dash: '5 3' }] : []),
      ]} />
      <Readouts>
        <Readout label="hit rates a, b" value={`${p.a}, ${p.b}`} id="length-rates" />
        <Readout label="ρ = 1 − (√a − √b)²" value={p.bound ? String(p.rho) : 'no bound: b > a'} id="length-rho" />
        <Readout label={`exact error at L = ${L}`} value={f3(p.exact[li])} id="length-exact" emphasis />
        <Readout label={`bound at L = ${L}`} value={p.bound ? f3(p.bound[li]) : '—'} id="length-bound" />
      </Readouts>
      <Note>
        On the corpus, positive filings average {LEN.doc.L} tokens. Their measured error is {f3(LEN.doc.err)}, the i.i.d.
        model gives {f3(LEN.doc.iid)}, and the bound gives {f3(LEN.doc.bound)}: loose, with the right exponent. With the
        general list, negative hits outnumber positive ones even in positive filings, so the error climbs toward 1 with
        length.
      </Note>
    </figure>
  );
}

// ----- C. Contamination -----

const CH = 320, CIW = W - M.left - M.right, CIH = CH - M.top - M.bottom;

function ContaminationView() {
  const [lex, setLex] = useState<'finance' | 'general'>('finance');
  const d = CON[lex];
  const E = CON.edges;
  const x = d3.scaleLinear().domain([E[0], E[E.length - 1]]).range([0, CIW]);
  const totals = Object.fromEntries(Object.entries(d.hist).map(([t, h]) => [t, h.reduce((a, b) => a + b, 0)]));
  const maxShare = Math.max(...Object.entries(d.hist).flatMap(([t, h]) => h.map((v) => v / totals[t])),
    ...Object.entries(CON.finance.hist).flatMap(([t, h]) => h.map((v) => v / totals[t])));
  const y = d3.scaleLinear().domain([0, Math.ceil(maxShare * 20) / 20]).range([CIH, 0]);
  const step = (h: number[], t: string) => {
    const pts: [number, number][] = [];
    h.forEach((v, i) => { pts.push([x(E[i]), y(v / totals[t])]); pts.push([x(E[i + 1]), y(v / totals[t])]); });
    return d3.line()(pts) ?? '';
  };
  return (
    <figure style={{ margin: 0 }} data-view="contamination">
      <Caption>Counting "tax" and "liability" as negative slides every tone to the left and widens it</Caption>
      <Row>
        <Pill active={lex === 'finance'} onClick={() => setLex('finance')} data={{ 'data-lexicon': 'finance' }}>finance stand-in</Pill>
        <Pill active={lex === 'general'} onClick={() => setLex('general')} data={{ 'data-lexicon': 'general' }}>general-purpose list</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="contamination"
        aria-label={`${lex} list: three-class accuracy ${f3(d.acc)}, Fisher ratio ${f3(d.snr)}`}>
        <text x={8} y={14} style={titleText}>share of test filings by net tone (per 100 tokens), one curve per tone</text>
        <g transform={`translate(${M.left},${M.top})`}>
          <YGrid y={y} ticks={y.ticks(4)} iw={CIW} fmt={(t) => t.toFixed(2)} />
          {[-6, -4, -2, 0, 2, 4, 6].map((t) => (
            <text key={t} x={x(t)} y={CIH + 14} textAnchor="middle" style={axisText}>{t}</text>
          ))}
          {['-1', '0', '1'].map((t) => (
            <path key={t} d={step(d.hist[t], t)} data-tone={t} style={{ fill: 'none', stroke: TONE_COLOR[t], strokeWidth: 2 }} />
          ))}
          {[d.lo, d.hi].map((c) => (
            <line key={c} x1={x(c)} x2={x(c)} y1={0} y2={CIH} style={{ stroke: 'var(--color-accent)', strokeWidth: 1.4, strokeDasharray: '5 3' }} />
          ))}
          <line x1={x(0)} x2={x(0)} y1={0} y2={CIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.6 }} />
          <text x={CIW / 2} y={CIH + 30} textAnchor="middle" style={axisText}>net tone (pos − neg) per 100 tokens; the outer bins hold the tails</text>
        </g>
      </svg>
      <Legend items={[
        ...['-1', '0', '1'].map((t) => ({ label: `${TONE_LABEL[t]} filings`, color: TONE_COLOR[t] })),
        { label: 'tuned cutpoints', color: 'var(--color-accent)', dash: '5 3' },
      ]} />
      <Readouts>
        <Readout label="three-class accuracy" value={f3(d.acc)} id="contam-acc" emphasis />
        <Readout label="polar sign accuracy" value={f3(d.sign_acc)} id="contam-sign" />
        <Readout label="Fisher ratio, + vs −" value={f3(d.snr)} id="contam-snr" />
        <Readout label="mean net tone (−, 0, +)" value={['-1', '0', '1'].map((t) => d.mean[t]).join(', ')} id="contam-mean" />
      </Readouts>
      <Note>
        The eight polysemes add {CON.c['-1']}, {CON.c['0']}, and {CON.c['1']} negative hits per 100 tokens to negative,
        neutral, and positive filings: a nearly constant shift, which re-tuned cutpoints follow. The extra variance stays
        (Proposition 2), and accuracy falls from {f3(CON.finance.acc)} to {f3(CON.general.acc)}.
      </Note>
    </figure>
  );
}

// ----- D. Constructions -----

const DH = 300, DIW = W - 170 - M.right, DROW = 30;

function ConstructionsView() {
  const [k1, setK1] = useState(false);
  const x = d3.scaleLinear().domain([0, 1]).range([0, DIW]);
  const NG = CST.negation;
  const nx = d3.scalePoint<number>().domain(NG.k).range([0, 220]).padding(0.3);
  const ny = d3.scaleLinear().domain([0.6, 0.72]).range([90, 0]);
  return (
    <figure style={{ margin: 0 }} data-view="constructions">
      <Caption>What a word list reads, sentence by sentence</Caption>
      <Row>
        <Pill active={!k1} onClick={() => setK1(false)} data={{ 'data-scope': '0' }}>no negation scope</Pill>
        <Pill active={k1} onClick={() => setK1(true)} data={{ 'data-scope': '1' }}>scope k = 1</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} ${DH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="constructions"
        aria-label={`sentence-level lexicon accuracy by construction, ${k1 ? 'with' : 'without'} negation scope; overall ${f3(k1 ? CST.all.acc_k1 : CST.all.acc)}`}>
        <text x={8} y={14} style={titleText}>share of single-polarity aspect sentences the lexicon's sign gets right</text>
        <g transform="translate(160,28)">
          {CST.rows.map((r, i) => {
            const a = k1 ? r.acc_k1 : r.acc;
            return (
              <g key={r.c} transform={`translate(0,${i * DROW})`} data-row={r.c}>
                <text x={-8} y={14} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{r.c}</text>
                <rect x={0} y={3} width={DIW} height={16} style={{ fill: 'var(--color-muted-bg)' }} />
                <rect x={0} y={3} width={x(a)} height={16} style={{ fill: 'var(--color-accent)' }} />
                <text x={a > 0.5 ? x(a) - 5 : x(a) + 5} y={15} textAnchor={a > 0.5 ? 'end' : 'start'}
                  style={{ ...axisText, fill: a > 0.5 ? 'var(--color-bg)' : 'var(--color-text)' }}>
                  {f3(a)}{r.abstain === 1 ? ' (abstains)' : ''} · n = {r.n}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <Readouts>
        <Readout label="all sentences" value={f3(k1 ? CST.all.acc_k1 : CST.all.acc)} id="cst-all" emphasis />
        <Readout label="negated adjectives" value={f3(CST.rows.find((r) => r.c === 'negated adjective')![k1 ? 'acc_k1' : 'acc'])} id="cst-neg" />
        <Readout label="cost-like 'exceeded expectations'" value={f3(CST.rows.find((r) => r.c === 'verb, cost-like')!.acc)} id="cst-trap" />
      </Readouts>
      <svg viewBox="0 0 300 140" style={{ width: '100%', maxWidth: '22rem', height: 'auto', marginTop: '0.75rem' }} role="img" data-panel="negation"
        aria-label={`document accuracy by negation scope: ${NG.k.map((k, i) => `k = ${k}: ${f3(NG.acc[i])}`).join(', ')}`}>
        <text x={4} y={12} style={titleText}>document accuracy against scope k</text>
        <g transform="translate(48,24)">
          {[0.6, 0.66, 0.72].map((t) => (
            <g key={t}>
              <line x1={0} x2={220} y1={ny(t)} y2={ny(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.6 }} />
              <text x={-6} y={ny(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(2)}</text>
            </g>
          ))}
          {NG.k.map((k, i) => (
            <g key={k}>
              <circle cx={nx(k)} cy={ny(NG.acc[i])} r={3.5} style={{ fill: 'var(--color-accent)' }} />
              <text x={nx(k)} y={106} textAnchor="middle" style={axisText}>{k}</text>
              {NG.far[i] > 0 && <text x={nx(k)} y={ny(NG.acc[i]) - 7} textAnchor="middle" style={axisText}>{NG.far[i]} false flips</text>}
            </g>
          ))}
        </g>
      </svg>
      <Note>
        A bare direction ("costs rose 5 percent") has no lexicon word, so the lexicon abstains. "Exceeded expectations"
        is bad news for a cost and is written with a positive word. A one-token negation scope repairs every "not weak"
        on this corpus, and wider scopes only start flipping the wrong words at k = 8.
      </Note>
    </figure>
  );
}

// ----- E. Prior -----

const PH = 340, PIW = W - M.left - M.right, PIH = PH - M.top - M.bottom;
const METHODS: { key: keyof typeof PRI & string; label: string; color: string; dash?: string }[] = [
  { key: 'net', label: 'dictionary: net tone', color: 'var(--color-accent)', dash: '2 3' },
  { key: 'categories', label: 'dictionary: two counts', color: 'var(--color-accent-secondary)', dash: '2 3' },
  { key: 'support', label: 'LR on lexicon words', color: 'var(--color-badge-amber-text)' },
  { key: 'lr', label: 'unigram LR', color: 'var(--color-text-secondary)' },
  { key: 'prior_net', label: 'prior toward net tone', color: 'var(--color-accent)' },
  { key: 'prior_categories', label: 'prior toward two counts', color: 'var(--color-accent-secondary)' },
];

function PriorView() {
  const [mode, setMode] = useState<'curves' | 'path'>('curves');
  const [kind, setKind] = useState<Kind>('net');
  const [ni, setNi] = useState(PRI.n.length - 1);
  const [pi, setPi] = useState(PRI.path.net.lam.indexOf('0.1'));
  const y = d3.scaleLinear().domain([0.35, 0.9]).range([PIH, 0]);
  const x = d3.scaleLog().domain([PRI.n[0], PRI.n[PRI.n.length - 1]]).range([0, PIW]);
  const line = d3.line<number>().x((_, i) => x(PRI.n[i])).y((v) => y(v));
  const path = PRI.path[kind];
  const px = d3.scalePoint<number>().domain(path.lam.map((_, i) => i)).range([0, PIW]).padding(0.2);
  const lim = PRI.limit[kind];
  return (
    <figure style={{ margin: 0 }} data-view="prior">
      <Caption>A dictionary as a prior: accuracy against the number of labeled filings</Caption>
      <Row>
        <Pill active={mode === 'curves'} onClick={() => setMode('curves')} data={{ 'data-mode': 'curves' }}>learning curves</Pill>
        <Pill active={mode === 'path'} onClick={() => setMode('path')} data={{ 'data-mode': 'path' }}>λ path at n = 1,500</Pill>
      </Row>
      {mode === 'curves'
        ? <Slider id="prior-n" label="labeled training filings n" value={ni} max={PRI.n.length - 1} onChange={setNi} text={PRI.n[ni].toLocaleString('en-US')} />
        : (
          <>
            <Row>
              <Pill active={kind === 'net'} onClick={() => setKind('net')} data={{ 'data-kind': 'net' }}>toward net tone</Pill>
              <Pill active={kind === 'categories'} onClick={() => setKind('categories')} data={{ 'data-kind': 'categories' }}>toward two counts</Pill>
            </Row>
            <Slider id="prior-lam" label="penalty λ" value={pi} max={path.lam.length - 1} onChange={setPi} text={path.lam[pi]} />
          </>
        )}
      <svg viewBox={`0 0 ${W} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="prior"
        aria-label={mode === 'curves'
          ? `at n = ${PRI.n[ni]}: ${METHODS.map((m) => `${m.label} ${f3((PRI[m.key] as number[])[ni])}`).join(', ')}`
          : `prior toward ${kind === 'net' ? 'net tone' : 'two counts'} at λ = ${path.lam[pi]}: ${f3(path.acc[pi])}`}>
        <text x={8} y={14} style={titleText}>{mode === 'curves' ? 'test accuracy against n (log axis); every λ chosen on 600 dev filings' : 'test accuracy against λ, from the dictionary end (left) to ridge LR (right)'}</text>
        <g transform={`translate(${M.left},${M.top})`}>
          <YGrid y={y} ticks={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} iw={PIW} fmt={(t) => t.toFixed(1)} />
          <line x1={0} x2={PIW} y1={y(PRI.ceiling)} y2={y(PRI.ceiling)} style={{ stroke: 'var(--color-text)', strokeWidth: 1, strokeDasharray: '1 3' }} />
          <text x={PIW - 4} y={y(PRI.ceiling) - 5} textAnchor="end" style={axisText}>full-text Bayes ceiling {f3(PRI.ceiling)}</text>
          {mode === 'curves' ? (
            <>
              {PRI.n.map((n) => <text key={n} x={x(n)} y={PIH + 14} textAnchor="middle" style={axisText}>{n}</text>)}
              {METHODS.map((m) => (
                <path key={m.key} d={line(PRI[m.key] as number[]) ?? ''} data-method={m.key}
                  style={{ fill: 'none', stroke: m.color, strokeWidth: m.key.startsWith('prior') ? 2.4 : 1.6, strokeDasharray: m.dash }} />
              ))}
              <line x1={x(PRI.n[ni])} x2={x(PRI.n[ni])} y1={0} y2={PIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.8 }} />
              <text x={PIW / 2} y={PIH + 30} textAnchor="middle" style={axisText}>labeled training filings n</text>
            </>
          ) : (
            <>
              {path.lam.map((l, i) => <text key={l} x={px(i)} y={PIH + 14} textAnchor="middle" style={axisText}>{l}</text>)}
              <path d={d3.line<number>().x((_, i) => px(i)!).y((v) => y(v))(path.acc) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
              {path.acc.map((a, i) => <circle key={i} cx={px(i)} cy={y(a)} r={i === pi ? 5 : 3} style={{ fill: 'var(--color-accent)' }} />)}
              <text x={PIW / 2} y={PIH + 30} textAnchor="middle" style={axisText}>penalty λ on the deviation from the dictionary</text>
            </>
          )}
        </g>
      </svg>
      {mode === 'curves' && <Legend items={METHODS.map((m) => ({ label: m.label, color: m.color, dash: m.dash }))} />}
      <Readouts>
        {mode === 'curves' ? (
          <>
            <Readout label={`dictionary, two counts (n = ${PRI.n[ni]})`} value={f3(PRI.categories[ni])} id="prior-cat" />
            <Readout label="unigram LR" value={f3(PRI.lr[ni])} id="prior-lr" />
            <Readout label="prior toward net tone" value={f3(PRI.prior_net[ni])} id="prior-pn" emphasis />
            <Readout label="LR first matches the dictionary at n =" value={String(PRI.crossover.categories)} id="prior-cross" />
          </>
        ) : (
          <>
            <Readout label={`accuracy at λ = ${path.lam[pi]}`} value={f3(path.acc[pi])} id="prior-path" emphasis />
            <Readout label="G = mean ‖x‖" value={f3(lim.G)} id="prior-G" />
            {lim.rows.map((r) => (
              <Readout key={r.lam} label={`λ = ${r.lam}: λ·max‖Δ‖, gap`} value={`${f3(r.lam_norm)}, ${r.gap}`} id={`prior-lim-${r.lam}`} />
            ))}
          </>
        )}
      </Readouts>
      <Note>
        {mode === 'curves'
          ? 'Dotted lines are the dictionary with fitted cutpoints (λ = ∞). Solid thick lines shrink logistic regression toward it instead of toward zero; they never fall more than 0.01 below the better end.'
          : 'Theorem 2: the deviation from the dictionary is at most G/λ, and the probabilities converge to the dictionary’s at rate 1/λ (the gap column). Small λ is ordinary ridge logistic regression.'}
      </Note>
    </figure>
  );
}

// ----- F. Induction -----

const IH = 300, IIW = W - 110 - M.right;
const IROWS = ['positive', 'negative', 'uncertainty', 'litigious', 'polyseme'];
const ICOLOR: Record<string, string> = {
  positive: 'var(--color-badge-green-text)', negative: 'var(--color-badge-red-text)', uncertainty: 'var(--color-badge-blue-text)',
  litigious: 'var(--color-badge-amber-text)', polyseme: 'var(--color-accent)',
};

function InductionView() {
  const [sel, setSel] = useState('tax');
  const x = d3.scaleLinear().domain([-3, 3]).range([0, IIW]);
  const word = IND.words.find((w) => w.w === sel) ?? IND.words[0];
  return (
    <figure style={{ margin: 0 }} data-view="induction">
      <Caption>Propagating polarity from four seed words over document co-occurrence</Caption>
      <svg viewBox={`0 0 ${W} ${IH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="induction"
        aria-label={`harmonic scores z of lexicon words; sign accuracy on polar words ${f3(IND.sign_acc)}; selected ${word.w}${word.z === null ? ', a seed' : ` z = ${word.z}`}`}>
        <text x={8} y={14} style={titleText}>harmonic score z (median-centered, sd-scaled) of every stand-in word; tap a dot</text>
        <g transform="translate(100,30)">
          {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={IROWS.length * 46} style={{ stroke: t === 0 ? 'var(--color-text-secondary)' : 'var(--color-border)', strokeWidth: t === 0 ? 1 : 0.6 }} />
              <text x={x(t)} y={IROWS.length * 46 + 14} textAnchor="middle" style={axisText}>{t}</text>
            </g>
          ))}
          {IROWS.map((cat, r) => (
            <g key={cat} transform={`translate(0,${r * 46 + 20})`}>
              <text x={-8} y={4} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{cat}</text>
              {IND.words.filter((w) => w.cat === cat).map((w, i) => {
                const zc = w.z === null ? (cat === 'positive' ? 3 : -3) : Math.max(-3, Math.min(3, w.z));
                return (
                  <circle key={w.w} cx={x(zc)} cy={((i % 3) - 1) * 7} r={w.w === sel ? 6 : 4.2} data-word={w.w}
                    onClick={() => setSel(w.w)} onMouseEnter={() => setSel(w.w)}
                    style={{ fill: w.seed ? 'var(--color-bg)' : ICOLOR[cat], stroke: ICOLOR[cat], strokeWidth: w.seed ? 2.2 : 0.8, cursor: 'pointer', opacity: 0.9 }}>
                    <title>{w.z === null ? `${w.w}: seed` : `${w.w}: z = ${w.z}`}</title>
                  </circle>
                );
              })}
            </g>
          ))}
          <text x={IIW / 2} y={IROWS.length * 46 + 30} textAnchor="middle" style={axisText}>z, clipped at ±3; the rings at the edges are the four seeds</text>
        </g>
      </svg>
      <Readouts>
        <Readout label="selected word" value={word.z === null ? `${word.w} (${word.cat} seed)` : `${word.w} (${word.cat}): z = ${word.z}`} id="induce-word" emphasis />
        <Readout label={`sign right on ${IND.n_polar} polar words`} value={f3(IND.sign_acc)} id="induce-sign" />
        <Readout label="mean |z|: polar, polysemes" value={`${f3(IND.mean_abs.polar)}, ${f3(IND.mean_abs.polysemes)}`} id="induce-abs" />
        <Readout label="200 random seed sets, median" value={f3(IND.robust.median)} id="induce-robust" />
      </Readouts>
      <Note>
        Seeds: {IND.seeds[0].join(', ')} / {IND.seeds[1].join(', ')}, over the {IND.V} word types in at least {IND.min_df} training
        filings. The signs of polar words come out mostly right. But the polysemes land as far from zero as polar words do,
        and the ten most positive words in the whole vocabulary include only one lexicon word: {IND.top.slice(0, 5).join(', ')}, ….
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'simplex', label: 'Simplex' },
  { key: 'length', label: 'Length' },
  { key: 'contamination', label: 'Contamination' },
  { key: 'constructions', label: 'Constructions' },
  { key: 'prior', label: 'Prior' },
  { key: 'induction', label: 'Induction' },
] as const;

export default function LexiconSentimentLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('simplex');
  return (
    <div data-lab="lexicon-sentiment" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'simplex' && <SimplexView />}
      {view === 'length' && <LengthView />}
      {view === 'contamination' && <ContaminationView />}
      {view === 'constructions' && <ConstructionsView />}
      {view === 'prior' && <PriorView />}
      {view === 'induction' && <InductionView />}
    </div>
  );
}

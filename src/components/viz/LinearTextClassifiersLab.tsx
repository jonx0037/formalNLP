import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/linear-text-classifiers/linear_text_classifiers.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_generative_discriminative_regimes, test_bursty_world_keeps_lr_well_specified,
// test_smoothing_bounds_log_count_ratios, test_l2_path_turns_toward_max_margin, test_finance_tone, test_label_noise,
// test_aspect_xor). Change the .py, regenerate, never the reverse.
import payload from './data/linear-text-classifiers-lab.json';

/**
 * Linear Text Classifiers Laboratory — four figures for the linear-text-classifiers topic.
 *   A. "Crossover": NB and LR test accuracy against the training size in four simulated worlds (V = 101 or 3,071;
 *      naive Bayes true, or bursty), with the Bayes accuracy and NB's asymptote (Proposition 2).
 *   B. "Funnel": every plotted word's NB log-count ratio, or LR weight, against its count, with Proposition 3(b)'s
 *      bound, at three smoothing strengths or three penalties.
 *   C. "Separation": the L2 path's norm, cosine to the SVM direction, and normalized margin as λ falls, and gradient
 *      descent's cosine as t grows (Theorem 4).
 *   D. "Finance tone": the method × features table against the full-text ceiling, each cell's confusion matrix and
 *      calibration, accuracy under label noise, and the aspect-XOR weights (Proposition 4).
 */

type Band = { mean: number[]; lo: number[]; hi: number[] };
type World = { V: number; bursty: boolean; bayes: number; nb_inf: number; crossover: number | null; nb: Band; lr: Band };
const CURVES = payload.curves as unknown as { n: number[]; worlds: World[] };
const NS = CURVES.n;
type Funnel = {
  alphas: string[]; lambdas: string[]; words: string[]; count: number[]; informative: boolean[];
  r: Record<string, number[]>; bound: Record<string, number[]>; lr: Record<string, number[]>;
};
const FUN = payload.funnel as unknown as Funnel;
type PathPt = { lam: string; norm: number; cos: number; margin: number };
type GdPt = { t: number; norm: number; cos: number; loss: number };
const SEP = payload.separation as unknown as { n: number; V: number; svm_margin: number; slope: number; path: PathPt[]; gd: GdPt[] };
type Method = { acc: number; confusion: number[][]; hp: string; conf: number; ece: number; V: number };
const FIN = payload.finance as unknown as {
  ceiling: number; n_test: number; methods: Record<string, Method>;
  noise: { levels: number[]; nb: number[]; lr: number[]; nbw: number[] };
  xor: Record<string, { uni: number | null; bi: number | null }>;
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const NB_COLOR = 'var(--color-accent)';
const LR_COLOR = 'var(--color-accent-secondary)';
const NBW_COLOR = 'var(--color-badge-green-text)';

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

function Legend({ items }: { items: { label: string; color: string; dash?: string; band?: boolean; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="10" aria-hidden="true">
            {s.band
              ? <rect x="0" y="1" width="22" height="8" style={{ fill: s.color, opacity: 0.25 }} />
              : s.dot
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
      <label htmlFor={`ltc-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`ltc-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

const W = 600, M = { top: 24, right: 14, bottom: 36, left: 44 };

// ----- A. Crossover -----

const AH = 330, AIW = W - M.left - M.right, AIH = AH - M.top - M.bottom;

function CrossoverChart({ w, ni }: { w: World; ni: number }) {
  const x = d3.scaleLog().base(2).domain([NS[0], NS[NS.length - 1]]).range([0, AIW]);
  const y = d3.scaleLinear().domain([0.45, 0.9]).range([AIH, 0]);
  const line = d3.line<number>().x((_, i) => x(NS[i])).y((v) => y(v));
  const band = (b: Band) => d3.area<number>().x((_, i) => x(NS[i])).y0((_, i) => y(b.lo[i])).y1((_, i) => y(b.hi[i]))(b.lo) ?? '';
  return (
    <svg viewBox={`0 0 ${W} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="crossover"
      aria-label={`V = ${w.V}, ${w.bursty ? 'bursty' : 'naive Bayes true'}: at n = ${NS[ni]}, naive Bayes ${f3(w.nb.mean[ni])}, logistic regression ${f3(w.lr.mean[ni])}`}>
      <text x={8} y={14} style={titleText}>test accuracy against training documents (mean of 4 draws, band = min to max)</text>
      <g transform={`translate(${M.left},${M.top})`}>
        {[0.5, 0.6, 0.7, 0.8, 0.9].map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={AIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(1)}</text>
          </g>
        ))}
        {NS.map((n) => (
          <text key={`x${n}`} x={x(n)} y={AIH + 14} textAnchor="middle" style={axisText}>{n >= 1000 ? `${n / 1024}k` : n}</text>
        ))}
        <line x1={0} x2={AIW} y1={y(w.bayes)} y2={y(w.bayes)} data-series="bayes"
          style={{ stroke: 'var(--color-text)', strokeWidth: 1.2, strokeDasharray: '6 3' }} />
        {w.bursty && (
          <line x1={0} x2={AIW} y1={y(w.nb_inf)} y2={y(w.nb_inf)} data-series="nb-inf"
            style={{ stroke: NB_COLOR, strokeWidth: 1.2, strokeDasharray: '2 3' }} />
        )}
        {w.crossover !== null && (
          <line x1={x(w.crossover)} x2={x(w.crossover)} y1={0} y2={AIH} data-series="crossover"
            style={{ stroke: 'var(--color-badge-red-text)', strokeWidth: 1.2, strokeDasharray: '3 3' }} />
        )}
        <line x1={x(NS[ni])} x2={x(NS[ni])} y1={0} y2={AIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.8 }} />
        <path d={band(w.nb)} style={{ fill: NB_COLOR, opacity: 0.18 }} />
        <path d={band(w.lr)} style={{ fill: LR_COLOR, opacity: 0.18 }} />
        <path d={line(w.nb.mean) ?? ''} data-series="nb" style={{ fill: 'none', stroke: NB_COLOR, strokeWidth: 2 }} />
        <path d={line(w.lr.mean) ?? ''} data-series="lr" style={{ fill: 'none', stroke: LR_COLOR, strokeWidth: 2 }} />
        <circle cx={x(NS[ni])} cy={y(w.nb.mean[ni])} r={4} style={{ fill: NB_COLOR }} />
        <circle cx={x(NS[ni])} cy={y(w.lr.mean[ni])} r={4} style={{ fill: LR_COLOR }} />
      </g>
      <text x={M.left + AIW / 2} y={AH - 4} textAnchor="middle" style={axisText}>training documents n (log scale)</text>
    </svg>
  );
}

function CrossoverView() {
  const [V, setV] = useState(CURVES.worlds[0].V);
  const [bursty, setBursty] = useState(false);
  const [ni, setNi] = useState(NS.indexOf(512));
  const w = CURVES.worlds.find((c) => c.V === V && c.bursty === bursty)!;
  const Vs = [...new Set(CURVES.worlds.map((c) => c.V))];
  return (
    <figure style={{ margin: 0 }} data-view="crossover">
      <Caption>A · Crossover — naive Bayes leads early; logistic regression overtakes it only where its asymptote is better</Caption>
      <Row>
        {Vs.map((v) => (
          <Pill key={v} active={V === v} onClick={() => setV(v)} data={{ 'data-v': String(v) }}>{`V = ${v.toLocaleString('en-US')}`}</Pill>
        ))}
      </Row>
      <Row>
        <Pill active={!bursty} onClick={() => setBursty(false)} data={{ 'data-world': 'nb-true' }}>naive Bayes true</Pill>
        <Pill active={bursty} onClick={() => setBursty(true)} data={{ 'data-world': 'bursty' }}>bursty (half the words × 4)</Pill>
      </Row>
      <Slider id="n" label="training documents n" value={ni} max={NS.length - 1} onChange={setNi} text={NS[ni].toLocaleString('en-US')} />
      <CrossoverChart w={w} ni={ni} />
      <Legend items={[
        { label: 'naive Bayes', color: NB_COLOR },
        { label: 'logistic regression', color: LR_COLOR },
        { label: 'Bayes accuracy', color: 'var(--color-text)', dash: '6 3' },
        ...(w.bursty ? [{ label: 'naive Bayes asymptote', color: NB_COLOR, dash: '2 3' }] : []),
        ...(w.crossover !== null ? [{ label: 'crossover', color: 'var(--color-badge-red-text)', dash: '3 3' }] : []),
      ]} />
      <Readouts>
        <Readout id="nb-acc" label={`naive Bayes at n = ${NS[ni].toLocaleString('en-US')}`} value={f3(w.nb.mean[ni])} />
        <Readout id="lr-acc" label={`logistic regression at n = ${NS[ni].toLocaleString('en-US')}`} value={f3(w.lr.mean[ni])} />
        <Readout id="bayes" label="Bayes accuracy" value={f3(w.bayes)} />
        <Readout id="nb-inf" label="naive Bayes asymptote" value={f3(w.nb_inf)} />
        <Readout id="crossover" label="crossover n" value={w.crossover === null ? 'none' : w.crossover.toLocaleString('en-US')} emphasis />
      </Readouts>
      <Note>
        When naive Bayes is the true model, its asymptote is the Bayes accuracy and it leads at every n. When half the
        words come in bursts of four, logistic regression stays well specified while naive Bayes overcounts the bursty
        words (Proposition 2): naive Bayes still leads early, and logistic regression passes it once the gap between
        their asymptotes outweighs the gap in how fast they learn.
      </Note>
    </figure>
  );
}

// ----- B. Funnel -----

const BH = 340, BIW = W - M.left - M.right, BIH = BH - M.top - M.bottom;
const informativeIdx = FUN.words.map((_, i) => i).filter((i) => FUN.informative[i]);

function FunnelView() {
  const [mode, setMode] = useState<'nb' | 'lr'>('nb');
  const [ai, setAi] = useState(FUN.alphas.indexOf('1'));
  const [li, setLi] = useState(FUN.lambdas.indexOf('0.001'));
  const [hover, setHover] = useState<number | null>(null);
  const key = mode === 'nb' ? FUN.alphas[ai] : FUN.lambdas[li];
  const vals = mode === 'nb' ? FUN.r[key] : FUN.lr[key];
  const x = d3.scaleLog().domain([1, d3.max(FUN.count)! * 1.2]).range([0, BIW]);
  const lim = Math.max(3, d3.max(vals.map(Math.abs))! * 1.05);
  const y = d3.scaleLinear().domain([-lim, lim]).range([BIH, 0]);
  const bound = mode === 'nb'
    ? [...new Map(FUN.count.map((c, i) => [c, FUN.bound[key][i]])).entries()].sort((a, b) => a[0] - b[0])
    : [];
  const top = [...informativeIdx].sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a])).slice(0, 3);
  const h = hover ?? top[0];
  return (
    <figure style={{ margin: 0 }} data-view="funnel">
      <Caption>B · Funnel — rare words get the most extreme naive Bayes weights, until smoothing or a penalty pulls them in</Caption>
      <Row>
        <Pill active={mode === 'nb'} onClick={() => setMode('nb')} data={{ 'data-mode': 'nb' }}>naive Bayes log-count ratio</Pill>
        <Pill active={mode === 'lr'} onClick={() => setMode('lr')} data={{ 'data-mode': 'lr' }}>logistic regression weight</Pill>
      </Row>
      <Row style={{ alignItems: 'center' }}>
        {mode === 'nb'
          ? (<>
            <span style={{ fontFamily: sans, fontSize: '0.8rem' }}>smoothing α</span>
            {FUN.alphas.map((a, i) => (
              <Pill key={a} active={ai === i} onClick={() => setAi(i)} data={{ 'data-alpha': a }}>{a}</Pill>
            ))}
          </>)
          : (<>
            <span style={{ fontFamily: sans, fontSize: '0.8rem' }}>penalty λ</span>
            {FUN.lambdas.map((l, i) => (
              <Pill key={l} active={li === i} onClick={() => setLi(i)} data={{ 'data-lambda': l }}>{l}</Pill>
            ))}
          </>)}
      </Row>
      <svg viewBox={`0 0 ${W} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="funnel"
        aria-label={`${mode === 'nb' ? `Log-count ratios at α = ${key}` : `Logistic regression weights at λ = ${key}`}, ${FUN.words.length} words`}>
        <text x={8} y={14} style={titleText}>
          {mode === 'nb' ? 'log-count ratio r against the word’s count; lines mark Proposition 3(b)’s bound' : 'logistic regression weight (positive minus negative) against the word’s count'}
        </text>
        <g transform={`translate(${M.left},${M.top})`}>
          {y.ticks(6).map((t) => (
            <g key={`y${t}`}>
              <line x1={0} x2={BIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: t === 0 ? 1 : 0.5 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
            </g>
          ))}
          {[1, 10, 100, 1000, 10000].filter((t) => t <= x.domain()[1]).map((t) => (
            <text key={`x${t}`} x={x(t)} y={BIH + 14} textAnchor="middle" style={axisText}>{t.toLocaleString('en-US')}</text>
          ))}
          {bound.length > 0 && (
            <>
              <path d={d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]))(bound) ?? ''} data-series="bound-hi"
                style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
              <path d={d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(-d[1]))(bound) ?? ''} data-series="bound-lo"
                style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
            </>
          )}
          {FUN.words.map((wd, i) => (
            <circle key={wd} cx={x(FUN.count[i])} cy={y(vals[i])} r={FUN.informative[i] ? 3.2 : 2.2}
              data-word={wd} onMouseEnter={() => setHover(i)}
              style={{
                fill: FUN.informative[i] ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                fillOpacity: FUN.informative[i] ? 0.85 : 0.3, cursor: 'pointer',
              }} />
          ))}
          <circle cx={x(FUN.count[h])} cy={y(vals[h])} r={6} style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.5 }} />
        </g>
        <text x={M.left + BIW / 2} y={BH - 4} textAnchor="middle" style={axisText}>the word’s count in 1,000 polar training documents (log scale)</text>
      </svg>
      <Legend items={[
        { label: `informative words (${informativeIdx.length})`, color: 'var(--color-accent)', dot: true },
        { label: `other words (${FUN.words.length - informativeIdx.length}, sampled)`, color: 'var(--color-text-secondary)', dot: true },
        ...(mode === 'nb' ? [{ label: 'bound ± (ln((c + α)/α) + const)', color: 'var(--color-text-secondary)', dash: '4 3' }] : []),
      ]} />
      <Readouts>
        <Readout id="word" label="word (hover a point)" value={`${FUN.words[h]} · count ${FUN.count[h]}`} />
        <Readout id="weight" label={mode === 'nb' ? 'log-count ratio' : 'weight'} value={f3(vals[h])} emphasis />
        {mode === 'nb' && <Readout id="bound" label="its bound" value={f3(FUN.bound[key][h])} />}
        <Readout id="top" label="most extreme informative words" value={top.map((i) => `${FUN.words[i]} ${f3(vals[i])}`).join(', ')} />
      </Readouts>
      <Note>
        A word seen once can carry a log-count ratio of at most ln(1 + 1/α) plus a constant, and the points on the left
        edge reach that bound. Logistic regression shrinks the whole cloud of uninformative words toward zero and keeps
        the lexicon words, which is what naive Bayes cannot do on its own.
      </Note>
    </figure>
  );
}

// ----- C. Separation -----

const CH = 210, CIW = W - M.left - M.right, CIH = CH - M.top - M.bottom;
const decades = SEP.path.map((p) => -Math.log10(Number(p.lam)));

function PathChart({ pi, kind }: { pi: number; kind: 'norm' | 'direction' }) {
  const x = d3.scaleLinear().domain([decades[0], decades[decades.length - 1]]).range([0, CIW]);
  const y = kind === 'norm'
    ? d3.scaleLinear().domain([0, d3.max(SEP.path, (p) => p.norm)! * 1.1]).range([CIH, 0])
    : d3.scaleLinear().domain([0, 1]).range([CIH, 0]);
  const line = (vals: number[]) => d3.line<number>().x((_, i) => x(decades[i])).y((v) => y(v))(vals) ?? '';
  const last = SEP.path[SEP.path.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={`path-${kind}`}
      aria-label={kind === 'norm' ? `At λ = ${SEP.path[pi].lam}: norm ${f3(SEP.path[pi].norm)}` : `At λ = ${SEP.path[pi].lam}: cosine ${f3(SEP.path[pi].cos)}, normalized margin ${f3(SEP.path[pi].margin)}`}>
      <text x={8} y={14} style={titleText}>
        {kind === 'norm' ? '‖ŵ_λ‖ as λ falls; dashed: slope ln 10/γ* per decade through the last point' : 'cosine to the SVM direction, and normalized margin ÷ γ*'}
      </text>
      <g transform={`translate(${M.left},${M.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={CIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {SEP.path.map((p, i) => (
          <text key={`x${p.lam}`} x={x(decades[i])} y={CIH + 14} textAnchor="middle" style={axisText}>{`1e-${decades[i]}`}</text>
        ))}
        <line x1={x(decades[pi])} x2={x(decades[pi])} y1={0} y2={CIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.8 }} />
        {kind === 'norm' ? (
          <>
            <line x1={x(decades[0])} x2={x(decades[decades.length - 1])}
              y1={y(last.norm - SEP.slope * (decades[decades.length - 1] - decades[0]))} y2={y(last.norm)}
              style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '4 3' }} />
            <path d={line(SEP.path.map((p) => p.norm))} data-series="norm" style={{ fill: 'none', stroke: LR_COLOR, strokeWidth: 2 }} />
            {SEP.path.map((p, i) => <circle key={p.lam} cx={x(decades[i])} cy={y(p.norm)} r={i === pi ? 4.5 : 2.5} style={{ fill: LR_COLOR }} />)}
          </>
        ) : (
          <>
            <path d={line(SEP.path.map((p) => p.cos))} data-series="cos" style={{ fill: 'none', stroke: LR_COLOR, strokeWidth: 2 }} />
            <path d={line(SEP.path.map((p) => p.margin / SEP.svm_margin))} data-series="margin"
              style={{ fill: 'none', stroke: NB_COLOR, strokeWidth: 1.6, strokeDasharray: '5 3' }} />
            {SEP.path.map((p, i) => <circle key={p.lam} cx={x(decades[i])} cy={y(p.cos)} r={i === pi ? 4.5 : 2.5} style={{ fill: LR_COLOR }} />)}
          </>
        )}
      </g>
      <text x={M.left + CIW / 2} y={CH - 4} textAnchor="middle" style={axisText}>penalty λ (log scale, falling to the right)</text>
    </svg>
  );
}

function GdChart({ gi }: { gi: number }) {
  const x = d3.scaleLog().domain([SEP.gd[0].t, SEP.gd[SEP.gd.length - 1].t]).range([0, CIW]);
  const y = d3.scaleLinear().domain([0, 1]).range([CIH, 0]);
  const line = d3.line<GdPt>().x((g) => x(g.t)).y((g) => y(g.cos));
  return (
    <svg viewBox={`0 0 ${W} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="gd"
      aria-label={`After ${SEP.gd[gi].t} steps: cosine ${f3(SEP.gd[gi].cos)}`}>
      <text x={8} y={14} style={titleText}>gradient descent from w = 0: cosine to the SVM direction against steps</text>
      <g transform={`translate(${M.left},${M.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={CIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {[10, 100, 1000, 10000, 100000].map((t) => (
          <text key={`x${t}`} x={x(t)} y={CIH + 14} textAnchor="middle" style={axisText}>{t.toLocaleString('en-US')}</text>
        ))}
        <line x1={x(SEP.gd[gi].t)} x2={x(SEP.gd[gi].t)} y1={0} y2={CIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 0.8 }} />
        <path d={line(SEP.gd) ?? ''} data-series="gd-cos" style={{ fill: 'none', stroke: NBW_COLOR, strokeWidth: 2 }} />
        {SEP.gd.map((g, i) => <circle key={g.t} cx={x(g.t)} cy={y(g.cos)} r={i === gi ? 4.5 : 2.5} style={{ fill: NBW_COLOR }} />)}
      </g>
      <text x={M.left + CIW / 2} y={CH - 4} textAnchor="middle" style={axisText}>gradient-descent steps t (log scale)</text>
    </svg>
  );
}

function SeparationView() {
  const [pi, setPi] = useState(SEP.path.length - 1);
  const [gi, setGi] = useState(SEP.gd.length - 1);
  const p = SEP.path[pi];
  const g = SEP.gd[gi];
  return (
    <figure style={{ margin: 0 }} data-view="separation">
      <Caption>{`C · Separation — on ${SEP.n} tone documents with ${SEP.V.toLocaleString('en-US')} words the weights run away, and their direction settles`}</Caption>
      <Slider id="lambda" label="penalty λ" value={pi} max={SEP.path.length - 1} onChange={setPi} text={p.lam} />
      <PathChart pi={pi} kind="norm" />
      <PathChart pi={pi} kind="direction" />
      <Legend items={[
        { label: 'cosine to the SVM direction', color: LR_COLOR },
        { label: 'normalized margin ÷ γ*', color: NB_COLOR, dash: '5 3' },
      ]} />
      <Readouts>
        <Readout id="norm" label="‖ŵ_λ‖" value={f3(p.norm)} />
        <Readout id="cos" label="cosine to the SVM direction" value={f3(p.cos)} emphasis />
        <Readout id="margin" label="normalized margin / SVM margin γ*" value={`${f3(p.margin)} / ${f3(SEP.svm_margin)}`} />
        <Readout id="slope" label="ln 10 / γ* (norm gained per decade, in the limit)" value={f3(SEP.slope)} />
      </Readouts>
      <Slider id="gd-t" label="gradient-descent steps t" value={gi} max={SEP.gd.length - 1} onChange={setGi} text={g.t.toLocaleString('en-US')} />
      <GdChart gi={gi} />
      <Readouts>
        <Readout id="gd-cos" label="cosine to the SVM direction" value={f3(g.cos)} emphasis />
        <Readout id="gd-norm" label="‖w(t)‖" value={f3(g.norm)} />
        <Readout id="gd-loss" label="training loss" value={g.loss.toFixed(5)} />
      </Readouts>
      <Note>
        No finite weight vector minimizes the logistic loss on separable data (Theorem 4(a)). As λ falls the norm grows
        by about ln 10/γ* per decade while the direction converges to the hard-margin SVM (Theorem 4(c)); gradient descent
        with no penalty follows the same direction, at the logarithmic pace Soudry and colleagues prove.
      </Note>
    </figure>
  );
}

// ----- D. Finance tone -----

const METHODS: { key: 'nb' | 'lr' | 'nbw'; label: string; color: string }[] = [
  { key: 'nb', label: 'naive Bayes', color: NB_COLOR },
  { key: 'lr', label: 'logistic regression', color: LR_COLOR },
  { key: 'nbw', label: 'NB-weighted LR', color: NBW_COLOR },
];
const TONE = ['negative', 'neutral', 'positive'];

function Confusion({ m }: { m: number[][] }) {
  const max = d3.max(m.flat())!;
  const cell = 46;
  return (
    <svg viewBox={`0 0 ${cell * 3 + 92} ${cell * 3 + 40}`} style={{ width: '100%', maxWidth: '260px', height: 'auto' }} role="img" data-panel="confusion"
      aria-label={`Confusion matrix, rows true tone, columns predicted: ${m.map((r) => r.join(' ')).join('; ')}`}>
      <text x={92 + (cell * 3) / 2} y={10} textAnchor="middle" style={axisText}>predicted</text>
      {TONE.map((t, j) => (
        <text key={`c${t}`} x={92 + j * cell + cell / 2} y={24} textAnchor="middle" style={axisText}>{t.slice(0, 3)}</text>
      ))}
      {m.map((row, i) => (
        <g key={`r${i}`}>
          <text x={86} y={30 + i * cell + cell / 2 + 3} textAnchor="end" style={axisText}>{`true ${TONE[i].slice(0, 3)}`}</text>
          {row.map((v, j) => (
            <g key={`${i}${j}`}>
              <rect x={92 + j * cell} y={30 + i * cell} width={cell - 2} height={cell - 2}
                style={{ fill: i === j ? 'var(--color-accent)' : 'var(--color-badge-red-text)', fillOpacity: 0.12 + 0.75 * (v / max) }} />
              <text x={92 + j * cell + cell / 2 - 1} y={30 + i * cell + cell / 2 + 3} textAnchor="middle"
                style={{ fontFamily: mono, fontSize: '11px', fill: 'var(--color-text)' }} data-cell={`${i}-${j}`}>{v}</text>
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}

const NH = 200, NIW = W - M.left - M.right, NIH = NH - M.top - M.bottom;

function NoiseChart() {
  const lv = FIN.noise.levels;
  const x = d3.scaleLinear().domain([0, lv[lv.length - 1]]).range([0, NIW]);
  const y = d3.scaleLinear().domain([0.45, 0.8]).range([NIH, 0]);
  return (
    <svg viewBox={`0 0 ${W} ${NH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="noise"
      aria-label={METHODS.map((m) => `${m.label}: ${FIN.noise[m.key].map(f3).join(', ')}`).join('; ')}>
      <text x={8} y={14} style={titleText}>unigram test accuracy against the share of training labels flipped</text>
      <g transform={`translate(${M.left},${M.top})`}>
        {[0.5, 0.6, 0.7, 0.8].map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={NIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(1)}</text>
          </g>
        ))}
        {lv.map((l) => <text key={`x${l}`} x={x(l)} y={NIH + 14} textAnchor="middle" style={axisText}>{l}</text>)}
        {METHODS.map((m) => (
          <g key={m.key}>
            <path d={d3.line<number>().x((_, i) => x(lv[i])).y((v) => y(v))(FIN.noise[m.key]) ?? ''} data-series={`noise-${m.key}`}
              style={{ fill: 'none', stroke: m.color, strokeWidth: 2 }} />
            {FIN.noise[m.key].map((v, i) => <circle key={i} cx={x(lv[i])} cy={y(v)} r={3} style={{ fill: m.color }} />)}
          </g>
        ))}
      </g>
      <text x={M.left + NIW / 2} y={NH - 4} textAnchor="middle" style={axisText}>label noise</text>
    </svg>
  );
}

function FinanceView() {
  const [method, setMethod] = useState<'nb' | 'lr' | 'nbw'>('lr');
  const [feat, setFeat] = useState<'uni' | 'bi'>('bi');
  const sel = FIN.methods[`${method}-${feat}`];
  const cellStyle: CSSProperties = { fontFamily: sans, fontSize: '0.8rem', padding: '0.3rem 0.5rem', borderBottom: '1px solid var(--color-border)', textAlign: 'right' };
  return (
    <figure style={{ margin: 0 }} data-view="finance">
      <Caption>{`D · Finance tone — three classifiers, two feature sets, ${FIN.n_test.toLocaleString('en-US')} test filings, against a full-text ceiling of ${f3(FIN.ceiling)}`}</Caption>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.6rem' }}>
        {METHODS.map((m) => (['uni', 'bi'] as const).map((fe) => {
          const c = FIN.methods[`${m.key}-${fe}`];
          const active = method === m.key && feat === fe;
          return (
            <button key={`${m.key}-${fe}`} type="button" aria-pressed={active} data-cell-method={`${m.key}-${fe}`}
              onClick={() => { setMethod(m.key); setFeat(fe); }}
              style={{
                fontFamily: sans, fontSize: '0.75rem', padding: '0.35rem 0.6rem', borderRadius: '0.5rem', cursor: 'pointer', minWidth: 0,
                border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
                background: active ? 'var(--color-surface)' : 'transparent', color: 'var(--color-text)', textAlign: 'left',
              }}>
              <div style={{ color: 'var(--color-text-secondary)' }}>{`${m.label}, ${fe === 'uni' ? 'unigrams' : 'bigrams'}`}</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: m.color }}>{f3(c.acc)}</div>
            </button>
          );
        }))}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 220px', minWidth: 0 }}><Confusion m={sel.confusion} /></div>
        <div style={{ flex: '1 1 220px', minWidth: 0 }}>
          <Readouts>
            <Readout id="fin-acc" label="accuracy" value={f3(sel.acc)} emphasis />
            <Readout id="fin-ceiling" label="full-text Bayes accuracy" value={f3(FIN.ceiling)} />
            <Readout id="fin-conf" label="mean confidence" value={f3(sel.conf)} />
            <Readout id="fin-ece" label="ECE, 15 bins" value={f3(sel.ece)} />
            <Readout id="fin-hp" label={method === 'nb' ? 'chosen α' : 'chosen λ'} value={sel.hp} />
            <Readout id="fin-v" label="features" value={sel.V.toLocaleString('en-US')} />
          </Readouts>
        </div>
      </div>
      <NoiseChart />
      <Legend items={METHODS.map((m) => ({ label: m.label, color: m.color }))} />
      <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
        <table data-panel="xor" style={{ borderCollapse: 'collapse', width: '100%', maxWidth: '420px' }}>
          <thead>
            <tr>
              <th style={{ ...cellStyle, textAlign: 'left' }}>feature</th>
              <th style={cellStyle}>unigram fit</th>
              <th style={cellStyle}>bigram fit</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(FIN.xor).map(([f, v]) => (
              <tr key={f} data-feature={f}>
                <td style={{ ...cellStyle, textAlign: 'left', fontFamily: mono }}>{f}</td>
                <td style={cellStyle}>{v.uni === null ? '—' : f3(v.uni)}</td>
                <td style={cellStyle}>{v.bi === null ? '—' : f3(v.bi)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Note>
        Logistic regression weights, positive minus negative tone. "rose" and "fell" get almost nothing on their own,
        because revenue rising is good news and costs rising bad (Proposition 4); the bigrams carry the sign. Naive Bayes is
        overconfident on both feature sets, and logistic regression on bigrams, with more features than documents, is too.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'crossover', label: 'A · Crossover' },
  { key: 'funnel', label: 'B · Funnel' },
  { key: 'separation', label: 'C · Separation' },
  { key: 'finance', label: 'D · Finance tone' },
] as const;

export default function LinearTextClassifiersLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('crossover');
  return (
    <div data-lab="linear-text-classifiers" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'crossover' && <CrossoverView />}
      {view === 'funnel' && <FunnelView />}
      {view === 'separation' && <SeparationView />}
      {view === 'finance' && <FinanceView />}
    </div>
  );
}

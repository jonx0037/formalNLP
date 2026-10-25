import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/transformer-block-residual-stream/transformer_block_residual_stream.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the views make. Change the .py, regenerate, never the reverse.
import payload from './data/residual-stream-lab.json';

/**
 * Residual Stream Laboratory — two figures for the transformer-block-residual-stream topic.
 *   A. "Smoothing without a self": ten tokens in the plane through twelve layers of attention, pure or with the skip,
 *      at five inverse temperatures β, beside their diameters and Theorem 1's bound (Thm 1, Prop 2); LayerNorm at
 *      d = 3 as a projection onto the plane x + y + z = 0 and then onto its circle of radius √3 (Prop 3); and the
 *      pre-LN and post-LN gradient profiles at initialization (Thm 2).
 *   B. "Two heads that compose": the trained 2-layer attention-only model on a test sequence of period 11 — the
 *      previous-token and induction heads' attention, the correct token's logit split into its nine paths (Thm 3),
 *      the logit lens, the ablations and composition scores, and the finance case's held-out filing.
 */

type Run = { X: number[][]; diam: number[]; spread: number[]; tau: number[]; bound: number[]; growth: number };
type Profile = { w2: number[]; norm: number[] };

const COLLAPSE = payload.collapse as unknown as { betas: number[]; depth: number; runs: Record<string, Run> };
const LN = payload.layernorm as unknown as { plane: number[]; along: number[]; ln_plane: number[]; ln_along: number[] };
const GRADS = payload.gradients as unknown as Record<'pre' | 'post', Record<string, Profile>>;
const IND = payload.induction as unknown as {
  tokens: number[]; targets: number[]; mask: boolean[]; period: number; prev_head: number[]; induction_head: number[];
  attention: { prev: number[][]; induction: number[][] }; paths: Record<string, number[]>; lens: number[][];
  lens_mean: number[]; metrics: Record<'two' | 'one' | 'cut', { loss: number; acc: number }>;
  composition: Record<'k' | 'q' | 'v', number[]>;
};
const NAMES = payload.names as unknown as {
  readers: string[]; accuracy: Record<'seen' | 'unseen', Record<string, number>>; counts: Record<string, number>;
  words: string[]; queries: (number | boolean)[][];
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;

const N_CTX = IND.tokens.length;
const DEPTH = COLLAPSE.depth;
const LN_SCALES = [0.25, 0.5, 1, 1.5, 2, 3, 4];
const GRAD_DEPTHS = Object.keys(GRADS.pre);
const READER_LABEL: Record<string, string> = {
  bigram: 'bigram table', 'filings-direct': 'filing-trained, direct path', 'filings-2': 'filing-trained model',
  'induction-1': 'induction, 1 layer', 'induction-2': 'induction, 2 layers',
};
const PATH_GROUPS = [
  { key: 'direct', label: 'direct path', color: 'var(--color-text-secondary)', test: (k: string) => k === 'direct' },
  { key: 'l0', label: 'layer-0 heads', color: 'var(--color-badge-blue-text)', test: (k: string) => /^0\.\d$/.test(k) },
  { key: 'l1', label: 'layer-1 heads on the direct path', color: 'var(--color-accent-secondary)', test: (k: string) => /^1\.\d$/.test(k) },
  { key: 'comp', label: 'composed: layer 0 → layer 1', color: 'var(--color-accent)', test: (k: string) => k.includes('+') },
];
const pathLabel = (k: string) => (k === 'direct' ? 'direct' : k.split('+').map((h) => `(${h.replace('.', ', ')})`).join(' → '));

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
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)' }}>
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

function Legend({ items }: { items: { label: string; color: string; dash?: string; swatch?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true">
            {s.swatch
              ? <rect x="5" y="0" width="12" height="8" style={{ fill: s.color }} />
              : <line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Slider({ id, label, value, min = 0, max, onChange, text }: {
  id: string; label: string; value: number; min?: number; max: number; onChange: (v: number) => void; text: string;
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
      <label htmlFor={`residual-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`residual-lab-${id}`} type="range" min={min} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

/** A small table of rows of cells, as wrapping flex rows so it never overflows a phone screen. */
function Grid({ head, rows, id }: { head: string[]; rows: string[][]; id: string }) {
  const cell: CSSProperties = { flex: '1 1 0', minWidth: 0, padding: '0.2rem 0.3rem', overflowWrap: 'anywhere' };
  return (
    <div data-panel={id} style={{ fontFamily: sans, fontSize: '0.78rem', marginTop: '0.5rem' }}>
      <div style={{ display: 'flex', color: 'var(--color-text-secondary)', borderBottom: '1px solid var(--color-border)' }}>
        {head.map((h, i) => <div key={i} style={{ ...cell, flex: i === 0 ? '2 1 0' : '1 1 0' }}>{h}</div>)}
      </div>
      {rows.map((r, j) => (
        <div key={j} data-row={j} style={{ display: 'flex', borderBottom: '1px solid var(--color-border)' }}>
          {r.map((v, i) => <div key={i} style={{ ...cell, flex: i === 0 ? '2 1 0' : '1 1 0', fontFamily: i === 0 ? sans : mono }}>{v}</div>)}
        </div>
      ))}
    </div>
  );
}

const PW = 600, PH = 250, PM = { top: 22, right: 18, bottom: 34, left: 52 };
const IW = PW - PM.left - PM.right, IH = PH - PM.top - PM.bottom;
const FLOOR = 1e-12; // the log axis's floor: a pure stack's diameter reaches 0

const pairs = (flat: number[]) => d3.range(flat.length / 2).map((i) => [flat[2 * i], flat[2 * i + 1]] as [number, number]);
const sci = (v: number) => (v === 0 ? '0' : v < 1e-3 || v >= 1e4 ? v.toExponential(2) : v.toPrecision(4));

export default function ResidualStreamLab() {
  const [viewA, setViewA] = useState<'collapse' | 'layernorm' | 'gradients'>('collapse');
  const [betaIdx, setBetaIdx] = useState(2);
  const [skip, setSkip] = useState(false);
  const [depth, setDepth] = useState(DEPTH);
  const [lnStep, setLnStep] = useState<'raw' | 'centered' | 'normalized'>('raw');
  const [lnScale, setLnScale] = useState(2);
  const [gradDepth, setGradDepth] = useState(GRAD_DEPTHS[GRAD_DEPTHS.length - 1]);
  const [viewB, setViewB] = useState<'attention' | 'paths' | 'ablation' | 'filing'>('attention');
  const [head, setHead] = useState<'prev' | 'induction'>('induction');
  const [pos, setPos] = useState(() => IND.mask.lastIndexOf(true));
  const [reader, setReader] = useState('induction-2');

  // ----- Figure A: the cloud -----
  const beta = COLLAPSE.betas[betaIdx];
  const run = COLLAPSE.runs[`${beta}|${skip ? 'skip' : 'pure'}`];
  const other = COLLAPSE.runs[`${beta}|${skip ? 'pure' : 'skip'}`];
  const pts = pairs(run.X[depth]);
  const centroid: [number, number] = [d3.mean(pts, (p) => p[0])!, d3.mean(pts, (p) => p[1])!];
  const first = pairs(run.X[0]);
  const c0: [number, number] = [d3.mean(first, (p) => p[0])!, d3.mean(first, (p) => p[1])!];
  const reach0 = d3.max(first, (p) => Math.hypot(p[0] - c0[0], p[1] - c0[1]))!;
  const reachL = d3.max(pts, (p) => Math.hypot(p[0] - centroid[0], p[1] - centroid[1]))!;
  // the pure stack is drawn at the input's scale, so the cloud visibly shrinks; the skip stack grows ×4000, so each
  // layer is redrawn to fit and the readout gives the scale
  const half = skip ? Math.max(reachL, 1e-300) * 1.15 : reach0 * 1.15;
  const CS = 260;
  const cxs = d3.scaleLinear().domain([centroid[0] - half, centroid[0] + half]).range([0, CS]);
  const cys = d3.scaleLinear().domain([centroid[1] - half, centroid[1] + half]).range([CS, 0]);
  const dx = d3.scaleLinear().domain([0, DEPTH]).range([0, IW]);
  const allDiam = [...run.diam, ...other.diam, ...COLLAPSE.runs[`${beta}|pure`].bound].map((v) => Math.max(v, FLOOR));
  const dy = d3.scaleLog().domain([FLOOR, d3.max(allDiam)! * 2]).range([IH, 0]);
  const dLine = (arr: number[]) => d3.line<number>().x((_, i) => dx(i)).y((v) => dy(Math.max(v, FLOOR)))(arr)!;
  const pureRun = COLLAPSE.runs[`${beta}|pure`], skipRun = COLLAPSE.runs[`${beta}|skip`];

  // ----- Figure A: LayerNorm -----
  const lnRaw = pairs(LN.plane), lnOut = pairs(LN.ln_plane);
  const lnHalf = Math.max(d3.max(lnRaw, (p) => Math.hypot(...p))! * 1.6, Math.sqrt(3) * 1.3);
  const lxs = d3.scaleLinear().domain([-lnHalf, lnHalf]).range([0, CS]);
  const lys = d3.scaleLinear().domain([-lnHalf, lnHalf]).range([CS, 0]);
  const c = LN_SCALES[lnScale];
  const lnPts = lnStep === 'normalized' ? lnOut : lnRaw.map(([a, b]) => [c * a, c * b] as [number, number]);

  // ----- Figure A: gradients -----
  const gp = GRADS.pre[gradDepth], gq = GRADS.post[gradDepth];
  const gx = d3.scaleLinear().domain([1, gp.w2.length]).range([0, IW]);
  const gy = d3.scaleLinear().domain([0, d3.max([...gp.w2, ...gq.w2])! * 1.1]).range([IH, 0]).nice();
  const gLine = (arr: number[]) => d3.line<number>().x((_, i) => gx(i + 1)).y((v) => gy(v))(arr)!;

  // ----- Figure B -----
  const A = IND.attention[head];
  const HS = 300, cellW = HS / N_CTX;
  const t = Math.min(Math.max(pos, 0), N_CTX - 1);
  const rowWeights = A.filter((e) => e[0] === t).sort((a, b) => b[2] - a[2]);
  const indTarget = t - IND.period + 1;
  const pathKeys = Object.keys(IND.paths);
  const pathVals = pathKeys.map((k) => IND.paths[k][t]);
  const total = d3.sum(pathVals);
  const bx = d3.scaleBand<string>().domain(pathKeys).range([0, IW]).padding(0.2);
  const pExt = d3.extent([0, ...pathVals]) as [number, number];
  const by = d3.scaleLinear().domain(pExt).range([IH, 0]).nice();
  const colorOf = (k: string) => PATH_GROUPS.find((g) => g.test(k))!.color;
  const queries = NAMES.queries.map((q) => ({ i: q[0] as number, hits: q.slice(1) as boolean[] }));
  const rIdx = NAMES.readers.indexOf(reader);
  const hitAt = new Map(queries.map((q) => [q.i, q.hits[rIdx]]));

  return (
    <div data-lab="residual" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Smoothing without a self — collapse, the skip, LayerNorm, and the identity path
        </figcaption>
        <Row>
          <Pill active={viewA === 'collapse'} onClick={() => setViewA('collapse')} data={{ 'data-view': 'collapse' }}>the collapse</Pill>
          <Pill active={viewA === 'layernorm'} onClick={() => setViewA('layernorm')} data={{ 'data-view': 'layernorm' }}>LayerNorm at d = 3</Pill>
          <Pill active={viewA === 'gradients'} onClick={() => setViewA('gradients')} data={{ 'data-view': 'gradients' }}>pre-LN vs post-LN</Pill>
        </Row>

        {viewA === 'collapse' && (
          <>
            <Row>
              {COLLAPSE.betas.map((b, i) => (
                <Pill key={b} active={i === betaIdx} onClick={() => setBetaIdx(i)} data={{ 'data-beta': String(b) }}>{`β = ${b}`}</Pill>
              ))}
              <Pill active={!skip} onClick={() => setSkip(false)} data={{ 'data-stack': 'pure' }}>pure attention</Pill>
              <Pill active={skip} onClick={() => setSkip(true)} data={{ 'data-stack': 'skip' }}>with the skip</Pill>
            </Row>
            <Slider id="depth" label="layer" value={depth} max={DEPTH} onChange={setDepth} text={`l = ${depth}`} />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
              <svg viewBox={`0 0 ${CS} ${CS}`} style={{ flex: '1 1 14rem', minWidth: 0, maxWidth: '20rem', height: 'auto' }} role="img" data-panel="cloud"
                aria-label={`The ten tokens after ${depth} layers of ${skip ? 'attention with the skip' : 'pure attention'} at inverse temperature ${beta}`}>
                <rect x={0} y={0} width={CS} height={CS} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
                <line x1={cxs(centroid[0])} x2={cxs(centroid[0])} y1={0} y2={CS} style={{ stroke: 'var(--color-border)', strokeDasharray: '2 3' }} />
                <line y1={cys(centroid[1])} y2={cys(centroid[1])} x1={0} x2={CS} style={{ stroke: 'var(--color-border)', strokeDasharray: '2 3' }} />
                {pts.map(([a, b], i) => (
                  <circle key={i} data-token={i} cx={cxs(a)} cy={cys(b)} r={4.5}
                    style={{ fill: 'var(--color-accent)', fillOpacity: 0.75, stroke: 'var(--color-bg)', strokeWidth: 1 }} />
                ))}
                <text x={6} y={14} style={titleText}>{skip ? 'drawn to fit this layer' : 'drawn at the input’s scale'}</text>
              </svg>
              <svg viewBox={`0 0 ${PW} ${PH}`} style={{ flex: '2 1 18rem', minWidth: 0, height: 'auto' }} role="img" data-panel="diameter"
                aria-label="The cloud's diameter by layer on a log scale, pure and with the skip, with Theorem 1's bound for the pure stack">
                <g transform={`translate(${PM.left},${PM.top})`}>
                  {[1e-12, 1e-8, 1e-4, 1, 1e4].filter((v) => v <= dy.domain()[1]).map((v) => (
                    <g key={v}>
                      <line x1={0} x2={IW} y1={dy(v)} y2={dy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                      <text x={-6} y={dy(v) + 3} textAnchor="end" style={axisText}>{v === 1e-12 ? '≤10⁻¹²' : v >= 1 ? String(v) : `10${v === 1e-8 ? '⁻⁸' : '⁻⁴'}`}</text>
                    </g>
                  ))}
                  {d3.range(0, DEPTH + 1, 2).map((v) => <text key={v} x={dx(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>)}
                  <path d={dLine(pureRun.bound)} data-series="bound" style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.4, strokeDasharray: '5 3' }} />
                  <path d={dLine(pureRun.diam)} data-series="pure" style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: skip ? 1.6 : 2.8 }} />
                  <path d={dLine(skipRun.diam)} data-series="skip" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: skip ? 2.8 : 1.6 }} />
                  <line x1={dx(depth)} x2={dx(depth)} y1={0} y2={IH} style={{ stroke: 'var(--color-text)', strokeWidth: 1, strokeDasharray: '3 3' }} />
                </g>
                <text x={8} y={13} style={titleText}>{`diameter of the cloud, β = ${beta} (log scale)`}</text>
                <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>layer l</text>
              </svg>
            </div>
            <Legend items={[
              { label: 'pure attention', color: 'var(--color-badge-blue-text)' },
              { label: 'with the skip', color: 'var(--color-accent)' },
              { label: 'Theorem 1 bound, ∏τ · diam(X₀)', color: 'var(--color-text-secondary)', dash: '5 3' },
            ]} />
            <Readouts>
              <Readout id="diam" label="diameter" value={sci(run.diam[depth])} emphasis />
              <Readout id="tau" label={depth > 0 ? `τ of layer ${depth}` : 'τ of layer 1'} value={run.tau[Math.max(depth - 1, 0)].toFixed(3)} />
              <Readout id="spread" label="relative spread ‖X − 1x̄ᵀ‖/‖X‖" value={run.spread[depth].toFixed(4)} />
              <Readout id="growth" label="diam(X₁₂) / diam(X₀)" value={run.growth >= 1e-3 ? run.growth.toFixed(run.growth < 100 ? 2 : 0) : sci(run.growth)} />
              {skip && <Readout id="scale" label="half-width of the view" value={sci(half)} />}
            </Readouts>
            <Note>
              A wide kernel (small β) keeps τ below 1, and the pure stack shrinks the cloud to a point, under Theorem 1’s bound. A
              narrow kernel (β = 4) makes each token attend to itself: τ stays near 1, the bound says nothing, and the cloud
              freezes into clusters. With the skip the diameter never shrinks (Proposition 2), but at β = 0.25 the relative
              spread still falls: the skip keeps the differences, not their share.
            </Note>
          </>
        )}

        {viewA === 'layernorm' && (
          <>
            <Row>
              <Pill active={lnStep === 'raw'} onClick={() => setLnStep('raw')} data={{ 'data-ln': 'raw' }}>the points</Pill>
              <Pill active={lnStep === 'centered'} onClick={() => setLnStep('centered')} data={{ 'data-ln': 'centered' }}>1. center: drop onto the plane</Pill>
              <Pill active={lnStep === 'normalized'} onClick={() => setLnStep('normalized')} data={{ 'data-ln': 'normalized' }}>2. normalize: out to the circle</Pill>
            </Row>
            <Slider id="ln-scale" label="scale the points by c" value={lnScale} max={LN_SCALES.length - 1} onChange={setLnScale} text={`c = ${c}`} />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
              <svg viewBox={`0 0 ${CS} ${CS}`} style={{ flex: '1 1 14rem', minWidth: 0, maxWidth: '22rem', height: 'auto' }} role="img" data-panel="layernorm"
                aria-label="Twenty-four points of R³ drawn in the plane x + y + z = 0, with their offset along the all-ones direction as a vertical tick, and the circle of radius √3">
                <rect x={0} y={0} width={CS} height={CS} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
                <circle cx={lxs(0)} cy={lys(0)} r={lxs(Math.sqrt(3)) - lxs(0)} data-series="circle"
                  style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 1.5, strokeDasharray: '4 3' }} />
                {lnPts.map(([a, b], i) => (
                  <g key={i} data-point={i}>
                    {lnStep === 'raw' && (
                      <line x1={lxs(a)} x2={lxs(a)} y1={lys(b)} y2={lys(b) - 6 * c * LN.along[i]}
                        style={{ stroke: 'var(--color-badge-blue-text)', strokeWidth: 1.2 }} />
                    )}
                    <circle cx={lxs(a)} cy={lys(b)} r={3.5} style={{ fill: 'var(--color-accent)', fillOpacity: 0.8 }} />
                  </g>
                ))}
                <text x={6} y={14} style={titleText}>the plane x + y + z = 0</text>
              </svg>
              <div style={{ flex: '1 1 12rem', minWidth: 0 }}>
                <Readouts>
                  <Readout id="ln-radius" label="radius of the circle, √d" value={Math.sqrt(3).toFixed(3)} emphasis />
                  <Readout id="ln-step" label="step" value={lnStep === 'raw' ? 'x' : lnStep === 'centered' ? 'Px' : '√d · Px/‖Px‖'} />
                </Readouts>
                <Note>
                  Each point of R³ is drawn at its shadow on the plane orthogonal to 1 = (1, 1, 1); the blue tick is its offset
                  along 1. Centering removes the tick, and normalizing pushes every point out along its ray to the circle
                  (Proposition 3). Move c in the last step: the normalized points do not move, because LN(cx) = LN(x).
                </Note>
              </div>
            </div>
          </>
        )}

        {viewA === 'gradients' && (
          <>
            <Row>
              {GRAD_DEPTHS.map((L) => (
                <Pill key={L} active={L === gradDepth} onClick={() => setGradDepth(L)} data={{ 'data-grad-depth': L }}>{`L = ${L}`}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="gradients"
              aria-label={`The gradient norm of each block's MLP output map at initialization, pre-LN and post-LN, for a stack of ${gradDepth} blocks`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {gy.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={gy(v)} y2={gy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={gy(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {gx.ticks(Math.min(gp.w2.length, 8)).map((v) => <text key={v} x={gx(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>)}
                <path d={gLine(gp.w2)} data-series="pre" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2.2 }} />
                <path d={gLine(gq.w2)} data-series="post" style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: 2.2, strokeDasharray: '6 3' }} />
              </g>
              <text x={8} y={13} style={titleText}>‖∂ℒ/∂W₂‖ of each block at initialization (RMS over 8 draws)</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>block</text>
            </svg>
            <Legend items={[{ label: 'pre-LN', color: 'var(--color-accent)' }, { label: 'post-LN', color: 'var(--color-badge-blue-text)', dash: '6 3' }]} />
            <Readouts>
              <Readout id="grad-pre" label="pre-LN, last block" value={gp.w2[gp.w2.length - 1].toFixed(3)} emphasis />
              <Readout id="grad-post" label="post-LN, last block" value={gq.w2[gq.w2.length - 1].toFixed(3)} />
              <Readout id="norm-pre" label="pre-LN ‖x_L‖" value={gp.norm[gp.norm.length - 1].toFixed(3)} />
              <Readout id="norm-post" label="post-LN ‖x_L‖" value={gq.norm[gq.norm.length - 1].toFixed(3)} />
            </Readouts>
            <Note>
              Switch L: the post-LN last block’s gradient stays near 22, and the pre-LN one falls like 1/√L, because the pre-LN
              stream grows with depth and LayerNorm’s Jacobian shrinks as 1/‖Px‖ (Theorem 2 (ii)), the scaling Xiong and
              colleagues predict. The profile’s shape depends on the initialization; the scaling is the claim.
            </Note>
          </>
        )}
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Two heads that compose — the trained two-layer model on a sequence of period {IND.period}
        </figcaption>
        <Row>
          <Pill active={viewB === 'attention'} onClick={() => setViewB('attention')} data={{ 'data-view': 'attention' }}>the heads</Pill>
          <Pill active={viewB === 'paths'} onClick={() => setViewB('paths')} data={{ 'data-view': 'paths' }}>the paths</Pill>
          <Pill active={viewB === 'ablation'} onClick={() => setViewB('ablation')} data={{ 'data-view': 'ablation' }}>ablations and the lens</Pill>
          <Pill active={viewB === 'filing'} onClick={() => setViewB('filing')} data={{ 'data-view': 'filing' }}>a filing</Pill>
        </Row>

        {(viewB === 'attention' || viewB === 'paths') && (
          <Slider id="position" label="position i" value={t} max={N_CTX - 1} onChange={setPos}
            text={`i = ${t}${IND.mask[t] ? '' : ' (not yet repeated)'}`} />
        )}

        {viewB === 'attention' && (
          <>
            <Row>
              <Pill active={head === 'prev'} onClick={() => setHead('prev')} data={{ 'data-head': 'prev' }}>{`previous-token head (0, ${IND.prev_head[1]})`}</Pill>
              <Pill active={head === 'induction'} onClick={() => setHead('induction')} data={{ 'data-head': 'induction' }}>{`induction head (1, ${IND.induction_head[1]})`}</Pill>
            </Row>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
              <svg viewBox={`-28 -14 ${HS + 34} ${HS + 30}`} style={{ flex: '1 1 16rem', minWidth: 0, maxWidth: '26rem', height: 'auto' }} role="img" data-panel="heatmap"
                aria-label={`The ${head === 'prev' ? 'previous-token' : 'induction'} head's attention: rows are query positions, columns key positions; weights below 0.02 are blank`}>
                <rect x={0} y={0} width={HS} height={HS} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
                <rect x={0} y={t * cellW} width={HS} height={cellW} style={{ fill: 'var(--color-muted-bg)' }} />
                {A.map(([i, j, w]) => (
                  <rect key={`${i}-${j}`} x={j * cellW} y={i * cellW} width={cellW} height={cellW} data-cell={`${i}-${j}`}
                    style={{ fill: 'var(--color-accent)', fillOpacity: Math.min(1, 0.15 + w) }} />
                ))}
                {head === 'induction' && IND.mask[t] && (
                  <rect x={indTarget * cellW} y={t * cellW} width={cellW} height={cellW}
                    style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.2 }} data-series="target" />
                )}
                {[0, 12, 24, 36, 47].map((v) => (
                  <g key={v}>
                    <text x={v * cellW + cellW / 2} y={-4} textAnchor="middle" style={axisText}>{v}</text>
                    <text x={-4} y={v * cellW + cellW} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                <text x={HS / 2} y={HS + 13} textAnchor="middle" style={axisText}>key position j → (rows: query i)</text>
              </svg>
              <div style={{ flex: '1 1 12rem', minWidth: 0 }}>
                <Readouts>
                  <Readout id="token" label={`token at i, then the next`} value={`${IND.tokens[t]} → ${IND.targets[t]}`} />
                  <Readout id="top" label="largest weight in row i" value={rowWeights.length ? `${rowWeights[0][2].toFixed(2)} on j = ${rowWeights[0][1]}` : '< 0.02'} emphasis />
                  {head === 'induction' && <Readout id="target" label="induction target i − p + 1" value={IND.mask[t] ? String(indTarget) : '—'} />}
                  {head === 'prev' && <Readout id="previous" label="previous position" value={t > 0 ? String(t - 1) : '—'} />}
                </Readouts>
                <Note>
                  The previous-token head’s weight sits just left of the diagonal, at j = i − 1. The induction head attends to
                  the earlier positions whose predecessor is the current token: i − p + 1 (outlined), and one or more periods
                  further back on a long repetition. Each holds a copy of the next token. Its keys can find it only because the previous-token head wrote each predecessor into
                  the stream (Proposition 5).
                </Note>
              </div>
            </div>
          </>
        )}

        {viewB === 'paths' && (
          <>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="paths"
              aria-label="The correct next token's centered logit at position i, split into the nine paths of Theorem 3">
              <g transform={`translate(${PM.left},${PM.top})`}>
                {by.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={by(v)} y2={by(v)} style={{ stroke: 'var(--color-border)', strokeWidth: v === 0 ? 1 : 0.5 }} />
                    <text x={-6} y={by(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {pathKeys.map((k, i) => (
                  <g key={k} data-path={k}>
                    <rect x={bx(k)} width={bx.bandwidth()} y={Math.min(by(0), by(pathVals[i]))} height={Math.abs(by(pathVals[i]) - by(0))}
                      style={{ fill: colorOf(k) }} />
                    <text x={bx(k)! + bx.bandwidth() / 2} y={IH + 14} textAnchor="middle" style={{ ...axisText, fontSize: '8.5px' }}>{pathLabel(k)}</text>
                  </g>
                ))}
              </g>
              <text x={8} y={13} style={titleText}>{`the correct token’s centered logit at i = ${t}, by path`}</text>
            </svg>
            <Legend items={PATH_GROUPS.map((g) => ({ label: g.label, color: g.color, swatch: true }))} />
            <Readouts>
              <Readout id="path-total" label="sum of the nine paths" value={total.toFixed(2)} emphasis />
              {PATH_GROUPS.map((g) => (
                <Readout key={g.key} id={`path-${g.key}`} label={g.label}
                  value={d3.sum(pathKeys.filter(g.test).map((k) => IND.paths[k][t])).toFixed(2)} />
              ))}
              <Readout id="lens-here" label="P(next token) after layers 0, 1, 2" value={IND.lens.map((l) => l[t].toFixed(3)).join(', ')} />
            </Readouts>
            <Note>
              The logit is centered (minus its mean over the vocabulary), so each bar is that path’s vote for the correct token
              over the average one. On the repeated half the layer-1 paths carry it; before the repetition starts, nothing
              does, and the probability stays near chance, 1/256.
            </Note>
          </>
        )}

        {viewB === 'ablation' && (
          <>
            <Grid id="metrics" head={['model, on 1000 test sequences', 'loss (nats)', 'accuracy']} rows={[
              ['two layers', IND.metrics.two.loss.toFixed(3), IND.metrics.two.acc.toFixed(3)],
              ['one layer', IND.metrics.one.loss.toFixed(3), IND.metrics.one.acc.toFixed(3)],
              ['two layers, layer 1’s keys from x₀', IND.metrics.cut.loss.toFixed(3), IND.metrics.cut.acc.toFixed(3)],
            ]} />
            <Grid id="composition" head={['layer-0 head → layer-1 head', 'K', 'Q', 'V']} rows={
              [[0, 0], [0, 1], [1, 0], [1, 1]].map(([h, h2], i) => [
                `(0, ${h}) → (1, ${h2})`, IND.composition.k[i].toFixed(3), IND.composition.q[i].toFixed(3), IND.composition.v[i].toFixed(3),
              ])} />
            <Grid id="lens" head={['logit lens, read after', 'P(correct token)']} rows={
              IND.lens_mean.map((v, l) => [l === 0 ? 'the embedding' : `layer ${l}`, v.toFixed(3)])} />
            <Note>
              Cutting the K-composition, so layer 1’s keys cannot read what layer 0 wrote, takes the solved model below the
              one-layer one. K-composition dominates Q and V for every pair of heads. The lens finds the answer only after the
              second layer.
            </Note>
          </>
        )}

        {viewB === 'filing' && (
          <>
            <Row>
              {NAMES.readers.map((r) => (
                <Pill key={r} active={r === reader} onClick={() => setReader(r)} data={{ 'data-reader': r }}>{READER_LABEL[r] ?? r}</Pill>
              ))}
            </Row>
            <div data-panel="filing" style={{ fontFamily: sans, fontSize: '0.8rem', lineHeight: 1.7, maxHeight: '14rem', overflowY: 'auto', overflowWrap: 'anywhere', border: '1px solid var(--color-border)', borderRadius: '0.5rem', padding: '0.5rem 0.65rem' }}>
              {NAMES.words.map((w, i) => {
                const hit = hitAt.get(i - 1);
                const isAnswer = hit !== undefined;
                return (
                  <span key={i} data-word={i} data-hit={isAnswer ? String(hit) : undefined}
                    style={isAnswer ? {
                      padding: '0 0.15rem', borderRadius: '0.2rem', fontWeight: 600,
                      background: hit ? 'var(--color-badge-green-bg)' : 'var(--color-badge-red-bg)',
                      color: hit ? 'var(--color-badge-green-text)' : 'var(--color-badge-red-text)',
                    } : undefined}>
                    {w}{' '}
                  </span>
                );
              })}
            </div>
            <Readouts>
              <Readout id="filing-hits" label={`${READER_LABEL[reader]}: completed in this filing`}
                value={`${queries.filter((q) => q.hits[rIdx]).length} of ${queries.length}`} emphasis />
            </Readouts>
            <Grid id="names" head={['reader', `seen (${NAMES.counts.seen})`, `unseen (${NAMES.counts.unseen})`]} rows={
              NAMES.readers.map((r) => [READER_LABEL[r] ?? r, NAMES.accuracy.seen[r].toFixed(3), NAMES.accuracy.unseen[r].toFixed(3)])} />
            <Note>
              A held-out filing about a company that never occurs in training. Each highlighted word is a name word the reader
              had to predict from the word before it: green if it did. The bigram table and the filing-trained model know only
              names they memorized; the induction model reads the name off the filing’s earlier text.
            </Note>
          </>
        )}
      </figure>
    </div>
  );
}

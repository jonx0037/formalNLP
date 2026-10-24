import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/attention-as-kernel-smoothing/attention_as_kernel_smoothing.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/attention-kernel-lab.json';

/**
 * Attention Kernel Laboratory — two panels for the attention-as-kernel-smoothing topic.
 *   A. "Temperature is bandwidth": n keys equally spaced on a circle, scores c·cos(θ_i − θ_j) (Thm 1). The smoothed
 *      curve is Definition 2 evaluated live on the payload's sample (θ, y); the risk, its bias² and variance, and the
 *      Fourier gains ρ₁…ρ₈ are the payload's exact values on its c grid; c* and the trained layer's c are marked.
 *   B. "A lookup instead of a state": a held-out document and the three EDGAR filings, token by token, with the bits
 *      Kneser–Ney and each kernel cache spend on every token (Def 3, Def 4) and, for marked tokens, the eight earlier
 *      positions the β = 2 kernel weights most; the dev and held-out U-curve over β; and the gradient census of the
 *      long-lag task (Props 5–6) against Topic 5's recurrent cells.
 */

type Case = {
  n: number; spectrum: string; risk: number[]; bias2: number[]; var: number[]; gains: number[][];
  c_star: number; c_learned: number; risk_star: number; risk_learned: number; start: number;
  theta: number[]; f: number[]; y: number[];
};
type Attn = { t: number; pos: number[]; w: number[]; copy: number; sources: number[] };
type Doc = {
  name: string; tokens: string[]; sent: number[]; oov: number[];
  bits: Record<string, number[]>; bpc: Record<string, number>; attn: Attn[];
};

const CIRCLE = payload.circle as unknown as { c_grid: number[]; sigma: number; cases: Record<string, Case> };
const REACH = payload.reach as unknown as {
  lags: number[]; attention: number[]; elman: number[]; 'lstm-1': number[]; 'lstm-3': number[]; accuracy: Record<string, number>;
};
const CACHE = payload.cache as unknown as {
  beta: number; lambda: number; betas: (number | null)[]; held_by_beta: number[]; dev_by_beta: number[]; documents: Doc[];
};
const C_GRID = CIRCLE.c_grid;
const CASE_KEYS = Object.keys(CIRCLE.cases);
const CASE_LABEL: Record<string, string> = { 'smooth-32': 'smooth f, n = 32', 'smooth-64': 'smooth f, n = 64', 'rough-32': 'rough f, n = 32' };
const VARIANTS = ['kn', 'b0', 'bstar', 'binf', 'learned'] as const;
type Variant = (typeof VARIANTS)[number];
const VARIANT_LABEL: Record<Variant, string> = {
  kn: 'Kneser–Ney alone', b0: 'β = 0: unigram cache', bstar: `β = ${CACHE.beta}: the dev choice`, binf: 'β = ∞: exact context', learned: 'learned kernel',
};
const RECURRENT = [
  { key: 'attention', label: 'one attention layer', color: 'var(--color-accent)', dash: undefined as string | undefined, marker: 'circle' },
  { key: 'elman', label: 'Elman RNN', color: 'var(--color-text-secondary)', dash: '6 3', marker: 'square' },
  { key: 'lstm-1', label: 'LSTM, forget bias 1', color: 'var(--color-badge-blue-text)', dash: '2 3', marker: 'triangle' },
  { key: 'lstm-3', label: 'LSTM, forget bias 3', color: 'var(--color-accent-secondary)', dash: '8 3 2 3', marker: 'diamond' },
] as const;
const RISK_SERIES = [
  { key: 'risk', label: 'risk R(c)', color: 'var(--color-accent)', dash: undefined as string | undefined, width: 2.5 },
  { key: 'bias2', label: 'bias²', color: 'var(--color-accent-secondary)', dash: '6 3', width: 2 },
  { key: 'var', label: 'variance', color: 'var(--color-badge-blue-text)', dash: '2 3', width: 2 },
] as const;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** 3.76e-5 → "3.76 × 10⁻⁵" (the payload stores three significant figures). */
function sci(v: number): string {
  const [m, e] = v.toExponential(2).split('e');
  return `${m} × 10${String(Number(e)).split('').map((ch) => SUP[ch]).join('')}`;
}
const fmtC = (c: number) => (c === 0 ? '0' : c < 10 ? c.toFixed(2) : c.toFixed(1));
const tokenText = (w: string) => (w === '</s>' ? '¶' : w);

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

function Legend({ items }: { items: { label: string; color: string; dash?: string }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} /></svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Marker({ kind, x, y, color, size = 4 }: { kind: string; x: number; y: number; color: string; size?: number }) {
  if (kind === 'square') return <rect x={x - size} y={y - size} width={2 * size} height={2 * size} style={{ fill: color }} />;
  if (kind === 'triangle') return <path d={`M${x},${y - size - 1} L${x + size + 1},${y + size} L${x - size - 1},${y + size} Z`} style={{ fill: color }} />;
  if (kind === 'diamond') return <path d={`M${x},${y - size - 1} L${x + size + 1},${y} L${x},${y + size + 1} L${x - size - 1},${y} Z`} style={{ fill: color }} />;
  return <circle cx={x} cy={y} r={size} style={{ fill: color }} />;
}

/** Definition 2 on the circle: o_i = Σ_j e^{c cos(θ_i − θ_j)} y_j / Σ_j e^{c cos(θ_i − θ_j)}, shifted by c for stability. */
function smoothRow(theta: number[], y: number[], c: number, i: number): { w: number[]; o: number } {
  const w = theta.map((t) => Math.exp(c * (Math.cos(theta[i] - t) - 1)));
  const z = w.reduce((s, v) => s + v, 0);
  return { w: w.map((v) => v / z), o: w.reduce((s, v, j) => s + v * y[j], 0) / z };
}

// Panel A geometry: the circle at left, the unrolled strip at right.
const AW = 600, AH = 270, CXC = 125, CYC = 135, RAD = 100, SX0 = 262, SX1 = 590, SY0 = 22, SY1 = 238;
// Risk plot, gains bars, and Panel B plots.
const RW = 600, RH = 250, RM = { top: 22, right: 18, bottom: 34, left: 46 };
const GW = 600, GH = 110, GM = { top: 16, right: 10, bottom: 22, left: 30 };
const UW = 600, UH = 170, UM = { top: 20, right: 16, bottom: 30, left: 50 };
const LW = 600, LH = 270, LM = { top: 22, right: 18, bottom: 34, left: 46 };

export default function AttentionKernelLab() {
  const [caseKey, setCaseKey] = useState(CASE_KEYS[0]);
  const [cIdx, setCIdx] = useState(CIRCLE.cases[CASE_KEYS[0]].start);
  const [viewA, setViewA] = useState<'smoother' | 'risk'>('smoother');
  const [query, setQuery] = useState(0);
  const [viewB, setViewB] = useState<'document' | 'reach'>('document');
  const [docIdx, setDocIdx] = useState(0);
  const [variant, setVariant] = useState<Variant>('bstar');
  const [tok, setTok] = useState(CACHE.documents[0].attn[0].t);
  const [lagIdx, setLagIdx] = useState(REACH.lags.indexOf(60));

  // ----- Panel A -----
  const cs = CIRCLE.cases[caseKey];
  const c = C_GRID[cIdx];
  const q = Math.min(query, cs.n - 1);
  const rowQ = useMemo(() => smoothRow(cs.theta, cs.y, c, q), [caseKey, cIdx, q]); // eslint-disable-line react-hooks/exhaustive-deps
  const smooth = useMemo(() => cs.theta.map((_, i) => smoothRow(cs.theta, cs.y, c, i).o), [caseKey, cIdx]); // eslint-disable-line react-hooks/exhaustive-deps
  const wMax = Math.max(...rowQ.w);
  const ang = (th: number) => ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const order = d3.range(cs.n).sort((a, b) => ang(cs.theta[a]) - ang(cs.theta[b]));
  const vals = [...cs.y, ...cs.f];
  const sx = d3.scaleLinear().domain([0, 2 * Math.PI]).range([SX0, SX1]);
  const sy = d3.scaleLinear().domain([d3.min(vals)! - 0.1, d3.max(vals)! + 0.1]).range([SY1, SY0]);
  const strip = (arr: number[]) => d3.line<number>().x((i) => sx(ang(cs.theta[i]))).y((i) => sy(arr[i]))(order)!;
  const pickCase = (k: string) => { setCaseKey(k); setCIdx(CIRCLE.cases[k].start); setQuery(0); };

  const rx = d3.scaleLog().domain([C_GRID[1], C_GRID[C_GRID.length - 1]]).range([0, RW - RM.left - RM.right]);
  const rMax = d3.max(cs.risk.slice(1))! * 1.05;
  const ry = d3.scaleLinear().domain([0, rMax]).range([RH - RM.top - RM.bottom, 0]);
  const riskLine = (arr: number[]) => d3.line<number>().x((i) => rx(C_GRID[i])).y((i) => ry(arr[i]))(d3.range(1, C_GRID.length))!;
  const gains = cs.gains[cIdx];
  const gx = d3.scaleBand<number>().domain(d3.range(gains.length)).range([0, GW - GM.left - GM.right]).padding(0.25);
  const gy = d3.scaleLinear().domain([0, 1]).range([GH - GM.top - GM.bottom, 0]);

  // ----- Panel B -----
  const doc = CACHE.documents[docIdx];
  const t = Math.min(tok, doc.tokens.length - 1);
  const attn = doc.attn.find((a) => a.t === t);
  const attnW = new Map(attn ? attn.pos.map((p, i) => [p, attn.w[i]] as [number, number]) : []);
  const attnMax = attn ? Math.max(...attn.w) : 1;
  const sources = new Set(attn ? attn.sources : []);
  const kn = doc.bits.kn, vb = doc.bits[variant];
  const marked = new Set(doc.attn.map((a) => a.t));
  const pickDoc = (i: number) => { setDocIdx(i); setTok(CACHE.documents[i].attn[0].t); };

  const betaLabel = (b: number | null) => (b === null ? '∞' : b === 0.5 ? '½' : String(b));
  const ux = d3.scalePoint<number>().domain(d3.range(CACHE.betas.length)).range([0, UW - UM.left - UM.right]).padding(0.3);
  const uAll = [...CACHE.held_by_beta, ...CACHE.dev_by_beta];
  const uy = d3.scaleLinear().domain([d3.min(uAll)! - 0.002, d3.max(uAll)! + 0.002]).range([UH - UM.top - UM.bottom, 0]);
  const iStar = CACHE.betas.indexOf(CACHE.beta);
  const variantBeta: Partial<Record<Variant, number>> = { b0: 0, bstar: iStar, binf: CACHE.betas.length - 1 };

  const lx = d3.scaleLog().domain([8, 600]).range([0, LW - LM.left - LM.right]);
  const ly = d3.scaleLinear().domain([-47, -1]).range([LH - LM.top - LM.bottom, 0]);
  const guide = REACH.lags.map((L) => [lx(L), ly(Math.log10((REACH.attention[0] * (REACH.lags[0] + 1)) / (L + 1)))] as [number, number]);

  return (
    <div data-lab="attention" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Temperature is bandwidth — attention over n keys on a circle, scores c·cos(θᵢ − θⱼ)
        </figcaption>
        <Row>
          {CASE_KEYS.map((k) => (
            <Pill key={k} active={k === caseKey} onClick={() => pickCase(k)} data={{ 'data-case': k }}>{CASE_LABEL[k] ?? k}</Pill>
          ))}
        </Row>
        <Row>
          <Pill active={viewA === 'smoother'} onClick={() => setViewA('smoother')} data={{ 'data-view': 'smoother' }}>the smoother</Pill>
          <Pill active={viewA === 'risk'} onClick={() => setViewA('risk')} data={{ 'data-view': 'risk' }}>the exact risk</Pill>
        </Row>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
          <label htmlFor="attention-lab-c" style={{ minWidth: 0 }}>concentration c</label>
          <input id="attention-lab-c" type="range" min={0} max={C_GRID.length - 1} step={1} value={cIdx} data-control="c"
            onChange={(e) => setCIdx(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }}
            aria-valuetext={`c = ${fmtC(c)}`} />
          <span data-readout="c-value" style={{ fontFamily: mono, minWidth: 0 }}>{`c = ${fmtC(c)}`}</span>
        </div>
        <Row>
          <Pill active={cIdx === 0} onClick={() => setCIdx(0)} data={{ 'data-c': 'zero' }}>c = 0: the mean</Pill>
          <Pill active={cIdx === cs.start} onClick={() => setCIdx(cs.start)} data={{ 'data-c': 'star' }}>grid point nearest c*</Pill>
          <Pill active={cIdx === C_GRID.length - 1} onClick={() => setCIdx(C_GRID.length - 1)} data={{ 'data-c': 'max' }}>{`c = ${fmtC(C_GRID[C_GRID.length - 1])}: nearly a lookup`}</Pill>
        </Row>

        {viewA === 'smoother' ? (
          <>
            <svg viewBox={`0 0 ${AW} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="smoother"
              aria-label={`${cs.n} keys on a circle sized by their attention weight for the chosen query at c = ${fmtC(c)}, and the noisy values, the true function, and the smoothed output unrolled by angle`}>
              <circle cx={CXC} cy={CYC} r={RAD} style={{ fill: 'none', stroke: 'var(--color-border)', strokeWidth: 1 }} />
              {cs.theta.map((th, j) => (
                <circle key={j} cx={CXC + RAD * Math.cos(th)} cy={CYC - RAD * Math.sin(th)} r={2 + 9 * Math.sqrt(rowQ.w[j] / wMax)}
                  data-key={j} onClick={() => setQuery(j)}
                  style={{ fill: j === q ? 'var(--color-accent)' : 'var(--color-accent-secondary)', opacity: j === q ? 1 : 0.75, cursor: 'pointer',
                    stroke: j === q ? 'var(--color-text)' : 'none', strokeWidth: 1.5 }} />
              ))}
              <text x={CXC} y={CYC + 4} textAnchor="middle" style={axisText}>click a key</text>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>{`weights for query ${q}`}</text>
              <line x1={SX0} x2={SX1} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <path d={strip(cs.f)} data-series="f" style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.5, strokeDasharray: '5 3' }} />
              {cs.theta.map((th, j) => (
                <circle key={j} cx={sx(ang(th))} cy={sy(cs.y[j])} r={2.2} style={{ fill: 'var(--color-text-secondary)', opacity: 0.7 }} />
              ))}
              <path d={strip(smooth)} data-series="smoothed" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2.5 }} />
              <line x1={sx(ang(cs.theta[q]))} x2={sx(ang(cs.theta[q]))} y1={SY0} y2={SY1} style={{ stroke: 'var(--color-accent)', strokeWidth: 0.8, strokeDasharray: '2 2' }} />
              <text x={(SX0 + SX1) / 2} y={AH - 6} textAnchor="middle" style={axisText}>angle θ, from 0 to 2π</text>
            </svg>
            <Legend items={[{ label: 'true f', color: 'var(--color-text-secondary)', dash: '5 3' }, { label: 'attention output', color: 'var(--color-accent)' }]} />
            <Note>Dots: the noisy values y. The output is Definition 2 computed from these values; the readouts below are the exact risk over all functions of this spectrum.</Note>
          </>
        ) : (
          <>
            <svg viewBox={`0 0 ${RW} ${RH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="risk"
              aria-label="The exact risk of attention on the circle, its squared bias, and its variance, against the concentration c, with the optimum c* and the trained layer's c">
              <g transform={`translate(${RM.left},${RM.top})`}>
                {ry.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={RW - RM.left - RM.right} y1={ry(v)} y2={ry(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ry(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {[0.25, 1, 4, 16, 64, 256].map((v) => (
                  <text key={v} x={rx(v)} y={RH - RM.top - RM.bottom + 14} textAnchor="middle" style={axisText}>{v}</text>
                ))}
                {RISK_SERIES.map((s) => (
                  <path key={s.key} d={riskLine(cs[s.key])} data-series={s.key}
                    style={{ fill: 'none', stroke: s.color, strokeWidth: s.width, strokeDasharray: s.dash }} />
                ))}
                {cIdx > 0 && <line x1={rx(c)} x2={rx(c)} y1={0} y2={RH - RM.top - RM.bottom} style={{ stroke: 'var(--color-text)', strokeWidth: 1, strokeDasharray: '3 3' }} />}
                <Marker kind="diamond" x={rx(cs.c_star)} y={ry(cs.risk_star)} color="var(--color-text)" size={5} />
                <circle cx={rx(cs.c_learned)} cy={ry(cs.risk_learned)} r={7} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>Theorem 1, exact · diamond: c* · ring: trained layer</text>
              <text x={RW / 2} y={RH - 4} textAnchor="middle" style={axisText}>concentration c (log scale) — larger c is a lower temperature, a narrower kernel</text>
            </svg>
            <Legend items={RISK_SERIES.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
          </>
        )}

        <svg viewBox={`0 0 ${GW} ${GH}`} style={{ width: '100%', height: 'auto', marginTop: '0.4rem' }} role="img" data-panel="gains"
          aria-label={`The Fourier gains rho 1 through 8 at c = ${fmtC(c)}`}>
          <g transform={`translate(${GM.left},${GM.top})`}>
            {gains.map((g, k) => (
              <g key={k}>
                <rect x={gx(k)} y={gy(g)} width={gx.bandwidth()} height={gy(0) - gy(g)} style={{ fill: 'var(--color-accent-secondary)', opacity: 0.85 }} />
                <text x={gx(k)! + gx.bandwidth() / 2} y={GH - GM.top - GM.bottom + 13} textAnchor="middle" style={axisText}>{`ρ${'₁₂₃₄₅₆₇₈'[k]}`}</text>
              </g>
            ))}
            <line x1={0} x2={GW - GM.left - GM.right} y1={gy(1)} y2={gy(1)} style={{ stroke: 'var(--color-border)', strokeDasharray: '3 3' }} />
            <text x={-4} y={gy(1) + 3} textAnchor="end" style={axisText}>1</text>
            <text x={-4} y={gy(0) + 3} textAnchor="end" style={axisText}>0</text>
          </g>
          <text x={6} y={11} style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-accent)' }}>the fraction of each Fourier mode the smoother keeps</text>
        </svg>
        <Readouts>
          <Readout id="risk" label="risk R(c)" value={cs.risk[cIdx].toFixed(4)} emphasis />
          <Readout id="bias2" label="bias²" value={cs.bias2[cIdx].toFixed(4)} />
          <Readout id="variance" label="variance" value={cs.var[cIdx].toFixed(4)} />
          <Readout id="rho1" label="gain ρ₁" value={gains[0].toFixed(3)} />
          <Readout id="c-star" label="c* (Theorem 1)" value={cs.c_star.toFixed(3)} emphasis />
          <Readout id="risk-star" label="R(c*)" value={cs.risk_star.toFixed(4)} />
          <Readout id="c-learned" label="trained layer's c" value={cs.c_learned.toFixed(3)} />
        </Readouts>
        <Note>
          Raising c narrows the kernel: the gains rise toward 1, so less of f is smoothed away (bias falls) and more of the noise
          passes through (variance rises). They balance at c*, and the attention layer trained on random functions of this kind
          lands there. At c = 0 every key gets the same weight and the output is the mean.
        </Note>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · A lookup instead of a state — a cache over the document’s own words, and the gradient that reaches back
        </figcaption>
        <Row>
          <Pill active={viewB === 'document'} onClick={() => setViewB('document')} data={{ 'data-view': 'document' }}>a document, token by token</Pill>
          <Pill active={viewB === 'reach'} onClick={() => setViewB('reach')} data={{ 'data-view': 'reach' }}>the gradient across lags</Pill>
        </Row>

        {viewB === 'document' ? (
          <>
            <Row>
              {CACHE.documents.map((d, i) => (
                <Pill key={d.name} active={i === docIdx} onClick={() => pickDoc(i)} data={{ 'data-doc': String(i) }}>{d.name}</Pill>
              ))}
            </Row>
            <Row>
              {VARIANTS.map((v) => (
                <Pill key={v} active={v === variant} onClick={() => setVariant(v)} data={{ 'data-variant': v }}>{VARIANT_LABEL[v]}</Pill>
              ))}
            </Row>
            <div data-panel="tokens" style={{ fontFamily: sans, fontSize: '0.8rem', lineHeight: 1.9, maxHeight: '15rem', overflowY: 'auto',
              border: '1px solid var(--color-border)', borderRadius: '0.5rem', padding: '0.4rem 0.5rem', overflowWrap: 'anywhere' }}>
              {doc.tokens.map((w, i) => {
                const save = kn[i] - vb[i];
                const pct = Math.min(45, (45 * Math.abs(save)) / 20);
                const tint = save > 0 ? 'var(--color-badge-green-text)' : 'var(--color-badge-red-text)';
                const aw = attnW.get(i);
                return (
                  <span key={i} data-token={i} onClick={() => setTok(i)} title={`${w}: ${kn[i].toFixed(2)} bits under Kneser–Ney, ${vb[i].toFixed(2)} under ${VARIANT_LABEL[variant]}`}
                    style={{
                      display: 'inline-block', margin: '0 0.12rem', padding: '0 0.18rem', borderRadius: '0.2rem', cursor: 'pointer',
                      maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', verticalAlign: 'bottom',
                      fontStyle: doc.oov[i] ? 'italic' : 'normal',
                      background: variant === 'kn' || Math.abs(save) < 0.05 ? 'transparent' : `color-mix(in srgb, ${tint} ${pct}%, transparent)`,
                      outline: i === t ? '2px solid var(--color-accent)' : sources.has(i) ? '1.5px dashed var(--color-accent)' : 'none',
                      borderBottom: aw !== undefined ? `3px solid color-mix(in srgb, var(--color-accent-secondary) ${Math.round(25 + 75 * aw / attnMax)}%, transparent)` : '3px solid transparent',
                    }}>
                    {tokenText(w)}{marked.has(i) ? <sup aria-hidden="true" style={{ color: 'var(--color-accent)' }}>•</sup> : null}
                  </span>
                );
              })}
            </div>
            <Note>
              Shading: bits the chosen model saves on each token against Kneser–Ney (green) or spends (red). Italic: not in the
              training vocabulary. ¶: a sentence end. Click any token; for the tokens marked •, the eight earlier positions the
              β = {CACHE.beta} kernel weights most are underlined, darker for more weight, and the word’s own earlier occurrences,
              the positions it is copied from, are outlined with dashes.
            </Note>
            <Readouts>
              <Readout id="bpc-kn" label="document, Kneser–Ney (bits per character)" value={doc.bpc.kn.toFixed(3)} />
              <Readout id="bpc-variant" label={`document, ${VARIANT_LABEL[variant]}`} value={doc.bpc[variant].toFixed(3)} emphasis />
              <Readout id="token" label="token" value={tokenText(doc.tokens[t])} />
              <Readout id="bits-kn" label="its bits, Kneser–Ney" value={kn[t].toFixed(2)} />
              <Readout id="bits-variant" label={`its bits, ${VARIANT_LABEL[variant]}`} value={vb[t].toFixed(2)} emphasis />
            </Readouts>
            {attn && (
              <Readouts>
                <Readout id="copy" label={`cache probability of this word, β = ${CACHE.beta}`} value={attn.copy.toFixed(3)} emphasis />
                <Readout id="sources" label="earlier occurrences it is copied from" value={String(attn.sources.length)} />
              </Readouts>
            )}
            {attn && (
              <div data-panel="weights" style={{ fontFamily: sans, fontSize: '0.78rem', marginTop: '0.5rem', overflowWrap: 'anywhere' }}>
                <span style={{ color: 'var(--color-text-secondary)' }}>{`β = ${CACHE.beta} kernel, heaviest earlier positions: `}</span>
                <span data-readout="weights" style={{ fontFamily: mono }}>
                  {attn.pos.map((p, i) => `${tokenText(doc.tokens[p])} (${attn.w[i].toFixed(3)})`).join(' · ')}
                </span>
              </div>
            )}
            <svg viewBox={`0 0 ${UW} ${UH}`} style={{ width: '100%', height: 'auto', marginTop: '0.6rem' }} role="img" data-panel="u-curve"
              aria-label="Nats per token of Kneser–Ney mixed with the kernel cache, on development and held-out documents, against the inverse bandwidth beta">
              <g transform={`translate(${UM.left},${UM.top})`}>
                {uy.ticks(4).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={UW - UM.left - UM.right} y1={uy(v)} y2={uy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={uy(v) + 3} textAnchor="end" style={axisText}>{v.toFixed(3)}</text>
                  </g>
                ))}
                {CACHE.betas.map((b, i) => (
                  <text key={i} x={ux(i)} y={UH - UM.top - UM.bottom + 14} textAnchor="middle"
                    style={{ ...axisText, fontWeight: i === variantBeta[variant] ? 700 : 400 }}>{`β = ${betaLabel(b)}`}</text>
                ))}
                {([['held_by_beta', 'held out', 'var(--color-accent)', undefined], ['dev_by_beta', 'development', 'var(--color-text-secondary)', '5 3']] as const).map(([key, , color, dash]) => (
                  <g key={key} data-series={key}>
                    <path d={d3.line<number>().x((i) => ux(i)!).y((i) => uy(CACHE[key][i]))(d3.range(CACHE.betas.length))!}
                      style={{ fill: 'none', stroke: color, strokeWidth: 2, strokeDasharray: dash }} />
                    {CACHE[key].map((v, i) => <circle key={i} cx={ux(i)} cy={uy(v)} r={i === iStar ? 5 : 3} style={{ fill: color }} />)}
                  </g>
                ))}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>nats per token, Kneser–Ney ⊕ cache · dev chooses β = {CACHE.beta}</text>
            </svg>
            <Legend items={[{ label: 'held out', color: 'var(--color-accent)' }, { label: 'development', color: 'var(--color-text-secondary)', dash: '5 3' }]} />
            <Readouts>
              <Readout id="held-beta" label={`held out at β = ${CACHE.beta} (dev λ = ${CACHE.lambda})`} value={CACHE.held_by_beta[iStar].toFixed(4)} emphasis />
              <Readout id="held-b0" label="held out at β = 0" value={CACHE.held_by_beta[0].toFixed(4)} />
              <Readout id="held-binf" label="held out at β = ∞" value={CACHE.held_by_beta[CACHE.betas.length - 1].toFixed(4)} />
            </Readouts>
          </>
        ) : (
          <>
            <Row>
              {REACH.lags.map((L, i) => (
                <Pill key={L} active={i === lagIdx} onClick={() => setLagIdx(i)} data={{ 'data-lag': String(L) }}>{`lag ${L}`}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${LW} ${LH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="reach"
              aria-label="The gradient reaching the bit at initialization against the lag, on a log scale, for one attention layer, an Elman network, and two LSTMs">
              <g transform={`translate(${LM.left},${LM.top})`}>
                {[-45, -35, -25, -15, -5].map((v) => (
                  <g key={v}>
                    <line x1={0} x2={LW - LM.left - LM.right} y1={ly(v)} y2={ly(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ly(v) + 3} textAnchor="end" style={axisText}>{`10${String(v).split('').map((ch) => SUP[ch]).join('')}`}</text>
                  </g>
                ))}
                {REACH.lags.map((L, i) => (
                  <text key={L} x={lx(L)} y={LH - LM.top - LM.bottom + 14} textAnchor="middle" style={{ ...axisText, fontWeight: i === lagIdx ? 700 : 400 }}>{L}</text>
                ))}
                <path d={d3.line()(guide)!} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 1, strokeDasharray: '1 3', opacity: 0.8 }} />
                {RECURRENT.map((s) => {
                  const pts = REACH.lags.map((L, i) => [lx(L), ly(Math.log10(REACH[s.key][i]))] as [number, number]);
                  return (
                    <g key={s.key} data-series={s.key}>
                      <path d={d3.line()(pts)!} style={{ fill: 'none', stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />
                      {pts.map(([x, y], i) => <Marker key={i} kind={s.marker} x={x} y={y} color={s.color} size={i === lagIdx ? 5 : 3.5} />)}
                    </g>
                  );
                })}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>gradient reaching the bit at initialization (median of 12)</text>
              <text x={LW / 2} y={LH - 4} textAnchor="middle" style={axisText}>lag L between the bit and the report (log scale) · dotted: 1/(L + 1)</text>
            </svg>
            <Legend items={RECURRENT.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
            <Readouts>
              {RECURRENT.map((s) => (
                <Readout key={s.key} id={`reach-${s.key}`} label={`${s.label}, lag ${REACH.lags[lagIdx]}`} value={sci(REACH[s.key][lagIdx])} emphasis={s.key === 'attention'} />
              ))}
            </Readouts>
            <Readouts>
              <Readout id="acc-attention" label="attention, trained at lag 60: accuracy at lag 60 · at lag 600"
                value={`${REACH.accuracy.attention.toFixed(3)} · ${REACH.accuracy.attention_lag600.toFixed(3)}`} emphasis />
              <Readout id="acc-recurrent" label="Topic 5, trained at lag 60: Elman · LSTM b = 1 · LSTM b = 3"
                value={`${REACH.accuracy.elman.toFixed(3)} · ${REACH.accuracy['lstm-1'].toFixed(3)} · ${REACH.accuracy['lstm-3'].toFixed(3)}`} />
            </Readouts>
            <Note>
              A recurrent cell carries the bit through one step Jacobian per symbol, so its gradient decays geometrically; a forget
              bias of 3 only postpones the decay. Attention reads position 0 directly: its gradient is diluted by the L + 1
              positions sharing the weight, and nothing more, so it follows the dotted 1/(L + 1) line.
            </Note>
          </>
        )}
      </figure>
    </div>
  );
}

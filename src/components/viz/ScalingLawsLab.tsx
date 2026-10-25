import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/scaling-laws/scaling_laws.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make. Change the .py, regenerate, never the reverse. Curves drawn from a fit's parameters are drawn only;
// every number shown as text is a payload value.
import payload from './data/scaling-laws-lab.json';

/**
 * Scaling-Laws Laboratory — three figures for the scaling-laws topic.
 *   A. "Power laws": the memorizer's new-type rate on the filings against Thm 1, and the solvable model's exact SGD
 *      loss L(N, D) (Prop 1) with the Approach-3 fit of Eq. 2, read against D at a fixed size or against N at fixed data.
 *   B. "Splitting a budget": the solvable model's IsoFLOP valley at a chosen budget with Approaches 1 and 2 (Thm 4
 *      predicts N* ∝ C^0.4); the toy transformers' three valleys in Chinchilla's or Kaplan's count; and Prop 3's local
 *      exponent for a published fit read in Kaplan's units.
 *   C. "A finite corpus": the repetition runs against the toy fit at Muennighoff's D′, and the capped optimum at
 *      BloombergGPT's compute as the corpus shrinks.
 */

type Fit = { E: number; A: number; B: number; alpha: number; beta: number };
const LIN = payload.linear as unknown as {
  N: number[]; D: number[]; L: number[][]; fit: Fit; a: { pred: number; a1: number; a2: number; a3: number };
  C: number[]; star1: number[]; star2: number[]; valleys: { N: number[]; L: number[] }[];
};
type Run = { budget: number; d: number; N: number; N_ne: number; D: number; loss: number };
const TOY = payload.toy as unknown as {
  runs: Run[]; N_star: number[]; Nne_star: number[]; Cne_star: number[]; a_chinchilla: number; a_kaplan: number;
  a3: number; fit: Fit; seed_sd: number; coef: number[][]; coef_ne: number[][];
};
const REP = payload.repeat as unknown as {
  N: number; D: number; r_star: number; rows: { epochs: number; unique: number; D_eff: number; loss: number; pred: number }[];
};
const KAP = payload.kaplan as unknown as {
  C: number[]; local: number[]; predicted: number[]; share: number[]; a: number; limit: number; a_small: number;
};
const CAP = payload.cap as unknown as {
  C: number; U: number[]; N: number[]; epochs: number[]; L: number[]; free_N: number; free_D: number; free_L: number;
  bbg_U: number;
};
const MEMO = payload.memorizer as unknown as { n: number[]; rate: number[]; theory: number[]; slope: number };

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3).replace('-', '−');
const BUDGETS = [...new Set(TOY.runs.map((r) => r.budget))];
const BUDGET_COLORS = ['var(--color-badge-blue-text)', 'var(--color-badge-green-text)', 'var(--color-accent)'];

/** 4.20e+10 → "4.2 × 10¹⁰" and 0.002 → "2 × 10⁻³"; numbers from 1 to 10⁵ as integers with thousands separators. */
const SUP: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function sci(v: number, digits = 3): string {
  if (Math.abs(v) >= 1 && Math.abs(v) < 1e5) return Math.round(v).toLocaleString('en-US');
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  return `${Number(m.toPrecision(digits))} × 10${String(e).split('').map((c) => SUP[c]).join('')}`;
}
const billions = (v: number) => `${(v / 1e9).toFixed(1)}B`;

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

function Legend({ items }: { items: { label: string; color: string; dash?: string; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true">
            {s.dot
              ? <circle cx="11" cy="4" r="3.2" style={{ fill: s.color }} />
              : <line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
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
      <label htmlFor={`sl-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`sl-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

const PW = 600, PH = 260, PM = { top: 24, right: 14, bottom: 38, left: 52 };
const IW = PW - PM.left - PM.right, IH = PH - PM.top - PM.bottom;

type Series = { key: string; pts: [number, number][]; color: string; width?: number; dash?: string; dots?: boolean; opacity?: number };

/** A chart on log (or linear) axes; the x axis is always logarithmic. */
function Chart({ id, title, xLabel, yLabel, x, y, series, vlines = [], hlines = [], marks = [], aria }: {
  id: string; title: string; xLabel: string; yLabel: string;
  x: d3.ScaleLogarithmic<number, number>; y: d3.ScaleLogarithmic<number, number> | d3.ScaleLinear<number, number>;
  series: Series[]; vlines?: { v: number; label: string; dash?: string }[]; hlines?: { v: number; label: string; dash?: string }[];
  marks?: { x: number; y: number; color: string; key: string }[]; aria: string;
}) {
  const line = d3.line<[number, number]>().x((p) => x(p[0])).y((p) => y(p[1]));
  const decades = (ts: number[]) => {
    const d = ts.filter((t) => Number.isInteger(Math.log10(t)));
    return d.length >= 2 ? d : ts.filter((t) => /^[125]/.test(t.toExponential()));
  };
  const xt = decades(x.ticks(5));
  const yIsLog = 'base' in y;
  const yt = yIsLog ? decades(y.ticks(4) as number[]) : (y.ticks(5) as number[]);
  const yFmt = (t: number) => (yIsLog ? sci(t, 1) : String(Number(t.toFixed(3))));
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id} aria-label={aria}>
      <text x={8} y={14} style={titleText}>{title}</text>
      <defs>
        <clipPath id={`${id}-clip`}><rect x={0} y={0} width={IW} height={IH} /></clipPath>
      </defs>
      <g transform={`translate(${PM.left},${PM.top})`}>
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={IH} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={x(t)} y={IH + 13} textAnchor="middle" style={axisText}>{sci(t, 1)}</text>
          </g>
        ))}
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-5} y={y(t) + 3} textAnchor="end" style={axisText}>{yFmt(t)}</text>
          </g>
        ))}
        <g clipPath={`url(#${id}-clip)`}>
          {hlines.map((h) => (
            <g key={`h${h.label}`}>
              <line x1={0} x2={IW} y1={y(h.v)} y2={y(h.v)} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: h.dash ?? '4 3' }} />
              <text x={IW - 4} y={y(h.v) - 3} textAnchor="end" style={axisText}>{h.label}</text>
            </g>
          ))}
          {vlines.map((v) => (
            <g key={`v${v.label}`}>
              <line x1={x(v.v)} x2={x(v.v)} y1={0} y2={IH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: v.dash ?? '4 3' }} />
              <text x={x(v.v) + 3} y={10} style={axisText}>{v.label}</text>
            </g>
          ))}
          {series.map((s) => (
            <g key={s.key} data-series={s.key} style={{ opacity: s.opacity ?? 1 }}>
              {s.pts.length > 1 && !s.dots && (
                <path d={line(s.pts) ?? ''} style={{ fill: 'none', stroke: s.color, strokeWidth: s.width ?? 1.6, strokeDasharray: s.dash }} />
              )}
              {s.dots && s.pts.map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r={3.2} style={{ fill: s.color }} />)}
            </g>
          ))}
          {marks.map((m) => (
            <circle key={m.key} data-mark={m.key} cx={x(m.x)} cy={y(m.y)} r={5.5}
              style={{ fill: 'none', stroke: m.color, strokeWidth: 2 }} />
          ))}
        </g>
      </g>
      <text x={PM.left + IW / 2} y={PH - 4} textAnchor="middle" style={axisText}>{xLabel}</text>
      <text transform={`translate(11,${PM.top + IH / 2}) rotate(-90)`} textAnchor="middle" style={axisText}>{yLabel}</text>
    </svg>
  );
}

const fitLoss = (f: Fit, N: number, D: number) => f.E + f.A / N ** f.alpha + f.B / D ** f.beta;
const logGrid = (lo: number, hi: number, k = 60) => d3.range(k + 1).map((i) => lo * (hi / lo) ** (i / k));
const parabola = (c: number[], N: number) => c[0] * Math.log(N) ** 2 + c[1] * Math.log(N) + c[2];

export default function ScalingLawsLab() {
  const [viewA, setViewA] = useState<'memo' | 'linear'>('linear');
  const [axisA, setAxisA] = useState<'D' | 'N'>('D');
  const [nIdx, setNIdx] = useState(6);
  const [dIdx, setDIdx] = useState(24);
  const [viewB, setViewB] = useState<'linear' | 'toy' | 'kaplan'>('linear');
  const [cIdx, setCIdx] = useState(12);
  const [count, setCount] = useState<'total' | 'ne'>('total');
  const [kIdx, setKIdx] = useState(8);
  const [uIdx, setUIdx] = useState(8);

  // ----- Figure A -----
  const lossAll = LIN.L.flat();
  const yLin = d3.scaleLog().domain([d3.min(lossAll)! * 0.8, d3.max(lossAll)! * 1.2]).range([IH, 0]);
  const xD = d3.scaleLog().domain([LIN.D[0], LIN.D[LIN.D.length - 1]]).range([0, IW]);
  const xN = d3.scaleLog().domain([LIN.N[0], LIN.N[LIN.N.length - 1]]).range([0, IW]);
  const vsD: Series[] = LIN.N.map((n, i) => ({
    key: `N${n}`, pts: LIN.D.map((d, j) => [d, LIN.L[i][j]] as [number, number]),
    color: i === nIdx ? 'var(--color-accent)' : 'var(--color-text-secondary)', width: i === nIdx ? 2.4 : 0.8, opacity: i === nIdx ? 1 : 0.45,
  }));
  vsD.push({ key: 'fit', pts: LIN.D.map((d) => [d, fitLoss(LIN.fit, LIN.N[nIdx], d)] as [number, number]), color: 'var(--color-badge-blue-text)', dash: '5 4', width: 1.8 });
  const dShown = d3.range(0, LIN.D.length, 4);
  const vsN: Series[] = dShown.map((j) => ({
    key: `D${LIN.D[j]}`, pts: LIN.N.map((n, i) => [n, LIN.L[i][j]] as [number, number]),
    color: j === dIdx ? 'var(--color-accent)' : 'var(--color-text-secondary)', width: j === dIdx ? 2.4 : 0.8, opacity: j === dIdx ? 1 : 0.45,
  }));
  if (!dShown.includes(dIdx)) {
    vsN.push({ key: `D${LIN.D[dIdx]}`, pts: LIN.N.map((n, i) => [n, LIN.L[i][dIdx]] as [number, number]), color: 'var(--color-accent)', width: 2.4 });
  }
  vsN.push({ key: 'fit', pts: logGrid(LIN.N[0], LIN.N[LIN.N.length - 1]).map((n) => [n, fitLoss(LIN.fit, n, LIN.D[dIdx])] as [number, number]), color: 'var(--color-badge-blue-text)', dash: '5 4', width: 1.8 });
  const xMemo = d3.scaleLog().domain([MEMO.n[0], MEMO.n[MEMO.n.length - 1]]).range([0, IW]);
  const yMemo = d3.scaleLog().domain([d3.min([...MEMO.rate, ...MEMO.theory])! * 0.8, 1]).range([IH, 0]);

  // ----- Figure B -----
  const valley = LIN.valleys[cIdx];
  const yValley = d3.scaleLog().domain([d3.min(valley.L)! * 0.8, d3.max(valley.L)! * 1.25]).range([IH, 0]);
  const toyN = (r: Run) => (count === 'total' ? r.N : r.N_ne);
  const toyStars = count === 'total' ? TOY.N_star : TOY.Nne_star;
  const toyCoef = count === 'total' ? TOY.coef : TOY.coef_ne;
  const toyXs = TOY.runs.map(toyN);
  const xToy = d3.scaleLog().domain([d3.min(toyXs)! * 0.8, d3.max(toyXs)! * 1.25]).range([0, IW]);
  const yToy = d3.scaleLinear().domain([1.8, 3.8]).range([IH, 0]);
  const toySeries: Series[] = BUDGETS.flatMap((C, b) => {
    const rs = TOY.runs.filter((r) => r.budget === C);
    const lo = d3.min(rs, toyN)!, hi = d3.max(rs, toyN)!;
    return [
      { key: `runs-${b}`, pts: rs.map((r) => [toyN(r), r.loss] as [number, number]), color: BUDGET_COLORS[b], dots: true },
      { key: `parabola-${b}`, pts: logGrid(lo, hi, 40).map((n) => [n, parabola(toyCoef[b], n)] as [number, number]), color: BUDGET_COLORS[b], width: 1.4, dash: '4 3' },
    ];
  });
  const xKap = d3.scaleLog().domain([KAP.C[0], KAP.C[KAP.C.length - 1]]).range([0, IW]);
  const yKap = d3.scaleLinear().domain([0.45, 0.8]).range([IH, 0]);

  // ----- Figure C -----
  const xCap = d3.scaleLog().domain([CAP.U[0], CAP.U[CAP.U.length - 1]]).range([0, IW]);
  const yCap = d3.scaleLog().domain([3e10, 1e11]).range([IH, 0]);
  const repMax = 3.4, repMin = 2.0;
  const rx = d3.scaleLinear().domain([repMin, repMax]).range([0, 600 - 150 - 50]);

  return (
    <div data-lab="scaling-laws" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Power laws — Zipf’s tail for data, a spectrum’s tail for size
        </figcaption>
        <Row>
          <Pill active={viewA === 'linear'} onClick={() => setViewA('linear')} data={{ 'data-view': 'linear' }}>solvable model</Pill>
          <Pill active={viewA === 'memo'} onClick={() => setViewA('memo')} data={{ 'data-view': 'memo' }}>memorizer on the filings</Pill>
        </Row>

        {viewA === 'linear' && (
          <>
            <Row>
              <Pill active={axisA === 'D'} onClick={() => setAxisA('D')} data={{ 'data-axis': 'D' }}>loss against data D</Pill>
              <Pill active={axisA === 'N'} onClick={() => setAxisA('N')} data={{ 'data-axis': 'N' }}>loss against size N</Pill>
            </Row>
            {axisA === 'D' ? (
              <>
                <Slider id="n" label="model size N" value={nIdx} max={LIN.N.length - 1} onChange={setNIdx} text={String(LIN.N[nIdx])} />
                <Chart id="lin-vs-d" title={`exact SGD loss, all 15 sizes; size N = ${LIN.N[nIdx]} highlighted`} xLabel="samples D (one SGD step each)"
                  yLabel="loss" x={xD} y={yLin} series={vsD}
                  aria={`Solvable model: loss against data for 15 sizes, size ${LIN.N[nIdx]} highlighted, with the fitted law`} />
              </>
            ) : (
              <>
                <Slider id="d" label="samples D" value={dIdx} max={LIN.D.length - 1} onChange={setDIdx} text={LIN.D[dIdx].toLocaleString('en-US')} />
                <Chart id="lin-vs-n" title={`exact SGD loss against size; D = ${LIN.D[dIdx].toLocaleString('en-US')} highlighted`} xLabel="model size N (coordinates kept)"
                  yLabel="loss" x={xN} y={yLin} series={vsN}
                  aria={`Solvable model: loss against size at ${LIN.D[dIdx]} samples, with the fitted law`} />
              </>
            )}
            <Legend items={[
              { label: 'exact expected loss (Proposition 1)', color: 'var(--color-accent)' },
              { label: 'fitted E + A/N^α + B/D^β (Approach 3)', color: 'var(--color-badge-blue-text)', dash: '5 4' },
            ]} />
            <Readouts>
              {axisA === 'D'
                ? <Readout id="lin-final" label={`exact loss at D = ${LIN.D[LIN.D.length - 1].toLocaleString('en-US')}`} value={String(LIN.L[nIdx][LIN.D.length - 1])} emphasis />
                : <Readout id="lin-final" label={`exact loss at N = ${LIN.N[LIN.N.length - 1]}`} value={String(LIN.L[LIN.N.length - 1][dIdx])} emphasis />}
              <Readout id="alpha" label="fitted α (Theorem 2: b − 1 = 1.5)" value={f3(LIN.fit.alpha)} />
              <Readout id="beta" label="fitted β (Theorem 3: (b − 1)/a = 1)" value={f3(LIN.fit.beta)} />
            </Readouts>
            <Note>
              Against D, each size’s curve falls with slope near −1 until it reaches the tail its N coordinates cannot touch,
              then flattens; bigger models flatten later and lower. The fitted law tracks the large models closely and
              overstates the small ones: a size-N model’s data term sums only its N kept coordinates.
            </Note>
          </>
        )}

        {viewA === 'memo' && (
          <>
            <Chart id="memo" title="share of tokens whose type is new, by position in the corpus" xLabel="tokens read so far n"
              yLabel="new-type rate" x={xMemo} y={yMemo}
              series={[
                { key: 'theory', pts: MEMO.n.map((n, i) => [n, MEMO.theory[i]] as [number, number]), color: 'var(--color-badge-blue-text)', dash: '5 4', width: 1.8 },
                { key: 'corpus', pts: MEMO.n.map((n, i) => [n, MEMO.rate[i]] as [number, number]), color: 'var(--color-accent)', width: 2.2 },
              ]}
              vlines={[{ v: 1e4, label: 'fit starts' }]}
              aria={`The filings' new-type rate against tokens read, slope ${f3(MEMO.slope)}, beside Theorem 1 for one restaurant`} />
            <Legend items={[
              { label: 'the filings (seven restaurants and template words)', color: 'var(--color-accent)' },
              { label: 'Theorem 1, one restaurant (d = 0.6, θ = 20)', color: 'var(--color-badge-blue-text)', dash: '5 4' },
            ]} />
            <Readouts>
              <Readout id="memo-slope" label="the filings’ slope past 10k tokens" value={f3(MEMO.slope)} emphasis />
              <Readout id="memo-theory" label="Theorem 1’s exponent −(1 − d)" value="−0.400" />
            </Readouts>
            <Note>
              A memorizer errs exactly when the next type is new. The filings follow one restaurant’s curve for their first
              two thousand tokens, then settle about a quarter below it, because their template words stop producing new
              types; past ten thousand tokens the two run parallel, at the slope of Zipf’s tail.
            </Note>
          </>
        )}
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Splitting a budget — IsoFLOP valleys, Theorem 4, and Kaplan’s units
        </figcaption>
        <Row>
          <Pill active={viewB === 'linear'} onClick={() => setViewB('linear')} data={{ 'data-view': 'iso-linear' }}>solvable model</Pill>
          <Pill active={viewB === 'toy'} onClick={() => setViewB('toy')} data={{ 'data-view': 'iso-toy' }}>toy transformers</Pill>
          <Pill active={viewB === 'kaplan'} onClick={() => setViewB('kaplan')} data={{ 'data-view': 'iso-kaplan' }}>Kaplan’s units</Pill>
        </Row>

        {viewB === 'linear' && (
          <>
            <Slider id="c" label="budget C = N·D" value={cIdx} max={LIN.C.length - 1} onChange={setCIdx} text={sci(LIN.C[cIdx])} />
            <Chart id="iso-linear" title={`every size that fits budget C = ${sci(LIN.C[cIdx])}, trained on D = C/N samples`}
              xLabel="model size N" yLabel="loss" x={xN} y={yValley}
              series={[{ key: 'valley', pts: valley.N.map((n, i) => [n, valley.L[i]] as [number, number]), color: 'var(--color-accent)', width: 2 },
                { key: 'valley-pts', pts: valley.N.map((n, i) => [n, valley.L[i]] as [number, number]), color: 'var(--color-accent)', dots: true }]}
              vlines={[{ v: LIN.star2[cIdx], label: 'vertex' }]}
              marks={[{ key: 'star1', x: LIN.star1[cIdx], y: valley.L[valley.N.indexOf(LIN.star1[cIdx])], color: 'var(--color-badge-blue-text)' }]}
              aria={`IsoFLOP valley at budget ${LIN.C[cIdx]}: lowest size ${LIN.star1[cIdx]}, parabola vertex ${LIN.star2[cIdx]}`} />
            <Legend items={[
              { label: 'loss at each size on this budget', color: 'var(--color-accent)', dot: true },
              { label: 'lowest size (Approach 1)', color: 'var(--color-badge-blue-text)', dot: true },
              { label: 'parabola vertex (Approach 2)', color: 'var(--color-text-secondary)', dash: '4 3' },
            ]} />
            <Readouts>
              <Readout id="star1" label="lowest size (Approach 1)" value={String(LIN.star1[cIdx])} emphasis />
              <Readout id="star2" label="parabola vertex (Approach 2)" value={String(LIN.star2[cIdx])} />
              <Readout id="a-pred" label="Theorem 4: N* ∝ C^a, a = 1/(1 + a_λ)" value={f3(LIN.a.pred)} />
              <Readout id="a-fit" label="a from Approaches 1 / 2 / 3" value={`${f3(LIN.a.a1)} / ${f3(LIN.a.a2)} / ${f3(LIN.a.a3)}`} />
            </Readouts>
            <Note>
              Left of the floor the model is too small and sits on its tail; right of it the model is too big for its data.
              Every factor of 16 in compute moves the floor right by about 16^0.4 ≈ 3.
            </Note>
          </>
        )}

        {viewB === 'toy' && (
          <>
            <Row>
              <Pill active={count === 'total'} onClick={() => setCount('total')} data={{ 'data-count': 'total' }}>Chinchilla’s count (with embedding)</Pill>
              <Pill active={count === 'ne'} onClick={() => setCount('ne')} data={{ 'data-count': 'ne' }}>Kaplan’s count (non-embedding)</Pill>
            </Row>
            <Chart id="iso-toy" title="three budgets, five widths each: held-out loss on the filings" xLabel={count === 'total' ? 'weights N (with the tied embedding)' : 'non-embedding weights N_ne'}
              yLabel="loss (nats)" x={xToy} y={yToy} series={toySeries}
              marks={BUDGETS.map((C, b) => ({ key: `star-${b}`, x: toyStars[b], y: parabola(toyCoef[b], toyStars[b]), color: BUDGET_COLORS[b] }))}
              aria={`Toy transformers: three IsoFLOP valleys with their parabolas, in ${count === 'total' ? 'total' : 'non-embedding'} parameters`} />
            <Legend items={BUDGETS.map((C, b) => ({ label: `C = ${sci(C, 2)}`, color: BUDGET_COLORS[b], dot: true }))} />
            <Readouts>
              <Readout id="toy-a" label={count === 'total' ? 'a, Approach 2 (Chinchilla’s count)' : 'a, Approach 2 (Kaplan’s count and compute)'}
                value={f3(count === 'total' ? TOY.a_chinchilla : TOY.a_kaplan)} emphasis />
              <Readout id="toy-stars" label="vertices N*" value={toyStars.map((v) => Math.round(v).toLocaleString('en-US')).join(' / ')} />
              <Readout id="toy-a3" label="a, Approach 3" value={f3(TOY.a3)} />
              <Readout id="toy-sd" label="one point’s seed spread" value={f3(TOY.seed_sd)} />
            </Readouts>
            <Note>
              Each valley’s floor moves right with the budget. In Kaplan’s count the small models shrink most, because the
              embedding is most of them, so the same three floors give a steeper exponent. Three budgets a factor of 6.3 apart
              cannot tell 0.5 from 0.73.
            </Note>
          </>
        )}

        {viewB === 'kaplan' && (
          <>
            <Slider id="k" label="compute C" value={kIdx} max={KAP.C.length - 1} onChange={setKIdx} text={sci(KAP.C[kIdx])} />
            <Chart id="kaplan" title="Besiroglu et al.’s law, sized in total parameters, measured the way Kaplan et al. did"
              xLabel="training compute C (FLOPs)" yLabel="local exponent of N_ne* in C_ne" x={xKap} y={yKap}
              series={[
                { key: 'local', pts: KAP.C.map((c, i) => [c, KAP.local[i]] as [number, number]), color: 'var(--color-accent)', width: 2.4 },
                { key: 'predicted', pts: KAP.C.map((c, i) => [c, KAP.predicted[i]] as [number, number]), color: 'var(--color-badge-blue-text)', dash: '2 3', width: 2 },
              ]}
              hlines={[{ v: 0.73, label: 'Kaplan 0.73' }, { v: KAP.a, label: `total-parameter a = ${f3(KAP.a)}`, dash: '1 3' }]}
              vlines={[{ v: KAP.C[kIdx], label: '' }]}
              marks={[{ key: 'kap-sel', x: KAP.C[kIdx], y: KAP.local[kIdx], color: 'var(--color-accent)' }]}
              aria={`Local exponent in Kaplan's units against compute: ${f3(KAP.local[kIdx])} at ${KAP.C[kIdx]} FLOPs`} />
            <Legend items={[
              { label: 'finite-difference local exponent', color: 'var(--color-accent)' },
              { label: 'Proposition 3: ρa/(ρa + 1 − a)', color: 'var(--color-badge-blue-text)', dash: '2 3' },
            ]} />
            <Readouts>
              <Readout id="kap-local" label="local exponent" value={f3(KAP.local[kIdx])} emphasis />
              <Readout id="kap-pred" label="Proposition 3" value={f3(KAP.predicted[kIdx])} />
              <Readout id="kap-share" label="embedding share of the optimal model" value={f3(KAP.share[kIdx])} />
              <Readout id="kap-small" label="regression below 10¹⁸ FLOPs" value={f3(KAP.a_small)} />
            </Readouts>
            <Note>
              GPT-2’s vocabulary and context, d_model = 64 · n_layer. Where the embedding dominates, counting only the rest
              makes the optimum look like it grows faster with compute, toward 3a/(1 + 2a) = {f3(KAP.limit)}; by 10²³ FLOPs the
              embedding is under 1% of the model and the two counts agree.
            </Note>
          </>
        )}
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          C · A finite corpus — repeated epochs and the capped optimum
        </figcaption>
        <svg viewBox="0 0 600 170" style={{ width: '100%', height: 'auto' }} role="img" data-panel="repeat"
          aria-label={`Same compute on fewer unique tokens: ${REP.rows.map((r) => `${r.epochs} epochs ${f3(r.loss)}`).join(', ')}`}>
          <text x={8} y={14} style={titleText}>{`d = 12, ${REP.D.toLocaleString('en-US')} training tokens; the corpus repeated 1, 4, 16, 64 times`}</text>
          <g transform="translate(150,26)">
            {[2.0, 2.4, 2.8, 3.2].map((v) => (
              <g key={v}>
                <line x1={rx(v)} x2={rx(v)} y1={0} y2={4 * 30} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                <text x={rx(v)} y={4 * 30 + 13} textAnchor="middle" style={axisText}>{v.toFixed(1)}</text>
              </g>
            ))}
            {REP.rows.map((r, i) => (
              <g key={r.epochs} data-row={i}>
                <text x={-6} y={i * 30 + 15} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>
                  {`${r.epochs} epoch${r.epochs > 1 ? 's' : ''} (${r.unique.toLocaleString('en-US')} unique)`}
                </text>
                <rect data-series="observed" x={0} y={i * 30 + 4} width={rx(r.loss)} height={10} style={{ fill: 'var(--color-accent)' }} />
                <rect data-series="predicted" x={0} y={i * 30 + 15} width={rx(r.pred)} height={6} style={{ fill: 'var(--color-badge-blue-text)' }} />
                <text x={rx(r.loss) + 4} y={i * 30 + 13} style={{ ...axisText, fontFamily: mono }}>{f3(r.loss)}</text>
                <text x={rx(r.pred) + 4} y={i * 30 + 22} style={{ ...axisText, fontFamily: mono }}>{f3(r.pred)}</text>
              </g>
            ))}
          </g>
          <text x={150 + rx(2.7)} y={166} textAnchor="middle" style={axisText}>held-out loss (nats); bars start at 2.0</text>
        </svg>
        <Legend items={[
          { label: 'observed', color: 'var(--color-accent)' },
          { label: 'toy fit at Muennighoff’s effective data D′ (R* = 15.4)', color: 'var(--color-badge-blue-text)' },
        ]} />
        <Note>
          Four epochs cost {f3(REP.rows[1].loss - REP.rows[0].loss)} nats; sixty-four cost half a nat. The templated filings
          forgive repetition more than Muennighoff’s law predicts.
        </Note>

        <div style={{ marginTop: '1.25rem' }}>
          <Slider id="u" label="unique tokens U" value={uIdx} max={CAP.U.length - 1} onChange={setUIdx} text={billions(CAP.U[uIdx])} />
          <Chart id="cap" title={`compute-optimal size at BloombergGPT’s compute, ${sci(CAP.C)} FLOPs, under a data cap`}
            xLabel="unique tokens U" yLabel="optimal parameters N*" x={xCap} y={yCap}
            series={[{ key: 'capped', pts: CAP.U.map((u, i) => [u, CAP.N[i]] as [number, number]), color: 'var(--color-accent)', width: 2.2 }]}
            hlines={[{ v: CAP.free_N, label: `unconstrained ${billions(CAP.free_N)}` }]}
            vlines={[{ v: CAP.bbg_U, label: '709B available' }, { v: CAP.free_D, label: 'D* wanted', dash: '1 3' }]}
            marks={[{ key: 'cap-sel', x: CAP.U[uIdx], y: CAP.N[uIdx], color: 'var(--color-accent)' }]}
            aria={`With ${CAP.U[uIdx]} unique tokens the optimum is ${CAP.N[uIdx]} parameters on ${CAP.epochs[uIdx]} epochs`} />
          <Readouts>
            <Readout id="cap-n" label="optimal parameters N*" value={billions(CAP.N[uIdx])} emphasis />
            <Readout id="cap-epochs" label="epochs over the corpus" value={CAP.epochs[uIdx].toFixed(2)} />
            <Readout id="cap-loss" label="predicted loss (Besiroglu et al.’s fit)" value={CAP.L[uIdx].toFixed(4)} />
            <Readout id="cap-free" label="unconstrained loss" value={CAP.free_L.toFixed(4)} />
          </Readouts>
          <Note>
            Above the unconstrained D* the cap does not bind. Just below it, a few repeated epochs are nearly free. Far below
            it, more passes stop paying, and the compute goes into a bigger model.
          </Note>
        </div>
      </figure>
    </div>
  );
}

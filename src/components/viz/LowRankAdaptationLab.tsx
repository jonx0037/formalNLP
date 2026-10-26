import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/fine-tuning-low-rank-adaptation/fine_tuning_low_rank_adaptation.py; test_viz_mirror_matches_payload asserts
// this file equals it exactly, and the harness's tests assert the claims the views make (test_thm2_conservation,
// test_thm2_adam_breaks_balance, test_thm1_rank_scaling_at_the_encoder, test_prop2_update_is_low_rank,
// test_lpft_forgets_less, test_lpft_dominates_lora, test_prop3_l2sp_tradeoff, test_prop4_intrinsic_dimension,
// test_prop5_kernel_regime, test_negation_shortcut_and_augmentation, test_tone_by_aggregation). Change the .py,
// regenerate, never the reverse.
import payload from './data/low-rank-adaptation-lab.json';

/**
 * Low-Rank Adaptation Laboratory — five figures for the fine-tuning-low-rank-adaptation topic.
 *   A. "One weight, two factors": w = ab in the (a, b) plane, the minimizer curve ab = w*, and the paths of GD and
 *      Adam from (a₀, 0) against the hyperbola a² − b² = a₀² that gradient flow conserves (Theorem 2).
 *   B. "Rank and scaling": the first step's size against r under α/r and α/√r, for SGD and Adam (Theorem 1), and what
 *      training made of each rank.
 *   C. "What full fine-tuning changed": each block matrix's singular values against a same-norm Gaussian's edge, and
 *      truncated merges (Proposition 2).
 *   D. "Learning vs. forgetting": every run's test accuracy against its held-out masked-LM loss (Theorem 3,
 *      Proposition 3), with the intrinsic-dimension grid (Proposition 4) and the kernel experiment (Proposition 5).
 *   E. "Reading negation": accuracy by construction, the negation quadruples before and after the counterfactual rows,
 *      and document tone.
 */

type Pt = [number, number];
const SC = payload.scalar as unknown as {
  w_star: number; a0: number[]; eta: number[]; every: number; steps: number;
  paths: Record<'gd' | 'adam', Record<string, Record<string, Pt[]>>>;
  max_drift: Record<'gd' | 'adam', Record<string, Record<string, number>>>;
};
type Spread = { mean: number; min: number; max: number };
type Trained = { r: number; acc: number; mlm: number; norm: number };
const RK = payload.rank as unknown as {
  r: number[]; first_step: Record<string, number[]>; expected_sgd: Record<'lin' | 'sqrt', number[]>; eta: number;
  alpha: number; trained: Record<'lin' | 'sqrt', Trained[]>; probe: number; full: Spread; lpft: Spread; lora: Spread;
};
type Mat = { block: number; matrix: string; shape: number[]; sv: number[]; edge: number; stable_rank: number; top4: number; top16: number };
const SP = payload.spectrum as unknown as { matrices: Mat[]; trunc: { r: number[]; acc: number[]; full: number } };
type Point = { label: string; method: string; acc: number; mlm: number; augment: boolean };
type Arm = { acc: number; loss: number; agree: number; kernel_move: number; balance_drift: number };
const FR = payload.frontier as unknown as {
  points: Point[]; mlm_pretrained: number;
  d90: Record<'pretrained' | 'random', { D: number[]; acc: number[]; full: number; threshold: number; d90: number | null }>;
  ntk: { acc_start: number; loss_start: number; acc_kernel: number; loss_kernel: number; gd: Arm; adam: Arm };
};
const NG = payload.negation as unknown as {
  constructions: string[]; n: number[]; readers: { label: string; by: number[]; acc: number }[];
  quads: Record<'full' | 'augmented', number[][]>; tone: { label: string; acc: number }[]; ceiling: number;
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const g3 = (v: number) => (v >= 0.01 ? v.toFixed(3) : v.toPrecision(3)); // small SGD steps keep 3 significant figures
const W = 640;

function Pill({ active, onClick, children, data }: { active: boolean; onClick: () => void; children: ReactNode; data: Record<string, string> }) {
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
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', alignItems: 'center', ...style }}>{children}</div>;
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

function Label({ children }: { children: ReactNode }) {
  return <span style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>{children}</span>;
}

function Key({ items }: { items: { label: string; color: string; dash?: boolean; hollow?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: '0.9rem', height: s.dash ? 0 : '0.55rem',
            borderTop: s.dash ? `2px dashed ${s.color}` : undefined,
            background: s.dash || s.hollow ? undefined : s.color, border: s.hollow ? `1.5px solid ${s.color}` : undefined,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

// ----- A. One weight, two factors -----

const PLANE = 2.6;

function ScalarView() {
  const [ai, setAi] = useState(1);
  const [ei, setEi] = useState(0);
  const a0 = SC.a0[ai], eta = SC.eta[ei];
  const ka = `${a0}`, ke = `${eta}`;
  const S = 360, M = 34;
  const x = d3.scaleLinear().domain([0, PLANE]).range([M, S - 10]);
  const y = d3.scaleLinear().domain([0, PLANE]).range([S - M, 10]);
  const line = d3.line<Pt>().x((p) => x(p[0])).y((p) => y(p[1]));
  const level = (c: number) => d3.range(c / PLANE, PLANE + 1e-9, 0.02).map((a) => [a, c / a] as Pt).filter((p) => p[1] <= PLANE);
  const hyper = d3.range(0, PLANE, 0.02).map((b) => [Math.sqrt(a0 * a0 + b * b), b] as Pt).filter((p) => p[0] <= PLANE);
  const gd = SC.paths.gd[ka][ke], adam = SC.paths.adam[ka][ke];
  const ticks = [0, 0.5, 1, 1.5, 2, 2.5];
  return (
    <figure style={{ margin: 0 }} data-view="scalar">
      <Caption>One weight, two factors: w = ab, from (a₀, 0) toward ab = {SC.w_star}</Caption>
      <Row>
        <Label>start a₀</Label>
        {SC.a0.map((v, i) => <Pill key={v} active={ai === i} onClick={() => setAi(i)} data={{ 'data-a0': String(v) }}>{String(v)}</Pill>)}
      </Row>
      <Row>
        <Label>step size η</Label>
        {SC.eta.map((v, i) => <Pill key={v} active={ei === i} onClick={() => setEi(i)} data={{ 'data-eta': String(v) }}>{String(v)}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: '26rem', height: 'auto' }} role="img" data-panel="plane"
        aria-label={`GD and Adam paths from a0 = ${a0} with step ${eta}; GD ends at (${gd[gd.length - 1].join(', ')}), Adam at (${adam[adam.length - 1].join(', ')})`}>
        {ticks.map((t) => (
          <g key={t}>
            <text x={x(t)} y={S - M + 14} textAnchor="middle" style={axisText}>{t}</text>
            <text x={M - 6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
          </g>
        ))}
        <text x={(M + S) / 2} y={S - 4} textAnchor="middle" style={axisText}>a (random factor)</text>
        <text x={10} y={(S - M) / 2} textAnchor="middle" transform={`rotate(-90 10 ${(S - M) / 2})`} style={axisText}>b (starts at 0)</text>
        <rect x={M} y={10} width={S - 10 - M} height={S - M - 10} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
        {[0.5, 1, 1.5, 2.5, 3.5, 5].map((c) => (
          <path key={c} d={line(level(c)) ?? ''} style={{ fill: 'none', stroke: 'var(--color-border)', strokeWidth: 1 }} />
        ))}
        <path d={line(level(SC.w_star)) ?? ''} style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 2 }} data-curve="minimizers" />
        <path d={line(hyper) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 2, strokeDasharray: '5 3' }} data-curve="hyperbola" />
        <path d={line(gd) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} data-path="gd" />
        <path d={line(adam) ?? ''} style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: 2 }} data-path="adam" />
        <circle cx={x(a0)} cy={y(0)} r={4} style={{ fill: 'var(--color-text)' }} />
      </svg>
      <Key items={[{ label: `minimizers ab = ${SC.w_star}`, color: 'var(--color-text)' },
        { label: 'a² − b² = a₀² (conserved by gradient flow)', color: 'var(--color-accent-secondary)', dash: true },
        { label: 'gradient descent', color: 'var(--color-accent)' }, { label: 'Adam', color: 'var(--color-badge-blue-text)' }]} />
      <Readouts>
        <Readout label="GD ends at (a, b)" value={`(${f3(gd[gd.length - 1][0])}, ${f3(gd[gd.length - 1][1])})`} id="scalar-gd-end" />
        <Readout label="Adam ends at (a, b)" value={`(${f3(adam[adam.length - 1][0])}, ${f3(adam[adam.length - 1][1])})`} id="scalar-adam-end" />
        <Readout label="GD: max |a² − b² − a₀²|" value={SC.max_drift.gd[ka][ke].toFixed(4)} id="scalar-gd-drift" emphasis />
        <Readout label="Adam: max |a² − b² − a₀²|" value={SC.max_drift.adam[ka][ke].toFixed(4)} id="scalar-adam-drift" />
      </Readouts>
      <Note>
        Theorem 2: gradient flow keeps a² − b² fixed, so the path rides its hyperbola to the one minimizer on it, and b can
        grow only as fast as a. GD's drift is O(η) over the run, so it grows with the step; Adam rescales each coordinate
        and leaves the hyperbola at once. Both paths run {SC.steps} steps.
      </Note>
    </figure>
  );
}

// ----- B. Rank and scaling -----

const OPT_LABEL = { sgd: 'SGD', adam: 'Adam' } as const;
const METRIC = [
  { key: 'acc', label: 'test accuracy' }, { key: 'mlm', label: 'masked-LM loss after' }, { key: 'norm', label: '‖sAB‖ after training' },
] as const;

function RankView() {
  const [opt, setOpt] = useState<'sgd' | 'adam'>('adam');
  const [metric, setMetric] = useState<(typeof METRIC)[number]['key']>('acc');
  const LW = 64, IW = W - LW - 20, H = 200;
  const xr = d3.scaleLog().base(2).domain([1, 64]).range([0, IW]);
  const sq = RK.first_step[`${opt}_sqrt`], li = RK.first_step[`${opt}_lin`];
  const vals = [...sq, ...li, ...(opt === 'sgd' ? [...RK.expected_sgd.sqrt, ...RK.expected_sgd.lin] : [])];
  const yv = d3.scaleLog().domain([d3.min(vals)! * 0.8, d3.max(vals)! * 1.25]).range([H, 0]);
  const ln = (v: number[]) => d3.line<number>().x((_, i) => xr(RK.r[i])).y((d) => yv(d))(v) ?? '';
  const tr = RK.trained;
  const mv = (t: Trained) => t[metric];
  const all = [...tr.lin.map(mv), ...tr.sqrt.map(mv), ...(metric === 'acc' ? [RK.probe, RK.full.max, RK.lpft.max] : [])];
  const lo = d3.min(all)!, hi = d3.max(all)!;
  const y2 = d3.scaleLinear().domain([lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1]).range([H, 0]);
  const ln2 = (v: Trained[]) => d3.line<Trained>().x((d) => xr(d.r)).y((d) => y2(mv(d)))(v) ?? '';
  const SQ = 'var(--color-accent)', LI = 'var(--color-badge-blue-text)';
  return (
    <figure style={{ margin: 0 }} data-view="rank">
      <Caption>The first step against the rank, and what training made of it</Caption>
      <Row>
        <Label>optimizer</Label>
        {(['adam', 'sgd'] as const).map((o) => <Pill key={o} active={opt === o} onClick={() => setOpt(o)} data={{ 'data-opt': o }}>{OPT_LABEL[o]}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${W} ${H + 46}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="first-step"
        aria-label={`${OPT_LABEL[opt]} first-step size against r: alpha over root r ${sq.map(g3).join(', ')}; alpha over r ${li.map(g3).join(', ')}`}>
        <text x={8} y={12} style={titleText}>‖ΔW₁‖ of the first step from B = 0 (log scale, mean of 16 draws of A₀)</text>
        <g transform={`translate(${LW},22)`}>
          {RK.r.map((r) => <text key={r} x={xr(r)} y={H + 14} textAnchor="middle" style={axisText}>{r}</text>)}
          <text x={IW / 2} y={H + 24} textAnchor="middle" style={axisText}>rank r</text>
          {yv.ticks(4).map((t) => (
            <g key={t}><line x1={0} x2={IW} y1={yv(t)} y2={yv(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={yv(t) + 3} textAnchor="end" style={axisText}>{t >= 0.01 ? t.toPrecision(2) : t.toExponential(0)}</text></g>
          ))}
          <path d={ln(sq)} style={{ fill: 'none', stroke: SQ, strokeWidth: 2 }} data-line="sqrt" />
          <path d={ln(li)} style={{ fill: 'none', stroke: LI, strokeWidth: 2 }} data-line="lin" />
          {opt === 'sgd' && <path d={ln(RK.expected_sgd.sqrt)} style={{ fill: 'none', stroke: SQ, strokeWidth: 1.5, strokeDasharray: '4 3' }} data-line="mean-sqrt" />}
          {opt === 'sgd' && <path d={ln(RK.expected_sgd.lin)} style={{ fill: 'none', stroke: LI, strokeWidth: 1.5, strokeDasharray: '4 3' }} data-line="mean-lin" />}
          {RK.r.map((r, i) => <circle key={`s${r}`} cx={xr(r)} cy={yv(sq[i])} r={3} style={{ fill: SQ }} />)}
          {RK.r.map((r, i) => <circle key={`l${r}`} cx={xr(r)} cy={yv(li[i])} r={3} style={{ fill: LI }} />)}
        </g>
      </svg>
      <Key items={[{ label: 's = α/√r', color: SQ }, { label: 's = α/r', color: LI },
        ...(opt === 'sgd' ? [{ label: 'the mean step (Theorem 1(c))', color: 'var(--color-text-secondary)', dash: true }] : [])]} />
      <Readouts>
        <Readout label={`${OPT_LABEL[opt]}, α/√r: r = 1 → 64`} value={`${g3(sq[0])} → ${g3(sq[sq.length - 1])}`} id="rank-sqrt" emphasis />
        <Readout label={`${OPT_LABEL[opt]}, α/r: r = 1 → 64`} value={`${g3(li[0])} → ${g3(li[li.length - 1])}`} id="rank-lin" />
        {opt === 'sgd' && <Readout label="mean step, α/√r (every r)" value={RK.expected_sgd.sqrt[0].toFixed(4)} id="rank-mean" />}
      </Readouts>
      <Row style={{ marginTop: '0.9rem' }}>
        <Label>after 3 epochs</Label>
        {METRIC.map((m) => <Pill key={m.key} active={metric === m.key} onClick={() => setMetric(m.key)} data={{ 'data-metric': m.key }}>{m.label}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${W} ${H + 46}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="trained"
        aria-label={`${metric} after training: alpha over root r ${tr.sqrt.map((t) => f3(mv(t))).join(', ')}; alpha over r ${tr.lin.map((t) => f3(mv(t))).join(', ')}`}>
        <text x={8} y={12} style={titleText}>{METRIC.find((m) => m.key === metric)!.label}, LoRA on Wq, Wv at r = 1, 4, 16, 64</text>
        <g transform={`translate(${LW},22)`}>
          {[1, 4, 16, 64].map((r) => <text key={r} x={xr(r)} y={H + 14} textAnchor="middle" style={axisText}>{r}</text>)}
          <text x={IW / 2} y={H + 24} textAnchor="middle" style={axisText}>rank r</text>
          {y2.ticks(4).map((t) => (
            <g key={t}><line x1={0} x2={IW} y1={y2(t)} y2={y2(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y2(t) + 3} textAnchor="end" style={axisText}>{metric === 'acc' ? t.toFixed(2) : String(t)}</text></g>
          ))}
          {metric === 'acc' && (['full', 'lpft'] as const).map((k) => (
            <g key={k} data-band={k}>
              <rect x={0} width={IW} y={y2(RK[k].max)} height={Math.max(1, y2(RK[k].min) - y2(RK[k].max))}
                style={{ fill: k === 'full' ? 'var(--color-muted-bg)' : 'var(--color-badge-amber-bg)', opacity: 0.8 }} />
              <text x={IW - 4} y={y2(RK[k].max) - 3} textAnchor="end" style={axisText}>
                {k === 'full' ? 'full fine-tuning' : 'LP-FT'}, 3 seeds: {f3(RK[k].min)}–{f3(RK[k].max)}
              </text>
            </g>
          ))}
          {metric === 'acc' && <g data-band="probe"><line x1={0} x2={IW} y1={y2(RK.probe)} y2={y2(RK.probe)} style={{ stroke: 'var(--color-text-secondary)', strokeDasharray: '3 2' }} />
            <text x={4} y={y2(RK.probe) - 3} style={axisText}>probe {f3(RK.probe)}</text></g>}
          <path d={ln2(tr.sqrt)} style={{ fill: 'none', stroke: SQ, strokeWidth: 2 }} />
          <path d={ln2(tr.lin)} style={{ fill: 'none', stroke: LI, strokeWidth: 2 }} />
          {(['sqrt', 'lin'] as const).flatMap((sc) => tr[sc].map((t) => (
            <circle key={`${sc}${t.r}`} cx={xr(t.r)} cy={y2(mv(t))} r={3.5} style={{ fill: sc === 'sqrt' ? SQ : LI }} data-point={`${sc}-${t.r}`} />
          )))}
        </g>
      </svg>
      <Readouts>
        {(['sqrt', 'lin'] as const).map((sc) => (
          <Readout key={sc} label={`${sc === 'sqrt' ? 'α/√r' : 'α/r'}, r = 1, 4, 16, 64`} value={tr[sc].map((t) => metric === 'norm' ? t.norm.toFixed(1) : f3(mv(t))).join(' · ')} id={`trained-${sc}`} />
        ))}
      </Readouts>
      <Note>
        Theorem 1: from B = 0 the first step is the full step pushed through A₀A₀ᵀ. Under Adam its size is pinned near
        ηs√(rm), so only α/√r keeps it level; under SGD the mean step is level under α/√r, but one draw sits a factor
        √((k+1)/r) away until r nears k = 64. Trained at one rate (Adam, 10⁻²), the scale sets the update size, not a
        winner.
      </Note>
    </figure>
  );
}

// ----- C. What full fine-tuning changed -----

function SpectrumView() {
  const [mi, setMi] = useState(0);
  const [ti, setTi] = useState(3);
  const m = SP.matrices[mi];
  const LW = 44, IW = W - LW - 20, H = 170;
  const xb = d3.scaleBand<number>().domain(d3.range(m.sv.length)).range([0, IW]).padding(0.15);
  const yb = d3.scaleLinear().domain([0, Math.max(m.sv[0], m.edge) * 1.1]).range([H, 0]);
  const T = SP.trunc;
  const xt = d3.scaleLinear().domain([0, T.r.length - 1]).range([0, IW]);
  const yt = d3.scaleLinear().domain([0.4, 1]).range([110, 0]);
  return (
    <figure style={{ margin: 0 }} data-view="spectrum">
      <Caption>Full fine-tuning&apos;s update to each matrix: a few directions above the noise</Caption>
      <Row>
        {SP.matrices.map((x, i) => (
          <Pill key={`${x.block}${x.matrix}`} active={mi === i} onClick={() => setMi(i)} data={{ 'data-matrix': `${x.block}-${x.matrix}` }}>
            {`block ${x.block + 1} ${x.matrix}`}
          </Pill>
        ))}
      </Row>
      <svg viewBox={`0 0 ${W} ${H + 40}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="sv"
        aria-label={`singular values of the update to block ${m.block + 1} ${m.matrix}: largest ${f3(m.sv[0])}, Gaussian edge ${f3(m.edge)}`}>
        <text x={8} y={12} style={titleText}>singular values of ΔW ({m.shape[0]} × {m.shape[1]}), largest first</text>
        <g transform={`translate(${LW},22)`}>
          {yb.ticks(4).map((t) => (
            <g key={t}><line x1={0} x2={IW} y1={yb(t)} y2={yb(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={yb(t) + 3} textAnchor="end" style={axisText}>{t}</text></g>
          ))}
          {m.sv.map((v, i) => (
            <rect key={i} x={xb(i)} width={xb.bandwidth()} y={yb(v)} height={H - yb(v)}
              style={{ fill: i < 4 ? 'var(--color-accent)' : 'var(--color-text-secondary)' }} />
          ))}
          <line x1={0} x2={IW} y1={yb(m.edge)} y2={yb(m.edge)} style={{ stroke: 'var(--color-badge-blue-text)', strokeWidth: 1.5, strokeDasharray: '5 3' }} data-line="edge" />
          <text x={IW - 4} y={yb(m.edge) - 4} textAnchor="end" style={{ ...axisText, fill: 'var(--color-badge-blue-text)' }}>same-norm Gaussian edge {f3(m.edge)}</text>
          <text x={IW / 2} y={H + 14} textAnchor="middle" style={axisText}>index (the first 4 in color)</text>
        </g>
      </svg>
      <Readouts>
        <Readout label="largest singular value" value={f3(m.sv[0])} id="sv-top" emphasis />
        <Readout label="stable rank ‖ΔW‖²/σ₁²" value={f3(m.stable_rank)} id="sv-stable" />
        <Readout label="energy in the top 4 / top 16" value={`${f3(m.top4)} / ${f3(m.top16)}`} id="sv-energy" />
      </Readouts>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', margin: '0.9rem 0 0.45rem' }}>
        <label htmlFor="lora-lab-trunc" style={{ minWidth: 0 }}>truncate every block update to rank</label>
        <input id="lora-lab-trunc" type="range" min={0} max={T.r.length - 1} step={1} value={ti} data-control="trunc"
          onChange={(e) => setTi(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={`rank ${T.r[ti]}`} />
        <span data-readout="trunc-r" style={{ fontFamily: mono, minWidth: 0 }}>r = {T.r[ti]}</span>
      </div>
      <svg viewBox={`0 0 ${W} 150`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="trunc"
        aria-label={`test accuracy of truncated merges by rank: ${T.r.map((r, i) => `${r}: ${f3(T.acc[i])}`).join(', ')}`}>
        <g transform={`translate(${LW},14)`}>
          {[0.4, 0.6, 0.8, 1].map((t) => (
            <g key={t}><line x1={0} x2={IW} y1={yt(t)} y2={yt(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={yt(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(1)}</text></g>
          ))}
          <line x1={0} x2={IW} y1={yt(T.full)} y2={yt(T.full)} style={{ stroke: 'var(--color-text)', strokeDasharray: '3 2' }} />
          <path d={d3.line<number>().x((_, i) => xt(i)).y((d) => yt(d))(T.acc) ?? ''} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
          {T.acc.map((a, i) => <circle key={i} cx={xt(i)} cy={yt(a)} r={i === ti ? 5 : 3} style={{ fill: 'var(--color-accent)' }} />)}
          {T.r.map((r, i) => <text key={r} x={xt(i)} y={128} textAnchor="middle" style={axisText}>{r}</text>)}
        </g>
      </svg>
      <Readouts>
        <Readout label={`test accuracy at rank ${T.r[ti]}`} value={f3(T.acc[ti])} id="trunc-acc" emphasis />
        <Readout label="the full update" value={f3(T.full)} id="trunc-full" />
      </Readouts>
      <Note>
        Proposition 2 (Eckart–Young–Mirsky): the truncated SVD is the best rank-r stand-in for each update. Rank 0 keeps
        only the fine-tuned embeddings and head; rank 4 already reads within 0.02 of the full update, though LoRA trained
        at rank 4 does not get there.
      </Note>
    </figure>
  );
}

// ----- D. Learning vs. forgetting -----

const GROUPS = [
  { key: 'probe', label: 'probe', color: 'var(--color-text-secondary)' },
  { key: 'full', label: 'full fine-tuning', color: 'var(--color-accent)' },
  { key: 'lpft', label: 'LP-FT', color: 'var(--color-accent-secondary)' },
  { key: 'lora', label: 'LoRA', color: 'var(--color-badge-blue-text)' },
  { key: 'l2sp', label: 'L2-SP', color: 'var(--color-badge-green-text)' },
  { key: 'random', label: 'never pretrained', color: 'var(--color-badge-red-text)' },
  { key: 'sketch', label: 'random subspace', color: 'var(--color-badge-amber-text)' },
] as const;
const group = (p: Point) => (p.label.startsWith('sketch') ? 'sketch' : p.label === 'random full' ? 'random' : p.method);

function FrontierView() {
  const [sel, setSel] = useState('lpft s0');
  const LW = 50, IW = W - LW - 20, H = 260;
  const x = d3.scaleLinear().domain([1.3, 7.2]).range([0, IW]);
  const y = d3.scaleLinear().domain([0.7, 1]).range([H, 0]);
  const p = FR.points.find((q) => q.label === sel)!;
  const n = FR.ntk;
  return (
    <figure style={{ margin: 0 }} data-view="frontier">
      <Caption>Every run: what it learned against what it forgot</Caption>
      <svg viewBox={`0 0 ${W} ${H + 46}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="scatter"
        aria-label={`test accuracy against held-out masked-LM loss for ${FR.points.length} runs; selected ${p.label}: accuracy ${f3(p.acc)}, loss ${f3(p.mlm)}`}>
        <text x={8} y={12} style={titleText}>test accuracy on aspect rows against held-out masked-LM loss (click a point)</text>
        <g transform={`translate(${LW},22)`}>
          {y.ticks(6).map((t) => (
            <g key={t}><line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(2)}</text></g>
          ))}
          {x.ticks(6).map((t) => <text key={t} x={x(t)} y={H + 14} textAnchor="middle" style={axisText}>{t}</text>)}
          <text x={IW / 2} y={H + 26} textAnchor="middle" style={axisText}>masked-LM loss after fine-tuning (forgetting →)</text>
          <line x1={x(FR.mlm_pretrained)} x2={x(FR.mlm_pretrained)} y1={0} y2={H} style={{ stroke: 'var(--color-text)', strokeDasharray: '3 2' }} />
          <text x={x(FR.mlm_pretrained) + 4} y={H - 4} style={axisText}>pretrained {f3(FR.mlm_pretrained)}</text>
          {FR.points.filter((q) => q.acc >= 0.7).map((q) => {
            const g = GROUPS.find((gg) => gg.key === group(q))!;
            const on = q.label === sel;
            return (
              <circle key={q.label} cx={x(q.mlm)} cy={y(q.acc)} r={on ? 7 : 4.5} onClick={() => setSel(q.label)} data-point={q.label}
                style={{ fill: q.augment ? 'var(--color-bg)' : g.color, stroke: on ? 'var(--color-text)' : g.color, strokeWidth: on ? 2 : 1.5, cursor: 'pointer' }}>
                <title>{`${q.label}: accuracy ${f3(q.acc)}, masked-LM loss ${f3(q.mlm)}`}</title>
              </circle>
            );
          })}
        </g>
      </svg>
      <Key items={[...GROUPS.map((g) => ({ label: g.label, color: g.color })), { label: 'with counterfactual rows', color: 'var(--color-text-secondary)', hollow: true }]} />
      <Row style={{ marginTop: '0.5rem' }}>
        <Label>select</Label>
        {['probe', 'full s0', 'lpft s0', 'lora s0', 'lora slow', 'l2sp 0.1', 'random full'].map((l) => (
          <Pill key={l} active={sel === l} onClick={() => setSel(l)} data={{ 'data-select': l }}>{l}</Pill>
        ))}
      </Row>
      <Readouts>
        <Readout label="run" value={p.label} id="frontier-run" />
        <Readout label="test accuracy" value={f3(p.acc)} id="frontier-acc" emphasis />
        <Readout label="masked-LM loss after" value={f3(p.mlm)} id="frontier-mlm" />
      </Readouts>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 2rem', marginTop: '1rem', fontFamily: sans, fontSize: '0.78rem' }}>
        <div data-panel="d90" style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>Random subspaces (Proposition 4)</div>
          {(['pretrained', 'random'] as const).map((k) => (
            <div key={k} style={{ marginBottom: '0.2rem' }}>
              {k === 'pretrained' ? 'pretrained' : 'never pretrained'}: D = {FR.d90[k].D.join(', ')} →{' '}
              <span style={{ fontFamily: mono }}>{FR.d90[k].acc.map(f3).join(' · ')}</span>; 90% of full = {f3(FR.d90[k].threshold)}
              {FR.d90[k].d90 === null ? ' (not reached)' : ''}
            </div>
          ))}
        </div>
        <div data-panel="ntk" style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>The kernel (Proposition 5), 120 rows from the probe&apos;s head</div>
          <div>start: accuracy {f3(n.acc_start)}, loss {String(n.loss_start)}; kernel GD: {f3(n.acc_kernel)}, loss {String(n.loss_kernel)}</div>
          {(['gd', 'adam'] as const).map((k) => (
            <div key={k}>LoRA {k === 'gd' ? 'GD' : 'Adam'}: accuracy {f3(n[k].acc)}, loss {String(n[k].loss)}, agrees with the kernel on {f3(n[k].agree)}, kernel moved {f3(n[k].kernel_move)}</div>
          ))}
        </div>
      </div>
      <Note>
        Up is learning, left is remembering. LP-FT sits up and left of every LoRA run (Kumar and colleagues&apos; probe-first
        start); L2-SP buys retention with accuracy; LoRA, confined to two matrix types at a high rate, forgets most among
        the pretrained runs.
      </Note>
    </figure>
  );
}

// ----- E. Reading negation -----

const POS = ['not ADJ₊ (−)', 'not ADJ₋ (+)', 'ADJ₊ (+)', 'ADJ₋ (−)'];

function NegationView() {
  const [ri, setRi] = useState(1);
  const r = NG.readers[ri];
  const LW = 150, IW = W - LW - 20, RH = 24;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  const tx = d3.scaleLinear().domain([0.5, 1]).range([0, W - 230 - 20]);
  return (
    <figure style={{ margin: 0 }} data-view="negation">
      <Caption>Reading negation, and the tone it adds up to</Caption>
      <Row>
        {NG.readers.map((rd, i) => <Pill key={rd.label} active={ri === i} onClick={() => setRi(i)} data={{ 'data-reader': rd.label }}>{rd.label}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${W} ${NG.constructions.length * RH + 30}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="constructions"
        aria-label={`${r.label}: accuracy by construction ${NG.constructions.map((c, i) => `${c} ${f3(r.by[i])}`).join(', ')}`}>
        <text x={8} y={14} style={titleText}>share of test aspects read correctly, by construction ({r.label}, all {f3(r.acc)})</text>
        <g transform={`translate(${LW},24)`}>
          {NG.constructions.map((c, i) => {
            const a = r.by[i];
            return (
              <g key={c} transform={`translate(0,${i * RH})`} data-row={c}>
                <text x={-8} y={14} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{c}</text>
                <rect x={0} y={3} width={IW} height={16} style={{ fill: 'var(--color-muted-bg)' }} />
                <rect x={0} y={3} width={x(a)} height={16} style={{ fill: a < 0.8 ? 'var(--color-badge-red-text)' : 'var(--color-accent)' }} />
                <text x={a > 0.5 ? x(a) - 5 : x(a) + 5} y={15} textAnchor={a > 0.5 ? 'end' : 'start'}
                  style={{ ...axisText, fill: a > 0.5 ? 'var(--color-bg)' : 'var(--color-text)' }}>{f3(a)} · n = {NG.n[i]}</text>
              </g>
            );
          })}
        </g>
      </svg>
      <div style={{ fontFamily: sans, fontSize: '0.8rem', fontWeight: 600, margin: '0.9rem 0 0.4rem' }}>
        Negation quadruples: share of each sentence read correctly (2,000 sampled quadruples)
      </div>
      <div data-panel="quads" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) repeat(4, minmax(0, 1fr))', gap: '0.2rem 0.4rem', fontFamily: sans, fontSize: '0.75rem' }}>
        <span />
        {POS.map((q) => <span key={q} style={{ color: 'var(--color-text-secondary)', overflowWrap: 'anywhere' }}>{q}</span>)}
        {(['full', 'augmented'] as const).flatMap((k) => NG.quads[k].map((row, s) => [
          <span key={`${k}${s}`} style={{ overflowWrap: 'anywhere' }}>{k === 'full' ? 'full FT' : 'with counterfactuals'}, seed {s}</span>,
          ...row.map((v, j) => (
            <span key={`${k}${s}${j}`} data-quad={`${k}-${s}-${j}`} style={{ fontFamily: mono, color: v < 0.5 ? 'var(--color-badge-red-text)' : 'var(--color-text)' }}>{f3(v)}</span>
          )),
        ]))}
      </div>
      <svg viewBox={`0 0 ${W} ${NG.tone.length * 26 + 50}`} style={{ width: '100%', height: 'auto', marginTop: '1rem' }} role="img" data-panel="tone"
        aria-label={`document tone accuracy: ${NG.tone.map((b) => `${b.label} ${f3(b.acc)}`).join(', ')}; ceiling ${f3(NG.ceiling)}`}>
        <text x={8} y={14} style={titleText}>three-class tone accuracy on 1,500 test filings (axis from 0.5)</text>
        <g transform="translate(230,26)">
          {NG.tone.map((b, i) => (
            <g key={b.label} transform={`translate(0,${i * 26})`} data-bar={b.label}>
              <text x={-8} y={15} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{b.label}</text>
              <rect x={0} y={4} width={tx(b.acc)} height={16} style={{ fill: b.label.includes('aggregated') ? 'var(--color-accent)' : 'var(--color-text-secondary)' }} />
              <text x={tx(b.acc) - 5} y={16} textAnchor="end" style={{ ...axisText, fill: 'var(--color-bg)' }}>{f3(b.acc)}</text>
            </g>
          ))}
          <line x1={tx(NG.ceiling)} x2={tx(NG.ceiling)} y1={-4} y2={NG.tone.length * 26} style={{ stroke: 'var(--color-text)', strokeWidth: 1.5, strokeDasharray: '3 2' }} />
          <text x={tx(NG.ceiling)} y={NG.tone.length * 26 + 14} textAnchor="middle" style={axisText}>Bayes ceiling {f3(NG.ceiling)}</text>
        </g>
      </svg>
      <Note>
        Without the counterfactual rows the encoder reads both negated forms and calls the deleted-&quot;not&quot; frame
        neutral: training shows that frame only with &quot;not&quot;. With them every seed reads all four. Aggregated by
        Topic 19&apos;s Bayes rule, the aspect reader recovers the filings&apos; tone; fine-tuned on document labels, it does not.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'scalar', label: 'One weight, two factors' },
  { key: 'rank', label: 'Rank and scaling' },
  { key: 'spectrum', label: 'What full fine-tuning changed' },
  { key: 'frontier', label: 'Learning vs. forgetting' },
  { key: 'negation', label: 'Reading negation' },
] as const;

export default function LowRankAdaptationLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('scalar');
  return (
    <div data-lab="low-rank-adaptation" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'scalar' && <ScalarView />}
      {view === 'rank' && <RankView />}
      {view === 'spectrum' && <SpectrumView />}
      {view === 'frontier' && <FrontierView />}
      {view === 'negation' && <NegationView />}
    </div>
  );
}

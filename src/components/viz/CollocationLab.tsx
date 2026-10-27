import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/collocations-keyword-extraction/collocations_keyword_extraction.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_thm1_g2_is_2n_kl_and_2n_mutual_information, test_prop2_hapax_closed_forms,
// test_prop3_perfect_pairs_invert_pmi, test_prop4_exact_sizes_far_from_nominal_at_small_e,
// test_prop5_selection_makes_every_sighting_significant, test_observed_only_bh_rejects_nulls_all_pairs_bh_does_not,
// test_g2_t_fisher_rank_fixed_phrases_pmi_ranks_nulls, test_keyness_finds_planted_sector_nouns,
// test_textrank_tracks_frequency, test_thm4_power_iteration_contracts_by_d).
// Change the .py, regenerate, never the reverse. Positions the views only draw (a projected point, a line, the
// dᵗ guide) are computed here from payload values; every printed number is a payload value.
import payload from './data/collocation-lab.json';

/**
 * Collocation Laboratory — five figures for the collocations-keyword-extraction topic.
 *   A. "Table geometry": the tetrahedron of 2×2 tables, the independence saddle, and the segment of tables with
 *      given margins; a slider moves the joint count (Definitions 1–3, Theorem 1, Proposition 1).
 *   B. "Rare pairs": the scores of perfect pairs seen k times (Propositions 2–3).
 *   C. "Test size": exact sizes of X², G², and Fisher's test against the expected count, and the selection curve
 *      (Lemma 1, Proposition 4, Proposition 5), with the finance stratum's counts.
 *   D. "Finance bigrams": precision at k of every score against the generator's answer key, and each score's top 15.
 *   E. "Keywords": keyness and TextRank by sector, across the damping factor, and power iteration's steps (Theorem 4,
 *      Proposition 7).
 */

type Preset = {
  n: number; r: number; c: number; expected: number; a: number[]; xyz: [number, number, number][];
  p: (number | null)[]; pmi: (number | null)[]; t: (number | null)[]; x2: (number | null)[]; g2: (number | null)[];
};
const GEO = payload.geometry as unknown as { presets: Preset[]; surface: [number, number, number][][] };
const RARE = payload.rare as unknown as {
  n: number; k: number[]; pmi: number[]; t: number[]; x2: number[]; g2: number[]; neglog_p: number[];
  hapax: { pmi: number; x2: number; g2: number; t: number; p_exact: number };
};
const SIZE = payload.size as unknown as { n: number; r: number[]; expected: number[]; x2: number[]; g2: number[]; fisher: number[] };
type Pair = [number, number];
const SEL = payload.selection as unknown as {
  n: number; r: number[]; expected: number[]; p_seen: number[]; p_reject: number[]; p_reject_given_seen: number[];
  corpus: { types: number; m_all: number; seen_once: number; observed: Record<'null' | 'topical', Pair>; all: Record<'null' | 'topical', Pair>; nominal: Record<'null' | 'topical', number> };
};
const MEASURES = ['pmi', 't', 'x2', 'g2', 'fisher', 'freq'] as const;
type Measure = (typeof MEASURES)[number];
type Cls = 'fixed' | 'null' | 'topical' | 'boundary';
const BIG = payload.bigrams as unknown as {
  n: number; types: number; classes: Record<Cls, number>; precision: Record<Measure, number[]>; nulls: Record<Measure, number[]>;
  top: Record<Measure, [string, Cls, number][]>; top_block: Record<Measure, number>;
};
const SECTORS = ['industrials', 'technology', 'banking', 'energy', 'retail', 'healthcare'] as const;
type Sector = (typeof SECTORS)[number];
type TR = { precision: number; spearman: number; iterations: number };
const KW = payload.keywords as unknown as Record<Sector, {
  docs: number; keyness_top: string[]; keyness: { g2: number; log_ratio: number; freq: number };
  textrank: Record<'weighted' | 'binary', Record<string, TR>>; freq_after_stop: number; nodes: number;
}>;
const CONV = payload.convergence as unknown as { sector: string; nodes: number; every: number; steps: Record<string, number[]> };
const DS = ['0.5', '0.85', '0.95', '0.99'] as const;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number | null) => (v === null ? '—' : v.toFixed(3));
const f2 = (v: number) => v.toFixed(2);
const int = (v: number) => v.toLocaleString('en-US');
/** A probability that may span decades: the payload keeps 4 significant figures; show them. */
const prob = (v: number | null) => (v === null ? '—' : v >= 0.001 ? `${v}` : v.toExponential(3));

const MEASURE_NAMES: Record<Measure, string> = { pmi: 'PMI', t: 't-score', x2: 'X²', g2: 'G²', fisher: 'Fisher', freq: 'frequency' };
const MEASURE_COLORS: Record<Measure, string> = {
  pmi: 'var(--color-badge-red-text)', t: 'var(--color-badge-blue-text)', x2: 'var(--color-badge-amber-text)',
  g2: 'var(--color-accent)', fisher: 'var(--color-accent-secondary)', freq: 'var(--color-text-secondary)',
};
const CLASS_COLORS: Record<Cls, string> = {
  fixed: 'var(--color-badge-green-text)', null: 'var(--color-badge-red-text)', topical: 'var(--color-badge-amber-text)',
  boundary: 'var(--color-text-secondary)',
};

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

function Sub({ children }: { children: ReactNode }) {
  return <div style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.7rem 0 0.3rem' }}>{children}</div>;
}

function Slider({ label, min, max, value, onChange, id }: { label: string; min: number; max: number; value: number; onChange: (v: number) => void; id: string }) {
  return (
    <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem', fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginBottom: '0.45rem' }}>
      {label}
      <input type="range" min={min} max={max} step={1} value={value} data-slider={id} aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0, maxWidth: '18rem', accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

function Key({ items }: { items: { label: string; color: string; dash?: boolean; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: s.dot ? '0.55rem' : '0.9rem', height: s.dash ? 0 : s.dot ? '0.55rem' : '0.2rem',
            borderRadius: s.dot ? '50%' : undefined, borderTop: s.dash ? `2px dashed ${s.color}` : undefined,
            background: s.dash ? undefined : s.color,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** A table that scrolls inside its own box on a narrow screen instead of widening the page. */
function Table({ head, rows, id }: { head: string[]; rows: { cells: ReactNode[]; key: string; strong?: boolean }[]; id: string }) {
  const cell: CSSProperties = { padding: '0.2rem 0.5rem', textAlign: 'right', whiteSpace: 'nowrap', borderBottom: '1px solid var(--color-border)' };
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <table data-table={id} style={{ fontFamily: mono, fontSize: '0.75rem', borderCollapse: 'collapse', margin: '0.3rem 0' }}>
        <thead>
          <tr>{head.map((h, i) => <th key={h} style={{ ...cell, fontFamily: sans, fontWeight: 600, textAlign: i === 0 ? 'left' : 'right' }}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-row={r.key}>
              {r.cells.map((c, i) => (
                <td key={i} style={{ ...cell, textAlign: i === 0 ? 'left' : 'right', fontFamily: i === 0 ? sans : mono, fontWeight: r.strong ? 600 : 400 }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A small line chart frame: log or linear x, log or linear y, with gridlines at the given ticks. */
function Frame({ W, H, m, x, y, xTicks, yTicks, xFmt, yFmt, xLabel, children, panel, aria }: {
  W: number; H: number; m: { l: number; r: number; t: number; b: number };
  x: (v: number) => number; y: (v: number) => number; xTicks: number[]; yTicks: number[];
  xFmt: (v: number) => string; yFmt: (v: number) => string; xLabel: string; children: ReactNode; panel: string; aria: string;
}) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '30rem', height: 'auto', display: 'block' }} role="img" data-panel={panel} aria-label={aria}>
      <defs><clipPath id={`cl-${panel}`}><rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} /></clipPath></defs>
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke="var(--color-border)" strokeWidth={0.5} />
          <text x={m.l - 4} y={y(v) + 3} textAnchor="end" style={axisText}>{yFmt(v)}</text>
        </g>
      ))}
      {xTicks.map((v) => <text key={`x${v}`} x={x(v)} y={H - m.b + 13} textAnchor="middle" style={axisText}>{xFmt(v)}</text>)}
      <text x={(m.l + W - m.r) / 2} y={H - 2} textAnchor="middle" style={axisText}>{xLabel}</text>
      <g clipPath={`url(#cl-${panel})`}>{children}</g>
    </svg>
  );
}

const sci = (v: number) => (v >= 1 ? `${v}` : `${v}`.replace(/^0\./, '.'));
const pow10 = (v: number) => {
  const e = Math.round(Math.log10(v));
  return e === 0 ? '1' : e === 1 ? '10' : `1e${e}`;
};

// ---------------------------------------------------------------------------------------------------------------------
// A. Table geometry.
// ---------------------------------------------------------------------------------------------------------------------

function GeometryView() {
  const [pi, setPi] = useState(0);
  const preset = GEO.presets[pi];
  const [ai, setAi] = useState(Math.round(preset.expected));
  const [yaw, setYaw] = useState(30);
  const i = Math.min(ai, preset.a.length - 1);
  const th = (yaw * Math.PI) / 180;
  const ph = (20 * Math.PI) / 180;
  // Orthographic projection: rotate about the vertical (z) axis by yaw, tilt by a fixed pitch.
  const proj = ([x, y, z]: [number, number, number]): [number, number] => {
    const xr = x * Math.cos(th) - y * Math.sin(th);
    const yr = x * Math.sin(th) + y * Math.cos(th);
    return [xr, z * Math.cos(ph) - yr * Math.sin(ph)];
  };
  const S = 320;
  const sc = d3.scaleLinear().domain([-1.75, 1.75]).range([0, S]);
  const P = (v: [number, number, number]) => { const [u, w] = proj(v); return [sc(u), S - sc(w)] as [number, number]; };
  const verts: [number, number, number][] = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
  const vNames = ['p₁₁', 'p₁₂', 'p₂₁', 'p₂₂'];
  const edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const surf = GEO.surface;
  const seg = preset.xyz.map(P);
  const lo = preset.a[0];
  const ef = preset.expected - lo; // fractional index of the crossing a = E along the (linear) segment
  const j0 = Math.floor(ef);
  const cross: [number, number, number] = j0 + 1 < preset.xyz.length
    ? preset.xyz[j0].map((v, k) => v + (ef - j0) * (preset.xyz[j0 + 1][k] - v)) as [number, number, number]
    : preset.xyz[j0];
  const pt = P(preset.xyz[i]);
  const cp = P(cross);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>A word pair's table is a point of the tetrahedron; independent tables form the saddle</Caption>
      <Row>
        <Label>margins (N, r, c)</Label>
        {GEO.presets.map((q, k) => (
          <Pill key={k} active={pi === k} onClick={() => { setPi(k); setAi(Math.round(q.expected)); }} data={{ 'data-preset': `${k}` }}>
            {`(${int(q.n)}, ${q.r}, ${q.c})`}
          </Pill>
        ))}
      </Row>
      <Slider label={`joint count a = ${preset.a[i]}`} min={0} max={preset.a.length - 1} value={i} onChange={setAi} id="a" />
      <Slider label="rotate" min={0} max={359} value={yaw} onChange={setYaw} id="yaw" />
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: '24rem', height: 'auto', display: 'block' }} role="img" data-panel="tetra"
        aria-label="The tetrahedron of 2 by 2 tables, the independence saddle, and the tables with the chosen margins">
        {surf.map((row, k) => (
          <polyline key={`u${k}`} points={row.map(P).map(([u, v]) => `${u},${v}`).join(' ')} fill="none" stroke="var(--color-accent-secondary)" strokeWidth={0.6} strokeOpacity={0.55} />
        ))}
        {surf[0].map((_, k) => (
          <polyline key={`v${k}`} points={surf.map((row) => P(row[k])).map(([u, v]) => `${u},${v}`).join(' ')} fill="none" stroke="var(--color-accent-secondary)" strokeWidth={0.6} strokeOpacity={0.55} />
        ))}
        {edges.map(([p, q]) => {
          const [x1, y1] = P(verts[p]); const [x2, y2] = P(verts[q]);
          return <line key={`${p}${q}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-text-secondary)" strokeWidth={1} />;
        })}
        {verts.map((v, k) => { const [u, w] = P(v); return <text key={k} x={u} y={w - 5} textAnchor="middle" style={axisText}>{vNames[k]}</text>; })}
        <polyline points={seg.map(([u, v]) => `${u},${v}`).join(' ')} fill="none" stroke="var(--color-accent)" strokeWidth={1.6} />
        <circle cx={cp[0]} cy={cp[1]} r={4} fill="none" stroke="var(--color-accent-secondary)" strokeWidth={1.5} />
        <circle cx={pt[0]} cy={pt[1]} r={4.5} fill="var(--color-accent)" />
      </svg>
      <Key items={[
        { label: 'independence saddle S', color: 'var(--color-accent-secondary)' },
        { label: 'tables with these margins', color: 'var(--color-accent)' },
        { label: 'the table at a', color: 'var(--color-accent)', dot: true },
        { label: 'crossing S at a = E', color: 'var(--color-accent-secondary)', dot: true },
      ]} />
      <Readouts>
        <Readout label="expected count E" value={f3(preset.expected)} id="E" />
        <Readout label="PMI" value={preset.pmi[i] === null ? '−∞' : f3(preset.pmi[i])} id="pmi" />
        <Readout label="t-score" value={f3(preset.t[i])} id="t" />
        <Readout label="X²" value={f3(preset.x2[i])} id="x2" />
        <Readout label="G²" value={f3(preset.g2[i])} id="g2" emphasis />
        <Readout label="exact p, P(A ≥ a)" value={prob(preset.p[i])} id="p" />
      </Readouts>
      <Note>Dividing a table by N makes it a point of the tetrahedron whose corners are the four cells. With the margins fixed, the tables form a straight segment that crosses the saddle once, at a = E, where every score is zero. G² is 2N times the KL divergence from the point to that crossing (Theorem 1), and X² is N times the χ² divergence (Proposition 1).</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. Rare pairs.
// ---------------------------------------------------------------------------------------------------------------------

function RareView() {
  const [ki, setKi] = useState(0);
  const W = 340, H = 220, m = { l: 44, r: 10, t: 10, b: 30 };
  const x = d3.scaleLinear().domain([1, 20]).range([m.l, W - m.r]);
  const y = d3.scaleLog().domain([0.5, 2e5]).range([H - m.b, m.t]);
  const series: { key: keyof typeof RARE; label: string; color: string }[] = [
    { key: 'x2', label: 'X² (flat at N)', color: MEASURE_COLORS.x2 },
    { key: 'g2', label: 'G² = 2N·H(k/N)', color: MEASURE_COLORS.g2 },
    { key: 'neglog_p', label: '−log p (exact)', color: MEASURE_COLORS.fisher },
    { key: 'pmi', label: 'PMI = log(N/k)', color: MEASURE_COLORS.pmi },
    { key: 't', label: 't-score', color: MEASURE_COLORS.t },
  ];
  const line = (vals: number[]) => d3.line<number>().x((_, j) => x(RARE.k[j])).y((v) => y(v))(vals) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Pairs seen k times and never apart, at N = ${int(RARE.n)}: PMI falls as the evidence rises`}</Caption>
      <Slider label={`k = ${RARE.k[ki]}`} min={0} max={RARE.k.length - 1} value={ki} onChange={setKi} id="k" />
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[1, 5, 10, 15, 20]} yTicks={[1, 10, 100, 1000, 10000, 100000]}
        xFmt={(v) => `${v}`} yFmt={pow10} xLabel="k (log y axis)" panel="rare" aria="Five scores of perfect pairs against k, on a log scale">
        {series.map((s) => <path key={s.key} d={line(RARE[s.key] as number[])} fill="none" stroke={s.color} strokeWidth={2} />)}
        <line x1={x(RARE.k[ki])} x2={x(RARE.k[ki])} y1={m.t} y2={H - m.b} stroke="var(--color-text-secondary)" strokeWidth={0.7} />
      </Frame>
      <Key items={series.map((s) => ({ label: s.label, color: s.color }))} />
      <Readouts>
        <Readout label="PMI" value={f3(RARE.pmi[ki])} id="pmi" />
        <Readout label="t-score" value={f3(RARE.t[ki])} id="t" />
        <Readout label="X²" value={f3(RARE.x2[ki])} id="x2" />
        <Readout label="G²" value={f3(RARE.g2[ki])} id="g2" emphasis />
        <Readout label="−log p" value={f3(RARE.neglog_p[ki])} id="neglogp" />
      </Readouts>
      <Note>{`At k = 1, the hapax pair: PMI = log N = ${f3(RARE.hapax.pmi)}, the largest PMI of any sighting, and X² = N = ${int(RARE.hapax.x2)}, the largest X² of any table, while G² = ${f3(RARE.hapax.g2)} and the exact p-value is 1/N = ${prob(RARE.hapax.p_exact)} (Proposition 2).`}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Test size and the selection.
// ---------------------------------------------------------------------------------------------------------------------

function SizeView() {
  const [ri, setRi] = useState(41); // r = c = 42: E = 0.1764, where the size of X² peaks (summary: size argmax E x2)
  const [si, setSi] = useState(13); // r = c = 14, the last r at which every sighting rejects
  const W = 340, H = 210, m = { l: 40, r: 10, t: 10, b: 30 };
  const x = d3.scaleLog().domain([1e-4, 20.25]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 0.18]).range([H - m.b, m.t]);
  const line = (xs: number[], vals: number[]) => d3.line<number>().x((_, j) => x(xs[j])).y((v) => y(v))(vals) ?? '';
  const xs = d3.scaleLog().domain([2e-4, 1.6]).range([m.l, W - m.r]);
  const ys = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  const lineS = (vals: number[]) => d3.line<number>().x((_, j) => xs(SEL.expected[j])).y((v) => ys(v))(vals) ?? '';
  const C = SEL.corpus;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Exact sizes at nominal α = 0.05 (N = ${int(SIZE.n)}, r = c), against the expected count E`}</Caption>
      <Slider label={`r = c = ${SIZE.r[ri]}`} min={0} max={SIZE.r.length - 1} value={ri} onChange={setRi} id="r" />
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[1e-4, 1e-3, 0.01, 0.1, 1, 10]} yTicks={[0, 0.05, 0.1, 0.15]}
        xFmt={(v) => sci(v)} yFmt={(v) => v.toFixed(2)} xLabel="E = r²/N (log scale)" panel="size" aria="Exact sizes of three tests against the expected count">
        <line x1={m.l} x2={W - m.r} y1={y(0.05)} y2={y(0.05)} stroke="var(--color-text-secondary)" strokeDasharray="4 3" />
        {(['x2', 'g2', 'fisher'] as const).map((k) => (
          <path key={k} d={line(SIZE.expected, SIZE[k])} fill="none" stroke={MEASURE_COLORS[k]} strokeWidth={1.6} />
        ))}
        <line x1={x(SIZE.expected[ri])} x2={x(SIZE.expected[ri])} y1={m.t} y2={H - m.b} stroke="var(--color-text-secondary)" strokeWidth={0.7} />
      </Frame>
      <Key items={[
        { label: 'X²', color: MEASURE_COLORS.x2 }, { label: 'G²', color: MEASURE_COLORS.g2 },
        { label: 'Fisher (exact)', color: MEASURE_COLORS.fisher }, { label: 'α = 0.05', color: 'var(--color-text-secondary)', dash: true },
      ]} />
      <Readouts>
        <Readout label="E" value={prob(SIZE.expected[ri])} id="size-E" />
        <Readout label="size of X²" value={prob(SIZE.x2[ri])} id="size-x2" />
        <Readout label="size of G²" value={prob(SIZE.g2[ri])} id="size-g2" />
        <Readout label="size of Fisher" value={prob(SIZE.fisher[ri])} id="size-fisher" emphasis />
      </Readouts>
      <Sub>{`Testing only what you saw (Proposition 5): one-sided exact test at N = ${int(SEL.n)}, the finance stratum's size`}</Sub>
      <Slider label={`r = c = ${SEL.r[si]}`} min={0} max={SEL.r.length - 1} value={si} onChange={setSi} id="sel-r" />
      <Frame W={W} H={H} m={m} x={xs} y={ys} xTicks={[1e-3, 0.01, 0.1, 1]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
        xFmt={(v) => sci(v)} yFmt={(v) => v.toFixed(2)} xLabel="E = r²/N (log scale)" panel="selection" aria="Probability of a sighting, of rejection, and of rejection given a sighting">
        <path d={lineS(SEL.p_reject_given_seen)} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
        <path d={lineS(SEL.p_seen)} fill="none" stroke="var(--color-badge-blue-text)" strokeWidth={1.4} />
        <path d={lineS(SEL.p_reject)} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.4} />
        <line x1={xs(SEL.expected[si])} x2={xs(SEL.expected[si])} y1={m.t} y2={H - m.b} stroke="var(--color-text-secondary)" strokeWidth={0.7} />
      </Frame>
      <Key items={[
        { label: 'P(reject | seen)', color: 'var(--color-accent)' }, { label: 'P(seen)', color: 'var(--color-badge-blue-text)' },
        { label: 'P(reject)', color: 'var(--color-text-secondary)' },
      ]} />
      <Readouts>
        <Readout label="E" value={prob(SEL.expected[si])} id="sel-E" />
        <Readout label="P(seen)" value={prob(SEL.p_seen[si])} id="sel-seen" />
        <Readout label="P(reject)" value={prob(SEL.p_reject[si])} id="sel-reject" />
        <Readout label="P(reject | seen)" value={prob(SEL.p_reject_given_seen[si])} id="sel-given" emphasis />
      </Readouts>
      <Sub>{`The finance stratum: ${int(C.types)} observed pairs of adjacent drawn words, ${int(C.m_all)} candidate pairs, ${f3(C.seen_once)} of the observed seen once`}</Sub>
      <Table id="selection" head={['BH at q = 0.05 over', 'null rejected', 'topical rejected']}
        rows={(['observed', 'all'] as const).map((k) => ({
          key: k, strong: k === 'observed',
          cells: [k === 'observed' ? 'observed pairs' : 'all candidate pairs', `${int(C[k].null[1])} of ${int(C[k].null[0])}`, `${int(C[k].topical[1])} of ${int(C[k].topical[0])}`],
        }))} />
      <Note>{`When a pair's chance of being seen is at most α, a sighting is a rejection. Counting only the pairs that were seen, ${f3(C.nominal.null)} of the null pairs reject at the uncorrected level. Counting every candidate pair, with p = 1 for those never seen, nothing does.`}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. Finance bigrams.
// ---------------------------------------------------------------------------------------------------------------------

function BigramsView() {
  const [mi, setMi] = useState<Measure>('g2');
  const [k, setK] = useState(50);
  const W = 340, H = 210, m = { l: 40, r: 10, t: 10, b: 30 };
  const x = d3.scaleLog().domain([1, 500]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  const line = (vals: number[]) => d3.line<number>().x((_, j) => x(j + 1)).y((v) => y(v))(vals) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Every bigram of the finance corpus (${int(BIG.types)} types, N = ${int(BIG.n)}), scored against the generator's answer key`}</Caption>
      <Row>
        <Label>score</Label>
        {MEASURES.map((q) => <Pill key={q} active={mi === q} onClick={() => setMi(q)} data={{ 'data-measure': q }}>{MEASURE_NAMES[q]}</Pill>)}
      </Row>
      <Slider label={`top k = ${k}`} min={1} max={500} value={k} onChange={setK} id="k" />
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[1, 10, 100, 500]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
        xFmt={(v) => `${v}`} yFmt={(v) => v.toFixed(2)} xLabel="k (log scale)" panel="precision" aria="Precision at k against the fixed phrases for each score">
        {MEASURES.filter((q) => q !== mi).map((q) => <path key={q} d={line(BIG.precision[q])} fill="none" stroke={MEASURE_COLORS[q]} strokeWidth={1} strokeOpacity={0.35} />)}
        <path d={line(BIG.nulls[mi])} fill="none" stroke={CLASS_COLORS.null} strokeWidth={1.6} strokeDasharray="4 3" />
        <path d={line(BIG.precision[mi])} fill="none" stroke={MEASURE_COLORS[mi]} strokeWidth={2.2} />
        <line x1={x(k)} x2={x(k)} y1={m.t} y2={H - m.b} stroke="var(--color-text-secondary)" strokeWidth={0.7} />
      </Frame>
      <Key items={[
        { label: `${MEASURE_NAMES[mi]}: precision at k (fixed phrases)`, color: MEASURE_COLORS[mi] },
        { label: `${MEASURE_NAMES[mi]}: null or topical share`, color: CLASS_COLORS.null, dash: true },
        { label: 'other scores (faint)', color: 'var(--color-text-secondary)' },
      ]} />
      <Readouts>
        <Readout label="precision at k" value={f3(BIG.precision[mi][k - 1])} id="precision" emphasis />
        <Readout label="null or topical share" value={f3(BIG.nulls[mi][k - 1])} id="nulls" />
        <Readout label="pairs tied at the top" value={int(BIG.top_block[mi])} id="block" />
      </Readouts>
      <Sub>{`Top 15 by ${MEASURE_NAMES[mi]} (alphabetical within a tie), with each bigram's count`}</Sub>
      <div data-top={mi} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.9rem', fontFamily: mono, fontSize: '0.75rem' }}>
        {BIG.top[mi].map(([g, c, n]) => (
          <span key={g} data-cls={c} style={{ color: CLASS_COLORS[c] }}>{g} <span style={{ color: 'var(--color-text-secondary)' }}>{n}</span></span>
        ))}
      </div>
      <Key items={(['fixed', 'null', 'topical', 'boundary'] as const).map((c) => ({ label: `${c} (${int(BIG.classes[c])} types)`, color: CLASS_COLORS[c], dot: true }))} />
      <Note>Ties are broken at random in expectation: PMI ties its top 97 pairs at log N, all seen once, and X² ties its top 196, every pair whose words never occur apart. Fixed: template text or a name; null: two adjacent independent draws; topical: two adjacent draws from one sector's vocabulary.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// E. Keywords.
// ---------------------------------------------------------------------------------------------------------------------

function KeywordsView() {
  const [s, setS] = useState<Sector>('industrials');
  const [d, setD] = useState<(typeof DS)[number]>('0.85');
  const [g, setG] = useState<'weighted' | 'binary'>('weighted');
  const kw = KW[s];
  const tr = kw.textrank[g][d];
  const W = 340, H = 200, m = { l: 44, r: 10, t: 10, b: 30 };
  const steps = CONV.steps[d];
  const tMax = 400;
  const x = d3.scaleLinear().domain([0, tMax]).range([m.l, W - m.r]);
  const y = d3.scaleLog().domain([1e-12, 2]).range([H - m.b, m.t]);
  const dv = Number(d);
  const line = d3.line<number>().x((_, j) => x(j * CONV.every)).y((v) => y(v))(steps) ?? '';
  const guide = d3.line<number>().x((t) => x(t)).y((t) => y(Math.max(1e-12, steps[0] * dv ** t)))(d3.range(0, tMax + 1, 4)) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Keywords by sector: keyness against the other sectors, TextRank on the sector alone</Caption>
      <Row>
        <Label>sector</Label>
        {SECTORS.map((q) => <Pill key={q} active={s === q} onClick={() => setS(q)} data={{ 'data-sector': q }}>{q}</Pill>)}
      </Row>
      <Sub>{`Top 10 by keyness G² (${kw.docs} documents; entity words removed)`}</Sub>
      <div data-keyness={s} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.8rem', fontFamily: mono, fontSize: '0.78rem' }}>
        {kw.keyness_top.map((w) => <span key={w}>{w}</span>)}
      </div>
      <Row style={{ marginTop: '0.7rem' }}>
        <Label>TextRank damping d</Label>
        {DS.map((q) => <Pill key={q} active={d === q} onClick={() => setD(q)} data={{ 'data-d': q }}>{q}</Pill>)}
      </Row>
      <Row>
        <Label>graph</Label>
        {(['weighted', 'binary'] as const).map((q) => <Pill key={q} active={g === q} onClick={() => setG(q)} data={{ 'data-graph': q }}>{q}</Pill>)}
      </Row>
      <Readouts>
        <Readout label="keyness G², precision at 25" value={f2(kw.keyness.g2)} id="kw-g2" emphasis />
        <Readout label="TextRank, precision at 25" value={f2(tr.precision)} id="tr-precision" />
        <Readout label="frequency, precision at 25" value={f2(kw.freq_after_stop)} id="tr-freq" />
        <Readout label="TextRank's Spearman with frequency" value={f3(tr.spearman)} id="tr-spearman" />
        <Readout label="iterations to 10⁻¹²" value={int(tr.iterations)} id="tr-iter" />
      </Readouts>
      <Sub>{`Power iteration's step sizes in ${CONV.sector} (${int(CONV.nodes)} words, weighted), against Theorem 4's bound dᵗ times the first step`}</Sub>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[0, 100, 200, 300, 400]} yTicks={[1e-12, 1e-9, 1e-6, 1e-3, 1]}
        xFmt={(v) => `${v}`} yFmt={pow10} xLabel="iteration t (clipped at 400 and 10⁻¹²)" panel="convergence" aria="Step sizes of power iteration against the geometric bound">
        <path d={guide} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.2} strokeDasharray="4 3" />
        <path d={line} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
      </Frame>
      <Key items={[{ label: '‖xₜ₊₁ − xₜ‖₁ (every 4th step)', color: 'var(--color-accent)' }, { label: 'dᵗ ‖x₁ − x₀‖₁', color: 'var(--color-text-secondary)', dash: true }]} />
      <Note>Precision is against the nouns the generator planted in the sector. TextRank sees only the sector's own text and ranks close to frequency, more so as d grows (Proposition 7); keyness compares it with the other sectors and finds what is distinctive.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------

const VIEWS = [
  { key: 'geometry', label: 'Table geometry' },
  { key: 'rare', label: 'Rare pairs' },
  { key: 'size', label: 'Test size' },
  { key: 'bigrams', label: 'Finance bigrams' },
  { key: 'keywords', label: 'Keywords' },
] as const;

export default function CollocationLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('geometry');
  return (
    <div data-lab="collocation" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'geometry' && <GeometryView />}
      {view === 'rare' && <RareView />}
      {view === 'size' && <SizeView />}
      {view === 'bigrams' && <BigramsView />}
      {view === 'keywords' && <KeywordsView />}
    </div>
  );
}

import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/significance-testing-for-nlp/significance_testing_for_nlp.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_prop2_inversions_and_symmetry, test_thm2_bootstrap_variance_matches_linearization,
// test_prop1_randomization_is_mcnemar, test_prop4_connor_n_has_about_80_percent_power,
// test_power_rises_with_n_and_flip_rate, test_thm1_exact_aa_size, test_prop2_bootstrap_size_is_asymptotic,
// test_prop5_metric_shopping, test_thm3_aspect_rows, test_thm4_total_variance_and_both_directions,
// test_holm_and_bh, test_case_study_the_metric_decides).
// Change the .py, regenerate, never the reverse. The reference curves (kα, 1 − (1 − α)^k, Theorem 3(a)'s design
// effect) are drawn here from their closed forms; every printed number is a payload value.
import payload from './data/significance-testing-lab.json';

/**
 * Significance Testing Laboratory — six figures for the significance-testing-for-nlp topic.
 *   A. "Two clouds": the bootstrap and randomization clouds of a corpus-BLEU gap, with three p-values (Defs 2–3,
 *      Proposition 2, Theorem 2).
 *   B. "McNemar": four tone-classifier pairs, their discordant counts, and exact power against test size
 *      (Propositions 1 and 4).
 *   C. "Power": subsampling power against the number of filings, per metric, when the truth is known.
 *   D. "A/A": p-value histograms under an exact null, and the size of metric shopping (Theorem 1, Propositions 2, 5).
 *   E. "Clusters & training": design effects on the aspect rows, and test-set p-values across training draws
 *      (Theorems 3 and 4).
 *   F. "Multiplicity": the 45 summarizer pairs under corpus BLEU and fact F1, raw, Holm, and Benjamini–Hochberg.
 */

type Cloud = {
  a: string; b: string; n: number; d: number; lo: number; hi: number; boot: number[]; null: number[];
  p_ar: number; p_koehn: number; p_shift: number; sd_boot: number; sd_lin: number | null;
};
const CLOUDS = payload.clouds as unknown as Cloud[];
type McRow = {
  a: string; b: string; acc_a: number; acc_b: number; n10: number; n01: number; n: number; p: number; n80: number;
  power: number; grid: number[]; power_curve: number[];
};
const MC = payload.mcnemar as unknown as McRow[];
const POWER = payload.power as unknown as { n: number[]; metrics: string[]; pairs: [string, string][]; curves: Record<string, number[]> };
type AARow = {
  n: number; experiments: number; hist_ar: number[]; hist_koehn: number[]; hist_shift: number[];
  size_ar: number; size_koehn: number; size_shift: number; shopping: number[] | null;
};
const AA = payload.aa as unknown as { n: number[]; metrics: string[]; rows: AARow[] };
type ClRow = { de: number; icc: number; mean: number; se_segment: number; se_cluster: number };
const CL = payload.clusters as unknown as { rows: number; docs: number; sentences: number; table: Record<string, ClRow> };
type Draw = { pop: number; test: number; p: number; p_large: number; n10: number; n01: number };
const DRAWS = payload.draws as unknown as {
  n: number; test_n: number; large_n: number; draws: Draw[];
  summary: Record<'total' | 'within' | 'between' | 'mean_pop' | 'se_mean_pop' | 'reject' | 'reject_lr' | 'reject_nb' | 'reject_large' | 'reject_large_lr' | 'reject_large_nb', number>;
};
type MatMetric = { d: number[]; p: number[]; holm: number[]; bh: boolean[] };
const MAT = payload.matrix as unknown as { systems: string[]; pairs: [string, string][]; bleu: MatMetric; f1: MatMetric };

const ALPHA = 0.05;
const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const f4 = (v: number) => v.toFixed(4);
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** A p-value: four decimals, or mantissa × 10^k below 10⁻⁴ (the payload keeps three significant figures there). */
const fp = (p: number) => {
  if (p === 0 || p >= 1e-4) return p.toFixed(4);
  const [m, e] = p.toExponential(2).split('e');
  return `${m} × 10${String(Number(e)).split('').map((c) => SUP[c]).join('')}`;
};
const NAMES: Record<string, string> = {
  writer: 'writer', 'noisy-0.1': 'noisy 0.1', 'noisy-0.3': 'noisy 0.3', 'noisy-0.5': 'noisy 0.5', terse: 'terse',
  flipped: 'flipped', extract: 'extract', 'textrank-3': 'TextRank', 'lead-3': 'lead-3', 'random-3': 'random-3',
};
const CLF: Record<string, string> = {
  'lr-uni': 'LR, unigrams', 'nb-uni': 'NB, unigrams', 'lr-bi': 'LR, bigrams', 'nbw-bi': 'NB-weighted, bigrams',
  'nbw-uni': 'NB-weighted, unigrams',
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

/** A small chart frame with gridlines at the given ticks. */
function Frame({ W, H, m, x, y, xTicks, yTicks, xFmt, yFmt, xLabel, children, panel, aria }: {
  W: number; H: number; m: { l: number; r: number; t: number; b: number };
  x: (v: number) => number; y: (v: number) => number; xTicks: number[]; yTicks: number[];
  xFmt: (v: number) => string; yFmt: (v: number) => string; xLabel: string; children: ReactNode; panel: string; aria: string;
}) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '32rem', height: 'auto', display: 'block' }} role="img" data-panel={panel} aria-label={aria}>
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

/** A histogram drawn as bars over equal-width bins of [lo, hi]. */
function Bars({ counts, lo, hi, x, y, base, color, opacity, id }: {
  counts: number[]; lo: number; hi: number; x: (v: number) => number; y: (v: number) => number; base: number;
  color: string; opacity: number; id: string;
}) {
  const w = (hi - lo) / counts.length;
  return (
    <g data-bars={id}>
      {counts.map((c, i) => c > 0 && (
        <rect key={i} x={x(lo + i * w)} width={Math.max(0.5, x(lo + (i + 1) * w) - x(lo + i * w) - 0.4)} y={y(c)} height={base - y(c)}
          fill={color} opacity={opacity} />
      ))}
    </g>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// A. Two clouds.
// ---------------------------------------------------------------------------------------------------------------------

const CLOUD_PAIRS = Array.from(new Set(CLOUDS.map((c) => `${c.a}|${c.b}`)));
const CLOUD_N = Array.from(new Set(CLOUDS.map((c) => c.n)));

function CloudsView() {
  const [pair, setPair] = useState(CLOUD_PAIRS[0]);
  const [n, setN] = useState(90);
  const c = CLOUDS.find((v) => `${v.a}|${v.b}` === pair && v.n === n)!;
  const W = 460, H = 240, m = { l: 36, r: 12, t: 10, b: 34 };
  const x = d3.scaleLinear().domain([c.lo, c.hi]).range([m.l, W - m.r]);
  const top = Math.max(...c.boot, ...c.null);
  const y = d3.scaleLinear().domain([0, top * 1.05]).range([H - m.b, m.t]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Corpus-BLEU gap of ${NAMES[c.a]} over ${NAMES[c.b]} on ${c.n} filings: 4,000 bootstrap resamples and 4,000 random swaps`}</Caption>
      <Row>
        <Label>pair</Label>
        {CLOUD_PAIRS.map((p) => {
          const [a, b] = p.split('|');
          return <Pill key={p} active={pair === p} onClick={() => setPair(p)} data={{ 'data-pair': p }}>{`${NAMES[a]} vs. ${NAMES[b]}`}</Pill>;
        })}
      </Row>
      <Row>
        <Label>filings</Label>
        {CLOUD_N.map((k) => <Pill key={k} active={n === k} onClick={() => setN(k)} data={{ 'data-n': `${k}` }}>{`${k}`}</Pill>)}
      </Row>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={x.ticks(5)} yTicks={y.ticks(4)} xFmt={(v) => v.toFixed(3)} yFmt={(v) => `${v}`}
        xLabel="gap in corpus BLEU" panel="clouds" aria="Histograms of bootstrap and randomization gaps">
        <Bars counts={c.null} lo={c.lo} hi={c.hi} x={x} y={y} base={H - m.b} color="var(--color-text-secondary)" opacity={0.45} id="null" />
        <Bars counts={c.boot} lo={c.lo} hi={c.hi} x={x} y={y} base={H - m.b} color="var(--color-accent)" opacity={0.6} id="boot" />
        <line x1={x(0)} x2={x(0)} y1={m.t} y2={H - m.b} stroke="var(--color-text)" strokeDasharray="3 3" strokeWidth={1} />
        <line x1={x(c.d)} x2={x(c.d)} y1={m.t} y2={H - m.b} stroke="var(--color-accent)" strokeWidth={2} />
      </Frame>
      <Key items={[
        { label: 'bootstrap δ* (centered near δ̂)', color: 'var(--color-accent)' },
        { label: 'randomization δ (centered at 0)', color: 'var(--color-text-secondary)' },
        { label: 'observed δ̂', color: 'var(--color-accent)' },
        { label: 'zero', color: 'var(--color-text)', dash: true },
      ]} />
      <Readouts>
        <Readout label="observed gap δ̂" value={f4(c.d)} id="cloud-d" />
        <Readout label="randomization p (two-sided)" value={fp(c.p_ar)} id="cloud-ar" emphasis />
        <Readout label="Koehn p (one-sided)" value={fp(c.p_koehn)} id="cloud-koehn" />
        <Readout label="2δ rule p (one-sided)" value={fp(c.p_shift)} id="cloud-shift" />
        <Readout label="bootstrap sd" value={f4(c.sd_boot)} id="cloud-sdb" />
        <Readout label="delta-method sd" value={c.sd_lin === null ? 'undefined' : f4(c.sd_lin)} id="cloud-sdl" />
      </Readouts>
      <Note>Koehn's p counts resamples where the first system does not win; the 2δ rule counts resamples that win by at least twice the observed gap. They invert the percentile and the basic interval (Proposition 2). The delta-method sd is Theorem 2's linearization; it is undefined when a system matches no 4-gram, and loose for the extractors, which match only a handful.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. McNemar.
// ---------------------------------------------------------------------------------------------------------------------

function McNemarView() {
  const [k, setK] = useState(2);
  const r = MC[k];
  const W = 460, H = 240, m = { l: 36, r: 12, t: 10, b: 34 };
  const x = d3.scaleLog().domain([r.grid[0], r.grid[r.grid.length - 1]]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  const line = d3.line<number>().x((_, i) => x(r.grid[i])).y((v) => y(v))(r.power_curve) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Tone classifiers on ${r.n.toLocaleString('en-US')} synthetic filings: McNemar's exact test and its power`}</Caption>
      <Row>
        <Label>pair</Label>
        {MC.map((v, i) => <Pill key={i} active={k === i} onClick={() => setK(i)} data={{ 'data-mc': `${i}` }}>{`${CLF[v.a]} vs. ${CLF[v.b]}`}</Pill>)}
      </Row>
      <Table id="mcnemar" head={['', `${CLF[r.b]} right`, `${CLF[r.b]} wrong`]}
        rows={[
          { key: 'a-right', cells: [`${CLF[r.a]} right`, '·', `${r.n10}`] },
          { key: 'a-wrong', cells: [`${CLF[r.a]} wrong`, `${r.n01}`, '·'] },
        ]} />
      <Readouts>
        <Readout label="accuracies" value={`${f3(r.acc_a)} vs. ${f3(r.acc_b)}`} id="mc-acc" />
        <Readout label="exact McNemar p" value={fp(r.p)} id="mc-p" emphasis />
        <Readout label={`power at ${r.n.toLocaleString('en-US')}`} value={f3(r.power)} id="mc-power" />
        <Readout label="filings for 80% power" value={r.n80.toLocaleString('en-US')} id="mc-n80" />
      </Readouts>
      <Sub>Exact power against the number of filings, at the observed discordance rates (Proposition 4)</Sub>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[100, 1000, 10000, 100000]} yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
        xFmt={(v) => v.toLocaleString('en-US')} yFmt={(v) => v.toFixed(1)} xLabel="test filings (log scale)" panel="mc-power" aria="Power of McNemar's test against test size">
        <line x1={m.l} x2={W - m.r} y1={y(0.8)} y2={y(0.8)} stroke="var(--color-text-secondary)" strokeDasharray="3 3" />
        <path d={line} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
        <line x1={x(r.n)} x2={x(r.n)} y1={m.t} y2={H - m.b} stroke="var(--color-text)" strokeWidth={1} />
        {r.n80 <= r.grid[r.grid.length - 1] && (
          <line x1={x(r.n80)} x2={x(r.n80)} y1={m.t} y2={H - m.b} stroke="var(--color-accent)" strokeDasharray="4 3" strokeWidth={1} />
        )}
      </Frame>
      <Key items={[
        { label: 'exact power', color: 'var(--color-accent)' },
        { label: 'the 1,500 test filings', color: 'var(--color-text)' },
        { label: "Connor's n for 80%", color: 'var(--color-accent)', dash: true },
      ]} />
      <Note>Only the discordant filings carry information: concordant ones are unchanged by any swap, so randomization on right-or-wrong outcomes is exactly this sign test (Proposition 1).</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Power.
// ---------------------------------------------------------------------------------------------------------------------

const METRIC_NAMES: Record<string, string> = { bleu: 'corpus BLEU', chrf: 'chrF', f1: 'fact F1' };
const METRIC_COLORS: Record<string, string> = {
  bleu: 'var(--color-accent)', chrf: 'var(--color-badge-blue-text)', f1: 'var(--color-badge-green-text)',
};

function PowerView() {
  const [k, setK] = useState(0);
  const [a, b] = POWER.pairs[k];
  const W = 460, H = 250, m = { l: 36, r: 12, t: 10, b: 34 };
  const x = d3.scaleLinear().domain([0, 47]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`How often randomization finds the ${NAMES[b]} summaries worse than the writer's, on random subsets of filings`}</Caption>
      <Row>
        <Label>facts reversed</Label>
        {POWER.pairs.map(([, bb], i) => <Pill key={bb} active={k === i} onClick={() => setK(i)} data={{ 'data-flip': bb }}>{bb.replace('noisy-', '') }</Pill>)}
      </Row>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={POWER.n} yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} xFmt={(v) => `${v}`} yFmt={(v) => v.toFixed(1)}
        xLabel="filings in the subset" panel="power" aria="Rejection rate against the number of filings">
        {POWER.metrics.map((mt) => {
          const c = POWER.curves[`${a}|${b}|${mt}`];
          const path = d3.line<number>().x((_, i) => x(POWER.n[i])).y((v) => y(v))(c) ?? '';
          return (
            <g key={mt} data-curve={mt}>
              <path d={path} fill="none" stroke={METRIC_COLORS[mt]} strokeWidth={2} />
              {c.map((v, i) => <circle key={i} cx={x(POWER.n[i])} cy={y(v)} r={2.6} fill={METRIC_COLORS[mt]} />)}
            </g>
          );
        })}
      </Frame>
      <Key items={POWER.metrics.map((mt) => ({ label: METRIC_NAMES[mt], color: METRIC_COLORS[mt] }))} />
      <Table id="power" head={['metric', ...POWER.n.map((v) => `n = ${v}`)]}
        rows={POWER.metrics.map((mt) => ({ key: mt, cells: [METRIC_NAMES[mt], ...POWER.curves[`${a}|${b}|${mt}`].map(f3)] }))} />
      <Note>Every reversed fact lowers fact F1, so each subset's null hypothesis is false and the rejection rate is power. Each point is 200 subsets drawn without replacement from the 90 filings, each tested with 499 random swaps at level 0.05. A reversed verb changes a few characters, and chrF averages over all of them.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. A/A.
// ---------------------------------------------------------------------------------------------------------------------

type Method = 'ar' | 'koehn' | 'shift';
const METHOD_NAMES: Record<Method, string> = { ar: 'randomization', koehn: "Koehn's rule", shift: '2δ rule' };

function AAView() {
  const [n, setN] = useState(10);
  const [method, setMethod] = useState<Method>('ar');
  const r = AA.rows.find((v) => v.n === n)!;
  const counts = method === 'ar' ? r.hist_ar : method === 'koehn' ? r.hist_koehn : r.hist_shift;
  const sizes: Record<Method, number> = { ar: r.size_ar, koehn: r.size_koehn, shift: r.size_shift };
  const W = 460, H = 220, m = { l: 40, r: 12, t: 10, b: 34 };
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, Math.max(...counts, r.experiments / 20) * 1.1]).range([H - m.b, m.t]);
  const S = AA.rows.filter((v) => v.shopping !== null);
  const ks = AA.metrics.map((_, i) => i + 1);
  const xs = d3.scaleLinear().domain([1, ks.length]).range([m.l, W - m.r]);
  const ys = d3.scaleLinear().domain([0, 0.35]).range([H - m.b, m.t]);
  const bound = (f: (k: number) => number) => d3.line<number>().x((k) => xs(k)).y((k) => ys(f(k)))(d3.range(1, ks.length + 0.01, 0.1)) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`An exact null: two trigram samples per prompt, ${r.experiments.toLocaleString('en-US')} independent experiments of ${n} prompts`}</Caption>
      <Row>
        <Label>prompts</Label>
        {AA.n.map((k) => <Pill key={k} active={n === k} onClick={() => setN(k)} data={{ 'data-aa': `${k}` }}>{`${k}`}</Pill>)}
      </Row>
      <Row>
        <Label>test</Label>
        {(Object.keys(METHOD_NAMES) as Method[]).map((k) => <Pill key={k} active={method === k} onClick={() => setMethod(k)} data={{ 'data-method': k }}>{METHOD_NAMES[k]}</Pill>)}
      </Row>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yTicks={y.ticks(4)} xFmt={(v) => v.toFixed(1)} yFmt={(v) => `${v}`}
        xLabel="p-value (corpus BLEU)" panel="aa" aria="Histogram of p-values under the null">
        <Bars counts={counts} lo={0} hi={1} x={x} y={y} base={H - m.b} color="var(--color-accent)" opacity={0.7} id={method} />
        <line x1={m.l} x2={W - m.r} y1={y(r.experiments / 20)} y2={y(r.experiments / 20)} stroke="var(--color-text)" strokeDasharray="3 3" />
      </Frame>
      <Key items={[{ label: 'p-values', color: 'var(--color-accent)' }, { label: 'uniform', color: 'var(--color-text)', dash: true }]} />
      <Readouts>
        {(Object.keys(METHOD_NAMES) as Method[]).map((k) => (
          <Readout key={k} label={`${METHOD_NAMES[k]}: share rejected at 0.05`} value={f3(sizes[k])} id={`aa-${k}`} emphasis={k === method} />
        ))}
      </Readouts>
      <Sub>Metric shopping: the share rejected when any of the first k metrics rejects (Proposition 5)</Sub>
      <Frame W={W} H={H} m={m} x={xs} y={ys} xTicks={ks} yTicks={[0, 0.1, 0.2, 0.3]} xFmt={(v) => `${v}`} yFmt={(v) => v.toFixed(1)}
        xLabel="metrics tested (k)" panel="shopping" aria="Size of metric shopping against the number of metrics">
        <path d={bound((k) => Math.min(1, k * ALPHA))} fill="none" stroke="var(--color-text-secondary)" strokeDasharray="4 3" />
        <path d={bound((k) => 1 - (1 - ALPHA) ** k)} fill="none" stroke="var(--color-text-secondary)" strokeDasharray="1 3" />
        {S.map((row, j) => {
          const c = row.shopping!;
          const color = j === 0 ? 'var(--color-accent)' : 'var(--color-badge-blue-text)';
          const path = d3.line<number>().x((_, i) => xs(i + 1)).y((v) => ys(v))(c) ?? '';
          return (
            <g key={row.n} data-shopping={row.n}>
              <path d={path} fill="none" stroke={color} strokeWidth={2} />
              {c.map((v, i) => <circle key={i} cx={xs(i + 1)} cy={ys(v)} r={2.6} fill={color} />)}
            </g>
          );
        })}
      </Frame>
      <Key items={[
        ...S.map((row, j) => ({ label: `${row.n} prompts`, color: j === 0 ? 'var(--color-accent)' : 'var(--color-badge-blue-text)' })),
        { label: 'kα (Bonferroni)', color: 'var(--color-text-secondary)', dash: true },
        { label: '1 − (1 − α)^k (independent)', color: 'var(--color-text-secondary)', dash: true },
      ]} />
      <Note>{`Metrics in order: ${AA.metrics.join(', ')}. Randomization stays within its binomial band at every size (Theorem 1); its p-values pile up below uniform at the top because many swaps of a short block change nothing. The 2δ rule leaves its band at 10 prompts (Proposition 2).`}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// E. Clusters and training draws.
// ---------------------------------------------------------------------------------------------------------------------

const CL_ROWS: { key: string; label: string }[] = [
  { key: 'blind/sent', label: 'aspect-blind reader, sentences' },
  { key: 'blind/doc', label: 'aspect-blind reader, documents' },
  { key: 'lexicon/doc', label: 'lexicon reader, documents' },
  { key: 'difference/doc', label: 'paired difference, documents' },
];

function ClustersView() {
  const [large, setLarge] = useState(false);
  const s = DRAWS.summary;
  const W = 460, H = 250, m = { l: 44, r: 12, t: 10, b: 34 };
  const ext = Math.max(...DRAWS.draws.map((d) => Math.abs(d.pop))) * 1.1;
  const x = d3.scaleLinear().domain([-ext, ext]).range([m.l, W - m.r]);
  const y = d3.scaleLog().domain([1e-4, 1]).range([H - m.b, m.t]).clamp(true);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Clustered aspects: ${CL.rows.toLocaleString('en-US')} aspect rows in ${CL.sentences.toLocaleString('en-US')} sentences of ${CL.docs.toLocaleString('en-US')} filings`}</Caption>
      <Table id="clusters" head={['segments, clusters', 'mean', 'design effect', 'within-cluster ρ', 'SE, segments', 'SE, clusters']}
        rows={CL_ROWS.map(({ key, label }) => {
          const r = CL.table[key];
          return { key, strong: key === 'lexicon/doc', cells: [label, f3(r.mean), f3(r.de), f3(r.icc), f4(r.se_segment), f4(r.se_cluster)] };
        })} />
      <Note>The design effect is the cluster bootstrap's variance over the segment bootstrap's (Theorem 3). Within sentences, ρ is the correlation between the two aspects of a contrast sentence, where the aspect-blind reader is always right on exactly one; within documents it is the ANOVA intraclass correlation. Pairing the two readers cancels most of the document effect.</Note>
      <Sub>{`A test set cannot test a procedure: 40 training sets of ${DRAWS.n} documents, where NB and LR are equal on average (Theorem 4)`}</Sub>
      <Row>
        <Label>test set</Label>
        <Pill active={!large} onClick={() => setLarge(false)} data={{ 'data-test': 'small' }}>{`${DRAWS.test_n.toLocaleString('en-US')} documents`}</Pill>
        <Pill active={large} onClick={() => setLarge(true)} data={{ 'data-test': 'large' }}>{`${DRAWS.large_n.toLocaleString('en-US')} documents`}</Pill>
      </Row>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={x.ticks(5)} yTicks={[1e-4, 1e-3, 1e-2, 0.05, 1]} xFmt={(v) => v.toFixed(3)}
        yFmt={(v) => (v >= 0.01 ? `${v}` : v === 1e-3 ? '0.001' : '0.0001')} xLabel="population accuracy gap, LR − NB" panel="draws" aria="Test-set p-values against the training draw's accuracy gap">
        <line x1={m.l} x2={W - m.r} y1={y(ALPHA)} y2={y(ALPHA)} stroke="var(--color-accent)" strokeDasharray="4 3" />
        <line x1={x(0)} x2={x(0)} y1={m.t} y2={H - m.b} stroke="var(--color-text-secondary)" strokeDasharray="2 3" />
        {DRAWS.draws.map((d, i) => {
          const p = large ? d.p_large : d.p;
          return <circle key={i} data-draw={i} cx={x(d.pop)} cy={y(Math.max(p, 1e-4))} r={3.4}
            fill={p <= ALPHA ? 'var(--color-accent)' : 'var(--color-surface)'} stroke="var(--color-accent)" strokeWidth={1} />;
        })}
      </Frame>
      <Key items={[{ label: 'rejects at 0.05', color: 'var(--color-accent)', dot: true }, { label: 'α = 0.05', color: 'var(--color-accent)', dash: true }]} />
      <Readouts>
        <Readout label="mean population gap" value={`${f4(s.mean_pop)} ± ${f4(s.se_mean_pop)}`} id="draws-mean" />
        <Readout label="share rejected" value={f3(large ? s.reject_large : s.reject)} id="draws-reject" emphasis />
        <Readout label="LR wins / NB wins" value={large ? `${f3(s.reject_large_lr)} / ${f3(s.reject_large_nb)}` : `${f3(s.reject_lr)} / ${f3(s.reject_nb)}`} id="draws-sides" />
      </Readouts>
      <Note>Each point is one training set: its gap measured on 20,000 documents, and the McNemar p-value on the chosen test set (clamped at 0.0001). A bigger test set rejects more often, in both directions, although the two procedures are equal on average.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// F. Multiplicity.
// ---------------------------------------------------------------------------------------------------------------------

type Correction = 'raw' | 'holm' | 'bh';
const CORR_NAMES: Record<Correction, string> = { raw: 'raw', holm: 'Holm', bh: 'Benjamini–Hochberg' };
const sig = (mt: MatMetric, i: number, c: Correction) => (c === 'raw' ? mt.p[i] <= ALPHA : c === 'holm' ? mt.holm[i] <= ALPHA : mt.bh[i]);

function MatrixView() {
  const [metric, setMetric] = useState<'bleu' | 'f1'>('bleu');
  const [corr, setCorr] = useState<Correction>('raw');
  const [sel, setSel] = useState(MAT.pairs.findIndex(([a, b]) => a === 'flipped' && b === 'extract'));
  const S = MAT.systems;
  const mt = MAT[metric];
  const cell = 26, left = 70, top = 70;
  const W = left + S.length * cell + 4, H = top + S.length * cell + 4;
  const differs = (i: number) => {
    const a = sig(MAT.bleu, i, 'raw'), b = sig(MAT.f1, i, 'raw');
    return a !== b || (a && b && Math.sign(MAT.bleu.d[i]) !== Math.sign(MAT.f1.d[i]));
  };
  const nSig = mt.p.filter((_, i) => sig(mt, i, corr)).length;
  const nDiff = MAT.pairs.filter((_, i) => differs(i)).length;
  const [sa, sb] = MAT.pairs[sel];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>The 45 summarizer pairs on 90 filings: two-sided randomization, 9,999 swaps</Caption>
      <Row>
        <Label>metric</Label>
        <Pill active={metric === 'bleu'} onClick={() => setMetric('bleu')} data={{ 'data-metric': 'bleu' }}>corpus BLEU</Pill>
        <Pill active={metric === 'f1'} onClick={() => setMetric('f1')} data={{ 'data-metric': 'f1' }}>fact F1</Pill>
      </Row>
      <Row>
        <Label>correction</Label>
        {(Object.keys(CORR_NAMES) as Correction[]).map((c) => <Pill key={c} active={corr === c} onClick={() => setCorr(c)} data={{ 'data-corr': c }}>{CORR_NAMES[c]}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '26rem', height: 'auto', display: 'block' }} role="img" data-panel="matrix" aria-label="Matrix of pairwise significance">
        {S.map((s, j) => (
          <text key={`c${s}`} transform={`translate(${left + j * cell + cell / 2 + 3},${top - 4}) rotate(-55)`} style={{ ...axisText, fill: 'var(--color-text)' }}>{NAMES[s]}</text>
        ))}
        {S.map((s, i) => <text key={`r${s}`} x={left - 4} y={top + i * cell + cell / 2 + 3} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{NAMES[s]}</text>)}
        {MAT.pairs.map(([a, b], k) => {
          const i = S.indexOf(a), j = S.indexOf(b);
          const on = sig(mt, k, corr);
          return [[i, j, 1], [j, i, -1]].map(([r, c, s]) => {
            const better = on ? (mt.d[k] * s > 0 ? 'var(--color-accent)' : 'var(--color-badge-blue-text)') : 'var(--color-surface)';
            return (
              <rect key={`${k}-${s}`} data-cell={`${S[r]}|${S[c]}`} x={left + c * cell + 1} y={top + r * cell + 1} width={cell - 2} height={cell - 2}
                fill={better} opacity={on ? 0.75 : 1} stroke={differs(k) ? 'var(--color-text)' : 'var(--color-border)'}
                strokeWidth={differs(k) ? 1.6 : 0.6} style={{ cursor: 'pointer' }} onClick={() => setSel(k)} />
            );
          });
        })}
        {S.map((s, i) => <rect key={`d${s}`} x={left + i * cell + 1} y={top + i * cell + 1} width={cell - 2} height={cell - 2} fill="var(--color-border)" />)}
      </svg>
      <Key items={[
        { label: 'row system significantly better', color: 'var(--color-accent)' },
        { label: 'row system significantly worse', color: 'var(--color-badge-blue-text)' },
        { label: 'outlined: BLEU and fact F1 disagree', color: 'var(--color-text)' },
      ]} />
      <Readouts>
        <Readout label={`significant pairs (${CORR_NAMES[corr]})`} value={`${nSig} of ${MAT.pairs.length}`} id="mat-sig" emphasis />
        <Readout label="pairs where the metrics disagree" value={`${nDiff} of ${MAT.pairs.length}`} id="mat-diff" />
      </Readouts>
      <Table id="pair" head={['selected pair', 'gap', 'p', 'Holm p']}
        rows={(['bleu', 'f1'] as const).map((k) => ({
          key: k, strong: k === metric,
          cells: [`${NAMES[sa]} vs. ${NAMES[sb]}, ${k === 'bleu' ? 'BLEU' : 'fact F1'}`, f4(MAT[k].d[sel]), fp(MAT[k].p[sel]), fp(MAT[k].holm[sel])],
        }))} />
      <Note>{`Click a cell to read its pair. Disagreement means significant on one metric only, or on both in opposite directions (raw p-values). The corrections change no verdict here: the p-values sit at the floor of 0.0001 or far above 0.05.`}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------

const VIEWS = [
  { key: 'clouds', label: 'Two clouds' },
  { key: 'mcnemar', label: 'McNemar' },
  { key: 'power', label: 'Power' },
  { key: 'aa', label: 'A/A' },
  { key: 'clusters', label: 'Clusters & training' },
  { key: 'matrix', label: 'Multiplicity' },
] as const;

export default function SignificanceTestingLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('clouds');
  return (
    <div data-lab="significance-testing" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'clouds' && <CloudsView />}
      {view === 'mcnemar' && <McNemarView />}
      {view === 'power' && <PowerView />}
      {view === 'aa' && <AAView />}
      {view === 'clusters' && <ClustersView />}
      {view === 'matrix' && <MatrixView />}
    </div>
  );
}

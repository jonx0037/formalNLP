import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/overlap-metrics-bleu-rouge/overlap_metrics_bleu_rouge.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_prop1_matching_picture, test_thm2_lcs_is_longest_path, test_thm1_length_envelope,
// test_thm1_decoding_points_under_envelope, test_cor1_flips_within_the_bound, test_case_study_flipped_beats_extract,
// test_prop7_bias_survives_averaging, test_thm4_mbr_on_the_corpus).
// Change the .py, regenerate, never the reverse. The envelope curves are drawn here from Theorem 1's closed form;
// every printed number is a payload value.
import payload from './data/overlap-metrics-lab.json';

/**
 * Overlap Metrics Laboratory — five figures for the overlap-metrics-bleu-rouge topic.
 *   A. "Match grid": one filing's summary against one reference: the marked cells, a maximum matching of unigrams and
 *      of bigrams, and the longest chain (Definitions 1 and 3, Proposition 1, Theorem 2).
 *   B. "Length envelope": BLEU against c/r under Theorem 1's envelopes, with reference prefixes and extensions and
 *      the decoding systems.
 *   C. "Flip vs. paraphrase": every metric and fact F1 as a growing share of facts is reversed (Theorem 3,
 *      Corollary 1), against the fact-bearing extract.
 *   D. "Correlation": Kendall's τ per document, per system, and among systems that differ only by flips, and its
 *      climb with the number of documents (Proposition 7).
 *   E. "MBR vs. mode": corpus BLEU of the decoding systems and five prompts whose mode is empty (Theorem 4).
 */

type Metric = 'bleu' | 'rouge-1' | 'rouge-2' | 'rouge-l' | 'chrf' | 'soft';
const METRICS: Metric[] = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'chrf', 'soft'];
const METRIC_NAMES: Record<Metric, string> = {
  bleu: 'BLEU', 'rouge-1': 'ROUGE-1', 'rouge-2': 'ROUGE-2', 'rouge-l': 'ROUGE-L', chrf: 'chrF', soft: 'soft F',
};
const METRIC_COLORS: Record<Metric, string> = {
  bleu: 'var(--color-accent)', 'rouge-1': 'var(--color-badge-blue-text)', 'rouge-2': 'var(--color-badge-amber-text)',
  'rouge-l': 'var(--color-badge-green-text)', chrf: 'var(--color-accent-secondary)', soft: 'var(--color-badge-red-text)',
};
type Pair = [number, number];
type GridSystem = {
  tokens: string[]; matches: Pair[]; clip1: Pair[]; clip2: Pair[]; lcs: Pair[];
  scores: Record<Metric, number>; f1: number;
};
const GRID = payload.grid as unknown as {
  doc: string; company: string; facts: [string, number][]; ref: string[]; refs: string[];
  systems: Record<'writer' | 'flipped' | 'extract', GridSystem>;
};
const ENV = payload.envelope as unknown as {
  sweep: { x: number; bleu: number; bound: number; segment_bound: number }[];
  systems: Record<DecSystem, { x: number; bleu: number }>;
};
const FLIP = payload.flip as unknown as Record<Metric | 'f1', number[]> & { rates: number[]; extract: Record<Metric | 'f1', number> };
const SYSTEMS = payload.systems as unknown as Record<string, Record<Metric | 'f1' | 'length', number>>;
const CORR = payload.correlation as unknown as {
  metrics: Record<Metric, { segment: number; system: number; family: number }>; n: number[]; curve: Record<Metric, number[]>;
};
type DecSystem = 'mode' | 'beam-16' | 'norm-16' | 'sample' | 'mbr';
const DEC: DecSystem[] = ['mode', 'beam-16', 'norm-16', 'sample', 'mbr'];
const DEC_NAMES: Record<DecSystem, string> = {
  mode: 'exact mode', 'beam-16': 'beam, k = 16', 'norm-16': 'normalized beam', sample: 'one sample', mbr: 'BLEU-MBR',
};
const MBR = payload.mbr as unknown as {
  systems: Record<DecSystem, { bleu: number; length: number; empty: number; bp: number }>; reference_length: number;
  examples: ({ prompt: string; refs: string[]; scores: Record<DecSystem, number> } & Record<DecSystem, string>)[];
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);

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

// ---------------------------------------------------------------------------------------------------------------------
// A. Match grid.
// ---------------------------------------------------------------------------------------------------------------------

type GridKey = keyof typeof GRID.systems;
const GRID_NAMES: Record<GridKey, string> = { writer: 'writer (facts right)', flipped: 'flipped (facts reversed)', extract: 'extract (document sentences)' };
type Layer = 'marks' | 'clip1' | 'clip2' | 'lcs';
const LAYER_NAMES: Record<Layer, string> = { marks: 'all matches', clip1: 'clipped unigrams', clip2: 'clipped bigrams', lcs: 'longest chain' };

function GridView() {
  const [sys, setSys] = useState<GridKey>('flipped');
  const [layer, setLayer] = useState<Layer>('lcs');
  const g = GRID.systems[sys];
  const cell = 16;
  const left = 132;
  const top = 92;
  const W = left + GRID.ref.length * cell + 8;
  const H = top + g.tokens.length * cell + 8;
  const cx = (j: number) => left + j * cell + cell / 2;
  const cy = (i: number) => top + i * cell + cell / 2;
  const clipped = new Set((layer === 'clip1' ? g.clip1 : []).map(([i, j]) => `${i},${j}`));
  const chain = d3.line<Pair>().x(([, j]) => cx(j)).y(([i]) => cy(i))(g.lcs) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`One summary of ${GRID.company} (${GRID.doc}) against one reference: rows are the summary, columns the reference`}</Caption>
      <Row>
        <Label>summary</Label>
        {(Object.keys(GRID_NAMES) as GridKey[]).map((k) => <Pill key={k} active={sys === k} onClick={() => setSys(k)} data={{ 'data-system': k }}>{GRID_NAMES[k]}</Pill>)}
      </Row>
      <Row>
        <Label>layer</Label>
        {(Object.keys(LAYER_NAMES) as Layer[]).map((k) => <Pill key={k} active={layer === k} onClick={() => setLayer(k)} data={{ 'data-layer': k }}>{LAYER_NAMES[k]}</Pill>)}
      </Row>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" data-panel="grid" aria-label="Match grid of a summary against a reference" style={{ display: 'block' }}>
          {GRID.ref.map((w, j) => (
            <text key={`r${j}`} transform={`translate(${cx(j) + 3},${top - 4}) rotate(-60)`} style={{ ...axisText, fill: 'var(--color-text)' }}>{w}</text>
          ))}
          {g.tokens.map((w, i) => (
            <text key={`c${i}`} x={left - 6} y={cy(i) + 3} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{w}</text>
          ))}
          {g.tokens.map((_, i) => GRID.ref.map((__, j) => (
            <rect key={`b${i}-${j}`} x={left + j * cell} y={top + i * cell} width={cell} height={cell} fill="none" stroke="var(--color-border)" strokeWidth={0.4} />
          )))}
          {g.matches.map(([i, j]) => (
            <circle key={`m${i}-${j}`} data-mark={`${i},${j}`} cx={cx(j)} cy={cy(i)} r={3.2}
              fill={clipped.has(`${i},${j}`) ? 'var(--color-accent)' : 'var(--color-text-secondary)'} opacity={layer === 'marks' || layer === 'clip1' ? 1 : 0.45} />
          ))}
          {layer === 'clip1' && g.clip1.map(([i, j]) => (
            <circle key={`k${i}-${j}`} cx={cx(j)} cy={cy(i)} r={6} fill="none" stroke="var(--color-accent)" strokeWidth={1.4} />
          ))}
          {layer === 'clip2' && g.clip2.map(([i, j]) => (
            <line key={`d${i}-${j}`} data-bigram={`${i},${j}`} x1={cx(j)} y1={cy(i)} x2={cx(j + 1)} y2={cy(i + 1)} stroke="var(--color-accent)" strokeWidth={3} strokeLinecap="round" />
          ))}
          {layer === 'lcs' && (
            <g>
              <path d={chain} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
              {g.lcs.map(([i, j]) => <circle key={`l${i}-${j}`} cx={cx(j)} cy={cy(i)} r={4.2} fill="var(--color-accent)" />)}
            </g>
          )}
        </svg>
      </div>
      <Readouts>
        <Readout label="BLEU+1" value={f3(g.scores.bleu)} id="grid-bleu" />
        <Readout label="ROUGE-1 F" value={f3(g.scores['rouge-1'])} id="grid-r1" />
        <Readout label="ROUGE-2 F" value={f3(g.scores['rouge-2'])} id="grid-r2" />
        <Readout label="ROUGE-L F" value={f3(g.scores['rouge-l'])} id="grid-rl" />
        <Readout label="chrF" value={f3(g.scores.chrf)} id="grid-chrf" />
        <Readout label="soft F" value={f3(g.scores.soft)} id="grid-soft" />
        <Readout label="fact F1" value={f3(g.f1)} id="grid-f1" emphasis />
      </Readouts>
      <Note>{`Gold facts: ${GRID.facts.map(([c, p]) => `${c} ${p > 0 ? '+' : '−'}`).join(', ')}. Scores are against the one reference drawn; the page's tables use all four. A clipped matching pairs each summary n-gram with at most one reference n-gram (Proposition 1); the longest chain rises in both directions and never has more cells than the clipped unigrams (Theorem 2).`}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. Length envelope.
// ---------------------------------------------------------------------------------------------------------------------

function EnvelopeView() {
  const W = 460, H = 280, m = { l: 36, r: 12, t: 10, b: 34 };
  const x = d3.scaleLinear().domain([0, 3.2]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1.02]).range([H - m.b, m.t]);
  const xs = d3.range(0.12, 3.2, 0.01);
  const left = (v: number) => Math.exp(1 - 1 / v);
  const corpus = d3.line<number>().x(x).y((v) => y(v <= 1 ? left(v) : v ** -0.25))(xs) ?? '';
  const segment = d3.line<number>().x(x).y((v) => y(v <= 1 ? left(v) : 1 / v))(xs.filter((v) => v >= 1)) ?? '';
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Corpus BLEU against the length ratio c/r, one reference per prompt (Theorem 1)</Caption>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={[0, 0.5, 1, 1.5, 2, 2.5, 3]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
        xFmt={(v) => `${v}`} yFmt={(v) => `${v}`} xLabel="c/r" panel="envelope" aria="BLEU against length ratio under the envelope">
        <path d={corpus} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
        <path d={segment} fill="none" stroke="var(--color-accent-secondary)" strokeWidth={1.5} strokeDasharray="5 3" />
        {ENV.sweep.map((p) => <circle key={`s${p.x}`} data-sweep={p.x} cx={x(p.x)} cy={y(p.bleu)} r={3} fill="var(--color-text-secondary)" />)}
        {DEC.map((k) => (
          <g key={k}>
            <circle data-system={k} cx={x(ENV.systems[k].x)} cy={y(ENV.systems[k].bleu)} r={5} fill="var(--color-badge-blue-text)" />
            <text x={x(ENV.systems[k].x) + 7} y={y(ENV.systems[k].bleu) - (k === 'beam-16' ? -10 : 4)} style={{ ...axisText, fill: 'var(--color-text)' }}>{DEC_NAMES[k]}</text>
          </g>
        ))}
      </Frame>
      <Key items={[
        { label: 'corpus envelope: e^(1 − r/c), then (r/c)^(1/4)', color: 'var(--color-accent)' },
        { label: 'one-segment right branch: r/c', color: 'var(--color-accent-secondary)', dash: true },
        { label: 'reference prefixes and extensions', color: 'var(--color-text-secondary)', dot: true },
        { label: 'decoding systems', color: 'var(--color-badge-blue-text)', dot: true },
      ]} />
      <Table id="envelope" head={['system', 'c/r', 'BLEU']}
        rows={DEC.map((k) => ({ key: k, cells: [DEC_NAMES[k], f3(ENV.systems[k].x), f3(ENV.systems[k].bleu)] }))} />
      <Note>Prefixes of each prompt's first reference lie exactly on the left branch, since every precision is 1. Appending unmatched words stays under the right branch. The short mode is crushed by the brevity penalty.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Flip vs. paraphrase.
// ---------------------------------------------------------------------------------------------------------------------

function FlipView() {
  const [ri, setRi] = useState(10);
  const rate = FLIP.rates[ri];
  const rows: (Metric | 'f1')[] = [...METRICS, 'f1'];
  const W = 460, rowH = 26, lab = 74, H = rows.length * rowH + 20;
  const x = d3.scaleLinear().domain([0, 1]).range([lab, W - 50]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>The writer's summaries with a growing share of facts reversed, against the fact-bearing extract</Caption>
      <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem', fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginBottom: '0.45rem' }}>
        {`share of facts reversed: ${rate.toFixed(1)}`}
        <input type="range" min={0} max={FLIP.rates.length - 1} step={1} value={ri} data-slider="flip" aria-label="share of facts reversed"
          onChange={(e) => setRi(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0, maxWidth: '18rem', accentColor: 'var(--color-accent)' }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '32rem', height: 'auto', display: 'block' }} role="img" data-panel="flip" aria-label="Metric bars against fact F1">
        {rows.map((k, i) => {
          const v = FLIP[k][ri];
          const e = FLIP.extract[k];
          const yy = 10 + i * rowH;
          const color = k === 'f1' ? 'var(--color-text)' : METRIC_COLORS[k];
          return (
            <g key={k} data-bar={k}>
              <text x={lab - 6} y={yy + 13} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)', fontWeight: k === 'f1' ? 700 : 400 }}>{k === 'f1' ? 'fact F1' : METRIC_NAMES[k]}</text>
              <rect x={lab} y={yy + 3} width={x(1) - lab} height={14} fill="var(--color-surface)" stroke="var(--color-border)" strokeWidth={0.5} />
              <rect x={lab} y={yy + 3} width={x(v) - lab} height={14} fill={color} opacity={0.85} />
              <line x1={x(e)} x2={x(e)} y1={yy} y2={yy + 20} stroke="var(--color-text)" strokeWidth={1.5} strokeDasharray="3 2" />
              <text x={x(1) + 6} y={yy + 14} style={{ ...axisText, fill: 'var(--color-text)' }}>{f3(v)}</text>
            </g>
          );
        })}
      </svg>
      <Key items={[{ label: 'the extract (every fact right)', color: 'var(--color-text)', dash: true }]} />
      <Readouts>
        <Readout label="fact F1 of these summaries" value={f3(FLIP.f1[ri])} id="flip-f1" emphasis />
        <Readout label="their corpus BLEU" value={f3(FLIP.bleu[ri])} id="flip-bleu" />
        <Readout label="extract's corpus BLEU" value={f3(FLIP.extract.bleu)} id="flip-extract-bleu" />
        <Readout label="extract's fact F1" value={f3(FLIP.extract.f1)} id="flip-extract-f1" />
      </Readouts>
      <Note>Each reversal swaps one word, so every metric moves by a bounded amount (Theorem 3) while fact F1 falls to 0. Even at the right end, every metric still ranks the reversed summaries above the extract.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. Correlation.
// ---------------------------------------------------------------------------------------------------------------------

const SYSTEM_ORDER = ['writer', 'noisy-0.1', 'noisy-0.3', 'noisy-0.5', 'terse', 'flipped', 'extract', 'textrank-3', 'lead-3', 'random-3'];

function CorrelationView() {
  const W = 460, H = 250, m = { l: 36, r: 12, t: 10, b: 34 };
  const x = d3.scaleLog().domain([1, 90]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0.25, 1]).range([H - m.b, m.t]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Kendall's τ between each metric and fact F1 (Proposition 7)</Caption>
      <Table id="tau" head={['metric', 'per document', 'per system', 'flips only']}
        rows={METRICS.map((k) => ({ key: k, cells: [METRIC_NAMES[k], f3(CORR.metrics[k].segment), f3(CORR.metrics[k].system), f3(CORR.metrics[k].family)] }))} />
      <Sub>System-level τ when every system is scored on the same n random documents (mean over 100 draws)</Sub>
      <Frame W={W} H={H} m={m} x={x} y={y} xTicks={CORR.n} yTicks={[0.25, 0.5, 0.75, 1]}
        xFmt={(v) => `${v}`} yFmt={(v) => `${v}`} xLabel="documents per system n (log scale)" panel="tau-curve" aria="System-level tau against documents per system">
        <line x1={m.l} x2={W - m.r} y1={y(1)} y2={y(1)} stroke="var(--color-text-secondary)" strokeDasharray="4 3" />
        {METRICS.map((k) => (
          <g key={k}>
            <path d={d3.line<number>().x((_, i) => x(CORR.n[i])).y((v) => y(v))(CORR.curve[k]) ?? ''} fill="none" stroke={METRIC_COLORS[k]} strokeWidth={1.8} />
            {CORR.curve[k].map((v, i) => <circle key={i} cx={x(CORR.n[i])} cy={y(v)} r={2.4} fill={METRIC_COLORS[k]} />)}
          </g>
        ))}
      </Frame>
      <Key items={METRICS.map((k) => ({ label: METRIC_NAMES[k], color: METRIC_COLORS[k] }))} />
      <Sub>The ten systems (corpus scores; fact F1 is the answer key)</Sub>
      <Table id="systems" head={['system', 'BLEU', 'R-1', 'R-L', 'chrF', 'soft', 'fact F1']}
        rows={SYSTEM_ORDER.map((s) => ({
          key: s, strong: s === 'flipped' || s === 'extract',
          cells: [s, f3(SYSTEMS[s].bleu), f3(SYSTEMS[s]['rouge-1']), f3(SYSTEMS[s]['rouge-l']), f3(SYSTEMS[s].chrf), f3(SYSTEMS[s].soft), f3(SYSTEMS[s].f1)],
        }))} />
      <Note>More documents average away noise, so τ climbs, but it stops near one half: systems that share the references' wording score above systems that copy the document, whatever their facts. Among the five systems that differ only by flipped facts, every metric ranks perfectly.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// E. MBR vs. mode.
// ---------------------------------------------------------------------------------------------------------------------

function MBRView() {
  const [ei, setEi] = useState(0);
  const ex = MBR.examples[ei];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Decoding the trigram on 188 MD&A prompts: corpus BLEU against every held-out reference (mean length ${MBR.reference_length.toFixed(3)} words)`}</Caption>
      <Table id="mbr" head={['system', 'BLEU', 'length', 'empty', 'BP']}
        rows={DEC.map((k) => ({
          key: k, strong: k === 'mbr',
          cells: [DEC_NAMES[k], f3(MBR.systems[k].bleu), f3(MBR.systems[k].length), f3(MBR.systems[k].empty), f3(MBR.systems[k].bp)],
        }))} />
      <Row style={{ marginTop: '0.7rem' }}>
        <Label>prompt</Label>
        {MBR.examples.map((e, i) => <Pill key={e.prompt} active={ei === i} onClick={() => setEi(i)} data={{ 'data-example': `${i}` }}>{e.prompt}</Pill>)}
      </Row>
      <Sub>{`References (${ex.refs.length})`}</Sub>
      <div data-refs style={{ fontFamily: mono, fontSize: '0.78rem', display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
        {ex.refs.map((r, i) => <span key={i} style={{ overflowWrap: 'anywhere' }}>{`${ex.prompt} ${r}`}</span>)}
      </div>
      <Table id="mbr-example" head={['system', 'output after the prompt', 'BLEU+1']}
        rows={DEC.map((k) => ({ key: k, strong: k === 'mbr', cells: [DEC_NAMES[k], ex[k] === '' ? '(stops at once)' : ex[k], f3(ex.scores[k])] }))} />
      <Note>The mode stops early because the end mark is often the likeliest next token (the decoding topic's Theorem 4), and BLEU's brevity penalty punishes it. Minimum Bayes risk picks the sample with the highest average BLEU+1 against the other samples (Theorem 4).</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------

const VIEWS = [
  { key: 'grid', label: 'Match grid' },
  { key: 'envelope', label: 'Length envelope' },
  { key: 'flip', label: 'Flip vs. paraphrase' },
  { key: 'correlation', label: 'Correlation' },
  { key: 'mbr', label: 'MBR vs. mode' },
] as const;

export default function OverlapMetricsLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('grid');
  return (
    <div data-lab="overlap-metrics" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'grid' && <GridView />}
      {view === 'envelope' && <EnvelopeView />}
      {view === 'flip' && <FlipView />}
      {view === 'correlation' && <CorrelationView />}
      {view === 'mbr' && <MBRView />}
    </div>
  );
}

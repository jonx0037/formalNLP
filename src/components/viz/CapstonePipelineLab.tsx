import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/capstone-earnings-call-nlp/capstone_earnings_call_nlp.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_thm1_ceiling_analysis_and_the_shapley_value, test_thm1_on_the_corpus_entities_and_relations_are_
// complements, test_cor1_the_aspect_stage_is_a_null_player, test_prop2_errors_add_over_disjoint_triples_f1_does_not,
// test_prop4_shapley_is_linear_so_one_bootstrap_serves_every_share, test_handoffs_on_the_corpus,
// test_thm3_functional_rows_are_totally_unimodular, test_prop3_composition_rules_break_total_unimodularity,
// test_joint_inference_on_the_corpus, test_tone_readers_compared, test_topic_stage_needs_the_fixed_nouns,
// test_filings_fill_what_the_calls_missed). Change the .py, regenerate, never the reverse. Every printed number is a
// payload value.
import payload from './data/capstone-pipeline-lab.json';

/**
 * Capstone Pipeline Lab — five figures for the capstone-earnings-call-nlp topic.
 *   A. "Trace a call": one call through every stage, predicted against gold (Def 1).
 *   B. "Oracle lattice": the 16 oracle coalitions, any order's ceiling analysis beside the Shapley value, the
 *      dividends, and the cluster-bootstrap intervals (Def 2, Theorem 1, Corollary 1, Propositions 2 and 4).
 *   C. "Hard vs. soft": fact P/R/F1 against the threshold for the hard, marginal, and sampled handoffs (Theorem 2).
 *   D. "Joint KB": threshold, merged, functional, and rule-constrained fact selection, and Proposition 3's
 *      fractional vertex (Def 4, Theorem 3).
 *   E. "Readers": three tone readers tested, the topic stage, and the facts the filings fill (case study).
 */

type Mention = [number, number, string, string | null];
type AspectMark = [number, number, string, number];
type Sentence = { tokens: string[]; gold: Mention[]; pred: Mention[]; aspects: AspectMark[]; gold_aspects: AspectMark[]; tone_bearing: boolean };
type Fact = { h: string; r: string; t: string; gold: boolean; pred: boolean; score: number };
type Trace = {
  test: string; doc_id: string; company: string; subject: string | null; tone: number; tones: Record<string, number>;
  aspect_facts: [string, number][]; sentences: Sentence[]; facts: Fact[]; errors: Record<string, boolean>;
};
type Game = {
  f1: Record<string, number>; errors: Record<string, number>; shapley: number[]; shapley_errors: number[];
  dividends: Record<string, number>; orders: Record<string, number[]>; orders_errors: Record<string, number[]>;
  lo: number[]; hi: number[]; p_E_gt_R: number;
};
type PRF = { p: number; r: number; f1: number; n: number };
type Handoff = { grid: PRF[]; at: PRF; ap: number; half: number; at_half: PRF; tuned: number; at_tuned: PRF };
type KBSet = { threshold: PRF[]; merged: PRF[]; functional: PRF[]; rules: PRF[]; fractional: [number, number][]; flips: Record<string, string[]>[] };

const P = payload as unknown as {
  stages: { key: string; name: string }[]; readers: string[]; views: string[]; coalitions: string[]; orders: string[];
  counts: Record<string, { calls: number; companies: number; facts: number; triples: number }>;
  trace: Record<string, Trace>; games: Record<string, Record<string, Game>>;
  bounds: Record<string, Record<string, number>>; tau_grid: number[]; tau: number;
  handoffs: Record<string, Record<string, Handoff>>; kb: Record<string, Record<string, KBSet>>;
  prop3: { names: string[]; A: number[][]; w: number[]; lp_x: number[]; lp: number; ilp: number; ilp_x: number[] };
  readers_compared: { accuracy: Record<string, number>; pairs: { a: string; b: string; n10: number; n01: number; mcnemar: number; cluster_ar: number }[]; by_test: Record<string, Record<string, number>> };
  topic: Record<string, { all: number; fixed: number }>;
  completion: { fact: string; in_filings: boolean; ranks: number[] }[];
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const f4 = (v: number) => v.toFixed(4);
const fp = (p: number) => (p < 1e-4 ? '< 0.0001' : p.toFixed(4));
const STAGE = Object.fromEntries(P.stages.map((s) => [s.key, s.name]));
const TEST_NAMES: Record<string, string> = { seen: 'seen roster', unseen: 'held-out roster', both: 'both runs' };
const READER_NAMES: Record<string, string> = { dictionary: 'dictionary', LR: 'LR', aspects: 'aspects' };
const TONE = (t: number) => (t > 0 ? 'positive' : t < 0 ? 'negative' : 'neutral');
const LINKED = new Set(['ORG', 'PERSON', 'GPE', 'SECTOR']); // the types the knowledge base names (Topic 23's linker)
const TYPE_COLOR: Record<string, string> = {
  ORG: 'var(--color-accent)', PERSON: 'var(--color-badge-blue-text)', GPE: 'var(--color-accent-secondary)',
  SECTOR: 'var(--color-badge-green-text)', PRODUCT: 'var(--color-badge-amber-text)', MONEY: 'var(--color-text-secondary)',
  PERCENT: 'var(--color-text-secondary)', DATE: 'var(--color-text-secondary)',
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

function Key({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{ display: 'inline-block', width: '0.9rem', height: s.dash ? 0 : '0.2rem', borderTop: s.dash ? `2px dashed ${s.color}` : undefined, background: s.dash ? undefined : s.color }} />
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

/** A threshold slider over the payload's grid, driven by its index (arrow keys step one grid point). */
function TauSlider({ k, setK, id }: { k: number; setK: (k: number) => void; id: string }) {
  return (
    <Row>
      <Label>threshold τ</Label>
      <input type="range" min={0} max={P.tau_grid.length - 1} step={1} value={k} data-slider={id} aria-label="threshold"
        onChange={(e) => setK(Number(e.target.value))} style={{ flex: '1 1 10rem', maxWidth: '16rem' }} />
      <span data-readout={`${id}-tau`} style={{ fontFamily: mono, fontSize: '0.8rem' }}>{P.tau_grid[k].toFixed(2)}</span>
    </Row>
  );
}

/** Curves against τ, with a vertical marker at the slider and optional reference ticks. */
function TauChart({ series, k, yLabel, panel, refs }: {
  series: { key: string; label: string; color: string; values: number[]; dash?: boolean }[]; k: number; yLabel: string; panel: string;
  refs?: { at: number; color: string }[];
}) {
  const W = 460, H = 220, m = { l: 36, r: 12, t: 10, b: 34 };
  const all = series.flatMap((s) => s.values);
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.02) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...all) + 0.02) * 20) / 20);
  const x = d3.scaleLinear().domain([P.tau_grid[0], P.tau_grid[P.tau_grid.length - 1]]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([lo, hi]).range([H - m.b, m.t]);
  const line = d3.line<number>().x((_, i) => x(P.tau_grid[i])).y((v) => y(v));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '32rem', height: 'auto', display: 'block' }} role="img" data-panel={panel} aria-label={`${yLabel} against the threshold`}>
      {y.ticks(4).map((v) => (
        <g key={`y${v}`}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke="var(--color-border)" strokeWidth={0.5} />
          <text x={m.l - 4} y={y(v) + 3} textAnchor="end" style={axisText}>{v.toFixed(2)}</text>
        </g>
      ))}
      {[0.1, 0.3, 0.5, 0.7, 0.9].map((v) => <text key={`x${v}`} x={x(v)} y={H - m.b + 13} textAnchor="middle" style={axisText}>{v.toFixed(1)}</text>)}
      <text x={(m.l + W - m.r) / 2} y={H - 2} textAnchor="middle" style={axisText}>{`threshold τ — ${yLabel}`}</text>
      {(refs ?? []).map((r, i) => <line key={`r${i}`} x1={x(r.at)} x2={x(r.at)} y1={m.t} y2={H - m.b} stroke={r.color} strokeDasharray="2 3" strokeWidth={1} />)}
      <line x1={x(P.tau_grid[k])} x2={x(P.tau_grid[k])} y1={m.t} y2={H - m.b} stroke="var(--color-text)" strokeWidth={1} opacity={0.5} />
      {series.map((s) => (
        <g key={s.key} data-series={s.key}>
          <path d={line(s.values) ?? ''} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dash ? '4 3' : undefined} />
          <circle cx={x(P.tau_grid[k])} cy={y(s.values[k])} r={3} fill={s.color} />
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// A. Trace a call.
// ---------------------------------------------------------------------------------------------------------------------

function SentenceView({ s, show }: { s: Sentence; show: 'pred' | 'gold' }) {
  const ms = show === 'pred' ? s.pred : s.gold;
  const asp = show === 'pred' ? s.aspects : s.gold_aspects;
  const same = JSON.stringify([...s.gold].sort()) === JSON.stringify([...s.pred].sort());
  const at = (i: number) => ms.find((m) => m[0] <= i && i < m[1]);
  const aspAt = (i: number) => asp.find((a) => a[0] <= i && i < a[1]);
  return (
    <div data-sentence={same ? 'same' : 'differs'} style={{ display: 'flex', gap: '0.4rem', fontFamily: sans, fontSize: '0.8rem', lineHeight: 1.7, padding: '0.15rem 0', borderBottom: '1px solid var(--color-border)' }}>
      <span aria-label={same ? 'spans agree' : 'spans differ'} title={same ? 'spans agree' : 'predicted spans differ from gold'}
        style={{ flex: '0 0 1rem', color: same ? 'var(--color-border)' : 'var(--color-accent)', fontWeight: 700 }}>{same ? '·' : '≠'}</span>
      <span style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
        {s.tone_bearing && <span style={{ fontSize: '0.65rem', color: 'var(--color-accent-secondary)', marginRight: '0.3rem' }}>[tone]</span>}
        {s.tokens.map((w, i) => {
          const m = at(i), a = aspAt(i);
          const style: CSSProperties = {};
          if (m) {
            style.color = TYPE_COLOR[m[2]];
            style.fontWeight = 600;
            if (m[3] === null && LINKED.has(m[2])) style.textDecoration = 'line-through';
          }
          if (a) {
            style.borderBottom = `2px ${a[3] > 0 ? 'solid' : a[3] < 0 ? 'dotted' : 'dashed'} var(--color-text-secondary)`;
          }
          return <span key={i} style={style} title={m ? `${m[2]}${m[3] ? ` → ${m[3]}` : LINKED.has(m[2]) ? ' (unlinked)' : ''}` : a ? `${a[2]}: ${TONE(a[3])}` : undefined}>{w}{i < s.tokens.length - 1 ? ' ' : ''}</span>;
        })}
      </span>
    </div>
  );
}

function TraceView() {
  const [test, setTest] = useState<'seen' | 'unseen'>('seen');
  const [show, setShow] = useState<'pred' | 'gold'>('pred');
  const t = P.trace[test];
  const bad = t.sentences.filter((s) => JSON.stringify([...s.gold].sort()) !== JSON.stringify([...s.pred].sort())).length;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`One earnings call through every stage: ${t.company} (${TEST_NAMES[test]} test run, ${t.doc_id})`}</Caption>
      <Row>
        <Label>call</Label>
        {(['seen', 'unseen'] as const).map((n) => <Pill key={n} active={test === n} onClick={() => setTest(n)} data={{ 'data-test': n }}>{`${P.trace[n].company} (${TEST_NAMES[n]})`}</Pill>)}
      </Row>
      <Row>
        <Label>spans</Label>
        <Pill active={show === 'pred'} onClick={() => setShow('pred')} data={{ 'data-show': 'pred' }}>predicted</Pill>
        <Pill active={show === 'gold'} onClick={() => setShow('gold')} data={{ 'data-show': 'gold' }}>gold</Pill>
      </Row>
      <div data-panel="trace" style={{ maxHeight: '22rem', overflowY: 'auto', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.3rem 0.5rem' }}>
        {t.sentences.map((s, i) => <SentenceView key={i} s={s} show={show} />)}
      </div>
      <Key items={[
        { label: 'organization', color: TYPE_COLOR.ORG }, { label: 'person', color: TYPE_COLOR.PERSON },
        { label: 'place', color: TYPE_COLOR.GPE }, { label: 'sector', color: TYPE_COLOR.SECTOR },
        { label: 'aspect term (solid +, dotted −, dashed neutral)', color: 'var(--color-text-secondary)' },
      ]} />
      <Readouts>
        <Readout label="sentences whose spans differ" value={`${bad} of ${t.sentences.length}`} id="trace-bad" />
        <Readout label="subject found" value={t.subject ?? 'none (opener missed)'} id="trace-subject" emphasis={t.subject === null} />
        <Readout label="gold tone" value={TONE(t.tone)} id="trace-tone" />
        {P.readers.map((r) => <Readout key={r} label={`${READER_NAMES[r]} reader`} value={TONE(t.tones[r])} id={`trace-${r}`} emphasis={t.tones[r] !== t.tone} />)}
        <Readout label="aspect polarities (predicted)" value={t.aspect_facts.map(([c, s]) => `${c} ${s > 0 ? '+' : '−'}`).join(', ')} id="trace-aspect-facts" />
      </Readouts>
      <Sub>Facts charged to this call</Sub>
      <Table id="trace-facts" head={['fact', 'score', 'extracted', 'gold']}
        rows={t.facts.map((f) => ({
          key: `${f.h}|${f.r}|${f.t}`, strong: f.gold !== f.pred,
          cells: [`${f.h} ${f.r} ${f.t}`, f3(f.score), f.pred ? 'yes' : 'no', f.gold ? 'yes' : 'no'],
        }))} />
      <Note>Bold rows are errors. A struck-through name is a span the tagger found but the linker could not place. Tone-bearing sentences, marked [tone], are the ones the aspect reader votes with.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. Oracle lattice.
// ---------------------------------------------------------------------------------------------------------------------

const ORDER = 'ERTA';
const keyOf = (S: Set<string>) => ORDER.split('').filter((c) => S.has(c)).join('');

function LatticeView() {
  const [view, setView] = useState('both');
  const [reader, setReader] = useState('LR');
  const [metric, setMetric] = useState<'f1' | 'errors'>('f1');
  const [order, setOrder] = useState('ERTA');
  const g = P.games[view][reader];
  const W = 460, H = 250, levels = [0, 1, 2, 3, 4];
  const byLevel = levels.map((l) => P.coalitions.filter((c) => c.length === l));
  const pos: Record<string, [number, number]> = {};
  byLevel.forEach((cs, l) => cs.forEach((c, j) => { pos[c] = [((j + 1) * W) / (cs.length + 1), H - 22 - l * 52]; }));
  const chain: string[] = [''];
  const S = new Set<string>();
  for (const c of order) { S.add(c); chain.push(keyOf(S)); }
  const onChain = (a: string, b: string) => chain.some((c, i) => i > 0 && chain[i - 1] === a && c === b);
  const val = (c: string) => (metric === 'f1' ? f3(g.f1[c]) : `${g.errors[c]}`);
  const seq = metric === 'f1' ? g.orders[order] : g.orders_errors[order];
  const sh = metric === 'f1' ? g.shapley : g.shapley_errors;
  const fmtShare = (v: number) => (metric === 'f1' ? f4(v) : `${v}`);
  // The bar chart of shares: sequential (this order) and Shapley, with bootstrap whiskers for the F1 game.
  const BW = 460, BH = 150, bm = { l: 70, r: 12, t: 6, b: 20 };
  const top = Math.max(...sh, ...seq, ...(metric === 'f1' ? g.hi : [0])) * 1.1 || 1;
  const bx = d3.scaleLinear().domain([0, top]).range([bm.l, BW - bm.r]);
  const rowH = (BH - bm.t - bm.b) / P.stages.length;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`The oracle lattice: ${metric === 'f1' ? 'call-graph F1' : 'errors (false + missing triples)'} with the stages in each coalition replaced by their oracles`}</Caption>
      <Row>
        <Label>calls</Label>
        {P.views.map((v) => <Pill key={v} active={view === v} onClick={() => setView(v)} data={{ 'data-view': v }}>{TEST_NAMES[v]}</Pill>)}
      </Row>
      <Row>
        <Label>tone reader</Label>
        {P.readers.map((r) => <Pill key={r} active={reader === r} onClick={() => setReader(r)} data={{ 'data-reader': r }}>{READER_NAMES[r]}</Pill>)}
      </Row>
      <Row>
        <Label>game</Label>
        <Pill active={metric === 'f1'} onClick={() => setMetric('f1')} data={{ 'data-metric': 'f1' }}>F1</Pill>
        <Pill active={metric === 'errors'} onClick={() => setMetric('errors')} data={{ 'data-metric': 'errors' }}>error count</Pill>
      </Row>
      <Row>
        <Label>order of the ceiling analysis</Label>
        <select value={order} onChange={(e) => setOrder(e.target.value)} data-select="order" aria-label="order of oracles"
          style={{ fontFamily: mono, fontSize: '0.8rem', padding: '0.15rem 0.3rem', border: '1px solid var(--color-border)', background: 'var(--color-surface)', color: 'var(--color-text)' }}>
          {P.orders.map((o) => <option key={o} value={o}>{o.split('').join(' → ')}</option>)}
        </select>
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '32rem', height: 'auto', display: 'block' }} role="img" data-panel="lattice" aria-label="Hasse diagram of the sixteen oracle coalitions">
        {P.coalitions.flatMap((a) => P.coalitions.filter((b) => b.length === a.length + 1 && a.split('').every((c) => b.includes(c))).map((b) => (
          <line key={`${a}-${b}`} x1={pos[a][0]} y1={pos[a][1]} x2={pos[b][0]} y2={pos[b][1]}
            stroke={onChain(a, b) ? 'var(--color-accent)' : 'var(--color-border)'} strokeWidth={onChain(a, b) ? 2.2 : 0.8} />
        )))}
        {P.coalitions.map((c) => (
          <g key={`n${c}`} data-node={c || 'empty'}>
            <rect x={pos[c][0] - 20} y={pos[c][1] - 13} width={40} height={26} rx={4}
              fill={chain.includes(c) ? 'var(--color-definition-bg)' : 'var(--color-surface)'} stroke={chain.includes(c) ? 'var(--color-accent)' : 'var(--color-border)'} />
            <text x={pos[c][0]} y={pos[c][1] - 2} textAnchor="middle" style={{ ...axisText, fill: 'var(--color-text)', fontWeight: 600 }}>{c || '∅'}</text>
            <text x={pos[c][0]} y={pos[c][1] + 9} textAnchor="middle" style={{ ...axisText, fontSize: '8.5px' }}>{val(c)}</text>
          </g>
        ))}
      </svg>
      <Sub>{`Shares: this order's ceiling analysis against the Shapley value${metric === 'f1' ? ' (whiskers: 90% cluster-bootstrap interval over companies)' : ''}`}</Sub>
      <svg viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', maxWidth: '32rem', height: 'auto', display: 'block' }} role="img" data-panel="shares" aria-label="Sequential and Shapley shares per stage">
        {P.stages.map((s, i) => {
          const y0 = bm.t + i * rowH;
          return (
            <g key={s.key} data-stage={s.key}>
              <text x={bm.l - 6} y={y0 + rowH / 2 + 3} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{s.name}</text>
              <rect x={bx(0)} y={y0 + 3} width={Math.max(0, bx(seq[i]) - bx(0))} height={rowH / 2 - 4} fill="var(--color-text-secondary)" opacity={0.55} />
              <rect x={bx(0)} y={y0 + rowH / 2} width={Math.max(0, bx(sh[i]) - bx(0))} height={rowH / 2 - 4} fill="var(--color-accent)" opacity={0.8} />
              {metric === 'f1' && (
                <line x1={bx(g.lo[i])} x2={bx(g.hi[i])} y1={y0 + rowH * 0.75 - 2} y2={y0 + rowH * 0.75 - 2} stroke="var(--color-text)" strokeWidth={1.2} />
              )}
            </g>
          );
        })}
        {bx.ticks(4).map((v) => <text key={v} x={bx(v)} y={BH - 6} textAnchor="middle" style={axisText}>{metric === 'f1' ? v.toFixed(3) : `${v}`}</text>)}
      </svg>
      <Key items={[{ label: `ceiling analysis, ${order.split('').join(' → ')}`, color: 'var(--color-text-secondary)' }, { label: 'Shapley value', color: 'var(--color-accent)' }]} />
      <Table id="shares" head={['stage', 'this order', 'Shapley', ...(metric === 'f1' ? ['90% interval'] : [])]}
        rows={P.stages.map((s, i) => ({
          key: s.key, cells: [s.name, fmtShare(seq[i]), fmtShare(sh[i]), ...(metric === 'f1' ? [`[${f4(g.lo[i])}, ${f4(g.hi[i])}]`] : [])],
        }))} />
      <Readouts>
        <Readout label="dividends of two or more stages" value={Object.keys(g.dividends).length ? Object.entries(g.dividends).map(([k, v]) => `${k} ${f4(v)}`).join(', ') : 'none'} id="lattice-dividends" />
        <Readout label="P(entities’ share > relations’)" value={f3(g.p_E_gt_R)} id="lattice-pE" />
      </Readouts>
      <Note>{metric === 'f1'
        ? 'Each node is a coalition of oracles; a ceiling analysis climbs one path. A stage’s share in an order is the sum of the dividends of the coalitions it completes (Theorem 1(a)), so only the ER dividend makes orders disagree much. The aspect stage is a null player: its share is 0 in every order (Corollary 1). The dividends are those of the F1 game; the small ones with tone come from F1’s denominator (Proposition 2).'
        : 'In the error game the shares are counts of errors removed. Stages that write different triples add exactly (Proposition 2), so the only interaction is entities × relations, and every order agrees on tone and aspects.'}</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Hard vs. soft.
// ---------------------------------------------------------------------------------------------------------------------

const HANDOFFS = [
  { key: 'hard', label: 'hard (Viterbi spans)', color: 'var(--color-text)' },
  { key: 'marginal', label: 'marginal (q-weighted spans)', color: 'var(--color-accent)' },
  { key: 'sampled', label: 'sampled (64 span sets)', color: 'var(--color-badge-blue-text)' },
] as const;

function HandoffView() {
  const [test, setTest] = useState<'seen' | 'unseen'>('seen');
  const [stat, setStat] = useState<'f1' | 'p' | 'r'>('f1');
  const [k, setK] = useState(P.tau_grid.indexOf(P.tau));
  const h = P.handoffs[test];
  const statName = { f1: 'F1', p: 'precision', r: 'recall' }[stat];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Fact ${statName} on the ${TEST_NAMES[test]} calls when the entity stage hands on its argmax, its marginals, or samples`}</Caption>
      <Row>
        <Label>calls</Label>
        {(['seen', 'unseen'] as const).map((n) => <Pill key={n} active={test === n} onClick={() => setTest(n)} data={{ 'data-test': n }}>{TEST_NAMES[n]}</Pill>)}
      </Row>
      <Row>
        <Label>curve</Label>
        {(['f1', 'p', 'r'] as const).map((s) => <Pill key={s} active={stat === s} onClick={() => setStat(s)} data={{ 'data-stat': s }}>{{ f1: 'F1', p: 'precision', r: 'recall' }[s]}</Pill>)}
      </Row>
      <TauSlider k={k} setK={setK} id="handoff" />
      <TauChart k={k} yLabel={statName} panel="handoffs"
        series={HANDOFFS.map((x, i) => ({ key: x.key, label: x.label, color: x.color, values: h[x.key].grid.map((g) => g[stat]), dash: i === 2 }))}
        refs={[{ at: P.tau, color: 'var(--color-text-secondary)' }]} />
      <Key items={HANDOFFS.map((x, i) => ({ label: x.label, color: x.color, dash: i === 2 }))} />
      <Table id="handoffs" head={['handoff', `P at τ`, `R at τ`, `F1 at τ`, 'AP', 'F*/2', 'F1 there', 'τ tuned on filings', 'F1 there']}
        rows={HANDOFFS.map((x) => {
          const v = h[x.key];
          return { key: x.key, cells: [x.key, f3(v.grid[k].p), f3(v.grid[k].r), f3(v.grid[k].f1), f3(v.ap), f3(v.half), f3(v.at_half.f1), v.tuned.toFixed(2), f3(v.at_tuned.f1)] };
        })} />
      <Note>Scaling each pair by its spans’ marginals pushes true facts below the hard pipeline’s 0.5 (dotted line), so soft handoffs lose there and win AP (Theorem 2(b), then 2(a)). Refit the threshold, on the same run’s filings, and on the seen calls they win F1 too. The F*/2 rule assumes calibrated scores, and noisy-OR’s soft scores are underconfident.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. Joint KB.
// ---------------------------------------------------------------------------------------------------------------------

const METHODS = [
  { key: 'threshold', label: 'threshold', color: 'var(--color-text-secondary)' },
  { key: 'merged', label: 'merged inverses', color: 'var(--color-badge-blue-text)' },
  { key: 'functional', label: '+ functional rows', color: 'var(--color-accent)' },
  { key: 'rules', label: '+ composition rules', color: 'var(--color-accent-secondary)' },
] as const;
const FLIPS = [
  { key: 'merge_added', label: 'merging adds' }, { key: 'removed', label: 'functional rows remove' },
  { key: 'rule_removed', label: 'rules remove' }, { key: 'rule_added', label: 'rules add' },
] as const;

function KBView() {
  const [test, setTest] = useState<'seen' | 'unseen'>('seen');
  const [spans, setSpans] = useState<'predicted' | 'gold'>('predicted');
  const [k, setK] = useState(P.tau_grid.indexOf(P.tau));
  const kb = P.kb[test][spans];
  const pr = P.prop3;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>{`Fact F1 on the ${TEST_NAMES[test]} calls (${spans} mentions): thresholding, merging inverse facts, and the knowledge base’s constraints`}</Caption>
      <Row>
        <Label>calls</Label>
        {(['seen', 'unseen'] as const).map((n) => <Pill key={n} active={test === n} onClick={() => setTest(n)} data={{ 'data-test': n }}>{TEST_NAMES[n]}</Pill>)}
      </Row>
      <Row>
        <Label>mentions</Label>
        {(['predicted', 'gold'] as const).map((s) => <Pill key={s} active={spans === s} onClick={() => setSpans(s)} data={{ 'data-spans': s }}>{s}</Pill>)}
      </Row>
      <TauSlider k={k} setK={setK} id="kb" />
      <TauChart k={k} yLabel="F1" panel="kb" series={METHODS.map((m, i) => ({ key: m.key, label: m.label, color: m.color, values: kb[m.key].map((x) => x.f1), dash: i === 1 }))} />
      <Key items={METHODS.map((m, i) => ({ label: m.label, color: m.color, dash: i === 1 }))} />
      <Readouts>
        {METHODS.map((m) => <Readout key={m.key} label={`${m.label}: F1`} value={f3(kb[m.key][k].f1)} id={`kb-${m.key}`} emphasis={m.key === 'functional'} />)}
        <Readout label="fractional LP variables (functional, with rules)" value={`${kb.fractional[k][0]}, ${kb.fractional[k][1]}`} id="kb-frac" />
      </Readouts>
      <Sub>{`What each step changes at τ = ${P.tau_grid[k].toFixed(2)}`}</Sub>
      <div data-panel="flips" style={{ fontFamily: sans, fontSize: '0.75rem' }}>
        {FLIPS.map((f) => (
          <div key={f.key} style={{ marginBottom: '0.25rem', overflowWrap: 'anywhere' }}>
            <span style={{ color: 'var(--color-text-secondary)' }}>{`${f.label}: `}</span>
            <span data-flips={f.key}>{kb.flips[k][f.key].length ? kb.flips[k][f.key].join('; ') : 'nothing'}</span>
          </div>
        ))}
      </div>
      <Sub>Proposition 3: two headquarters candidates, one country, and the composition rules</Sub>
      <Table id="prop3" head={['', ...pr.names, 'value']}
        rows={[
          { key: 'w', cells: ['weight', ...pr.w.map((v) => `${v}`), ''] },
          { key: 'lp', strong: true, cells: ['LP optimum', ...pr.lp_x.map((v) => `${v}`), f3(pr.lp)] },
          { key: 'ilp', cells: ['best integer point', ...pr.ilp_x.map((v) => `${v}`), f3(pr.ilp)] },
        ]} />
      <Note>The functional rows are totally unimodular (Theorem 3), so their LP is the MAP. The composition rules are not: the LP splits the headquarters between two cities and gains ½ (Proposition 3). On the calls the LP is integral anyway, at every threshold. The rules delete true LOCATED_IN facts whose BASED_IN leg no call states: the closed world.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// E. Readers.
// ---------------------------------------------------------------------------------------------------------------------

function ReadersView() {
  const rc = P.readers_compared;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Three tone readers on the 126 calls, the topic stage, and what the filings fill in</Caption>
      <Table id="reader-acc" head={['reader', 'all calls', 'seen roster', 'held-out roster']}
        rows={P.readers.map((r) => ({ key: r, cells: [READER_NAMES[r], f3(rc.accuracy[r]), f3(rc.by_test.seen[r]), f3(rc.by_test.unseen[r])] }))} />
      <Sub>Paired tests on the same calls</Sub>
      <Table id="reader-tests" head={['pair', 'n₁₀ / n₀₁', 'McNemar p', 'companies swapped, p']}
        rows={rc.pairs.map((x) => ({ key: `${x.a}|${x.b}`, cells: [`${READER_NAMES[x.a]} vs. ${READER_NAMES[x.b]}`, `${x.n10} / ${x.n01}`, fp(x.mcnemar), fp(x.cluster_ar)] }))} />
      <Sub>Topic stage: the dominant topic matches the call’s sector</Sub>
      <Table id="topic" head={['calls', 'all words', 'fixed nouns only']}
        rows={(['seen', 'unseen'] as const).map((n) => ({ key: n, cells: [TEST_NAMES[n], f3(P.topic[n].all), f3(P.topic[n].fixed)] }))} />
      <Sub>Facts the held-out calls state and the pipeline misses</Sub>
      <Table id="completion" head={['fact', 'in the filings’ KB', 'RotatE ranks']}
        rows={P.completion.map((c) => ({ key: c.fact, cells: [c.fact, c.in_filings ? 'yes' : 'inverse is', c.ranks.join(', ')] }))} />
      <Note>Swapping all of a company’s calls at once widens the LR-versus-dictionary p-value tenfold: the document-level readers err together on one company’s calls. Each generator run invents its own pseudo-words, so the topic stage folds in only the 80 seeded nouns. Every fact the calls miss is in the filings’ knowledge base or is the inverse of one that is.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------

const VIEWS = [
  { key: 'trace', label: 'Trace a call' },
  { key: 'lattice', label: 'Oracle lattice' },
  { key: 'handoff', label: 'Hard vs. soft' },
  { key: 'kb', label: 'Joint KB' },
  { key: 'readers', label: 'Readers' },
] as const;

export default function CapstonePipelineLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('trace');
  return (
    <div data-lab="capstone-pipeline" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'trace' && <TraceView />}
      {view === 'lattice' && <LatticeView />}
      {view === 'handoff' && <HandoffView />}
      {view === 'kb' && <KBView />}
      {view === 'readers' && <ReadersView />}
    </div>
  );
}
